from datasets import load_dataset
import json
import os
import torch
import torch.nn.functional as F

from circuit_tracer_import import attribute, ReplacementModel
from device_setup import device
from intervention import get_top_outputs
from template import lang_to_flores_key, langs_big

def get_activation_vector(
        prompt: str, 
        model: ReplacementModel,
        n_layers = 26,
        n_features = 16384,
        max_n_logits = 5,
        desired_logit_prob = 0.95,
        max_feature_nodes = None,
        batch_size = 64,
        offload = 'cpu',
        verbose = True,
        ) -> tuple[torch.Tensor, int, dict[str, float]]:
    graph = attribute(
            prompt=prompt,
            model=model,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            verbose=verbose,
        )
    active_features = graph.active_features # (n_active_features, 3) containing (layer, pos, feature_idx)
    activation_values = graph.activation_values
    activation_vector = torch.zeros((n_layers, n_features))
    n_pos = graph.n_pos
    values_dict = dict()
    for i, (layer, pos, feature_idx) in enumerate(active_features):
        layer = int(layer) if isinstance(layer, torch.Tensor) else layer
        feature_idx = int(feature_idx) if isinstance(feature_idx, torch.Tensor) else feature_idx
        activation_vector[layer, feature_idx] += 1
        
        key = f"{layer}.{feature_idx}"
        cur = values_dict.get(key, 0)
        activation_value = activation_values[i]
        activation_value = activation_value.item() if isinstance(activation_value, torch.Tensor) else activation_value
        values_dict[key] = max(cur, activation_value)

    return activation_vector, n_pos, values_dict

def get_lang_activation_vector(
        prompts: list[str],
        model: ReplacementModel,
        n_layers = 26, n_features=16384,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    activation_vector = torch.zeros((n_layers, n_features))
    active_examples = torch.zeros((n_layers, n_features))
    n_pos_total = 0
    max_values_dict = dict()
    for prompt in prompts:
        vec, n_pos, values_dict = get_activation_vector(prompt, model, n_layers, n_features)
        activation_vector += vec
        n_pos_total += n_pos
        bool_mask = (activation_vector > 0).float()
        active_examples += bool_mask

        for key, val in values_dict.items():
            cur = max_values_dict.get(key, 0)
            max_values_dict[key] = max(cur, val)

    activation_vector /= n_pos_total
    active_examples /= len(prompts)
    return activation_vector, active_examples, max_values_dict

def normalize(lang_activation_vec: dict[str, torch.Tensor]) -> tuple[list[str], torch.Tensor]:
    # the returned tensor is (n_layers, n_features, langs)
    tensor_list = []
    lang_list = []
    for lang, tensor in lang_activation_vec.items():
        tensor_list.append(tensor)
        lang_list.append(lang)
    stacked = torch.stack(tensor_list, dim=-1)
    normalized = F.normalize(stacked, p=1, dim=-1)
    return lang_list, normalized

def choose_language_specific_features(
        langs: list[str], 
        active_tokens: dict[str, torch.Tensor], 
        active_examples: dict[str, torch.Tensor],
        cross_lingual_thres: float,
        example_thres: float,
        token_thres: float = 0.1,
        ) -> dict[str, list[tuple[int, int]]]:
    active_features = torch.zeros_like(active_tokens[langs[0]]).bool()
    for lang in langs:
        tokens_mask = (active_tokens[lang] > token_thres)
        examples_mask = (active_examples[lang] > example_thres)
        result_mask = tokens_mask & examples_mask
        active_features = active_features | result_mask
    true_indices = torch.nonzero(active_features)

    tensor_list = []
    for lang in langs:
        tensor_list.append(active_tokens[lang])
    stacked = torch.stack(tensor_list, dim=-1)
    
    language_specific_features = dict()
    for lang in langs:
        language_specific_features[lang] = []

    for (layer, feature_idx) in true_indices:
        layer = layer.item() if isinstance(layer, torch.Tensor) else layer
        feature_idx = feature_idx.item() if isinstance(feature_idx, torch.Tensor) else feature_idx
        vals = stacked[layer, feature_idx, :]
        max_val = max(vals)
        active = (vals >= max_val * cross_lingual_thres)
        indices = torch.nonzero(active)
        if len(indices) > 1:
            continue # not specific
        else:
            lang = langs[indices[0]]
            language_specific_features[lang].append((layer, feature_idx))
    return language_specific_features

def scale_steer_to_A(
        language_features: dict[str, list[tuple[int, int]]],
        max_activation: dict[str, dict[str, float]],
        lang_A: str,
        prompt: str,
        model: ReplacementModel,
        alpha = 0.2,
        max_new_tokens = 64,
        max_n_logits = 5,
        desired_logit_prob = 0.95,
        max_feature_nodes = None,
        batch_size = 64,
        offload = 'cpu',
        verbose = True,
        ) -> str:
    lang_A_features = language_features[lang_A]

    generated = prompt
    for _ in range(max_new_tokens):
        graph = attribute(
                prompt=generated,
                model=model,
                max_n_logits=max_n_logits,
                desired_logit_prob=desired_logit_prob,
                batch_size=batch_size,
                max_feature_nodes=max_feature_nodes,
                offload=offload,
                verbose=verbose,
            )
        active_features = graph.active_features # (n_active_features, 3) containing (layer, pos, feature_idx)
        active_features = active_features.detach().cpu()
        activation_values = graph.activation_values
        n_pos = graph.n_pos

        interventions = []
        for layer, feature_idx in lang_A_features:
            pos = -1
            target_row = torch.tensor((layer, pos, feature_idx))

            matches = (active_features == target_row)
            row_matches_all = torch.all(matches, dim=1)
            indices = torch.nonzero(row_matches_all, as_tuple=False)
            original_activation = 0
            if indices.numel() > 0:
                index = indices.item()
                print(f"{layer}.{feature_idx} was active in prompt {prompt}")
                original_activation = activation_values[index].detach().cpu()
                original_activation = original_activation.item() if isinstance(original_activation, torch.Tensor) else original_activation
            
            activation_value = original_activation + alpha * max_activation[lang_A][f"{layer}.{feature_idx}"]
            # tuple of layer, position, feature_idx, value
            intervention = (layer, pos, feature_idx, activation_value)
            interventions.append(intervention)
        
        new_logits, new_activations = model.feature_intervention(prompt, interventions)
        token, prob = get_top_outputs(new_logits, model, 1)[0]
        generated += token
        print(generated)

    return generated

if __name__ == "__main__":
    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    data_directory = os.path.join(absolute_directory, "data/language_specific_features")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    model_name = 'google/gemma-2-2b'
    transcoder_name = "gemma"
    model = ReplacementModel.from_pretrained(model_name, transcoder_name, device=device, dtype=torch.bfloat16)

    lang_activation_vec = dict()
    lang_active_examples = dict()
    lang_max_vals = dict()
    for lang, ds_key in lang_to_flores_key.items():
        file_name = f"{lang}.pt"
        json_name = f"{lang}.json" # for max activation values
        pt_path = os.path.join(data_directory, file_name)
        json_path = os.path.join(data_directory, json_name)
        if os.path.exists(pt_path) and os.path.exists(json_path):
            activation_vector, active_examples = torch.load(pt_path)
            lang_activation_vec[lang] = activation_vector
            lang_active_examples[lang] = active_examples
            with open(json_path, 'r') as f:
                lang_max_vals[lang] = json.load(f)
        else:
            ds = load_dataset("openlanguagedata/flores_plus", ds_key, split="dev")
            ds = ds.shuffle(seed=42)
            df = ds.to_pandas()
            batch = df.loc[:100, 'text'].tolist()
            activation_vector, active_examples, max_values_dict = get_lang_activation_vector(batch, model)
            torch.save((activation_vector, active_examples), pt_path)
            lang_activation_vec[lang] = activation_vector
            lang_active_examples[lang] = active_examples
            with open(json_path, 'w') as f:
                json.dump(max_values_dict, f)
            lang_max_vals[lang] = max_values_dict
    
    example_thres = 0.98
    file_name = f"features_{example_thres}.json"
    full_path = os.path.join(data_directory, file_name)
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            language_specific_features = json.load(f)
    else:
        language_specific_features = choose_language_specific_features(
            list(langs_big), lang_activation_vec, lang_active_examples, 0.8, example_thres
        ) # example_thres is 0.98 for the original paper
        with open(full_path, 'w') as f:
            json.dump(language_specific_features, f)
    
    data_list = []
    alphas = [0.1, 0.3, 0.4, 0.5, 0.8]
    for lang in langs_big:
        for alpha in alphas:
            output = scale_steer_to_A(language_specific_features, lang_max_vals, lang, "", model, alpha)
            record = [lang, alpha, output]
            data_list.append(record)
    
    file_name = "text_generation.jsonl"
    full_path = os.path.join(data_directory, file_name)
    with open(full_path, 'w', encoding='utf-8') as file:
        for record in data_list:
            json_line = json.dumps(record, ensure_ascii=False)
            file.write(json_line + '\n')
