from datasets import load_dataset
import json
import os
import torch
import torch.nn.functional as F

from circuit_tracer_import import attribute, ReplacementModel
from device_setup import device
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
        ) -> tuple[torch.Tensor, int]:
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
    activation_vector = torch.zeros((n_layers, n_features))
    n_pos = graph.n_pos
    for layer, pos, feature_idx in active_features:
        layer = int(layer) if isinstance(layer, torch.Tensor) else layer
        feature_idx = int(feature_idx) if isinstance(feature_idx, torch.Tensor) else feature_idx
        activation_vector[layer, feature_idx] += 1
    return activation_vector, n_pos

def get_lang_activation_vector(
        prompts: list[str],
        model: ReplacementModel,
        n_layers = 26, n_features=16384,
) -> tuple[torch.Tensor, torch.Tensor]:
    activation_vector = torch.zeros((n_layers, n_features))
    active_examples = torch.zeros((n_layers, n_features))
    n_pos_total = 0
    for prompt in prompts:
        vec, n_pos = get_activation_vector(prompt, model, n_layers, n_features)
        activation_vector += vec
        n_pos_total += n_pos
        bool_mask = (activation_vector > 0).float()
        active_examples += bool_mask
    activation_vector /= n_pos_total
    active_examples /= len(prompts)
    return activation_vector, active_examples

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
    for lang, ds_key in lang_to_flores_key.items():
        file_name = f"{lang}.pt"
        full_path = os.path.join(data_directory, file_name)
        if os.path.exists(full_path):
            activation_vector, active_examples = torch.load(full_path)
            lang_activation_vec[lang] = activation_vector
            lang_active_examples[lang] = active_examples
        else:
            ds = load_dataset("openlanguagedata/flores_plus", ds_key, split="dev")
            ds = ds.shuffle(seed=42)
            df = ds.to_pandas()
            batch = df.loc[:100, 'text'].tolist()
            activation_vector, active_examples = get_lang_activation_vector(batch, model)
            torch.save((activation_vector, active_examples), full_path)
            lang_activation_vec[lang] = activation_vector
            lang_active_examples[lang] = active_examples
    
    example_thres = 0.9
    language_specific_features = choose_language_specific_features(
        list(langs_big), lang_activation_vec, lang_active_examples, 0.8, example_thres
    ) # example_thres is 0.98 for the original paper
    file_name = f"features_{example_thres}.json"
    with open(os.path.join(data_directory, file_name), 'w') as f:
        json.dump(language_specific_features, f)
    
