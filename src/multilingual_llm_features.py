from datasets import load_dataset
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch

from circuit_tracer_import import attribute, ReplacementModel
from device_setup import device
from intervention import get_top_outputs
from template import lang_to_flores_key

def get_activation(
        prompt: str, 
        model: ReplacementModel,
        max_n_logits = 5,
        desired_logit_prob = 0.95,
        max_feature_nodes = None,
        batch_size = 64,
        offload = 'cpu',
        verbose = True,
        ) -> tuple[dict[str, float], int]:
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
    n_active_features = active_features.shape[0]
    activation_values = graph.activation_values
    n_pos = graph.n_pos
    activation_values_sum = dict()
    for i in range(n_active_features):
        layer, pos, feature_idx = active_features[i, :]
        layer = layer.item() if isinstance(layer, torch.Tensor) else layer
        feature_idx = feature_idx.item() if isinstance(feature_idx, torch.Tensor) else feature_idx
        key = f"{layer}.{feature_idx}"
        activation_value = activation_values[i]
        activation_value = activation_value.item() if isinstance(activation_value, torch.Tensor) else activation_value
        try:
            val = activation_values_sum[key]
            activation_values_sum[key] = val + activation_value
        except KeyError:
            activation_values_sum[key] = activation_value
    for key, val in activation_values_sum.items():
        activation_values_sum[key] = val.item() if isinstance(val, torch.Tensor) else val
    del graph
    return activation_values_sum, n_pos

def get_mean_activation(
        prompts: list[str],
        model: ReplacementModel,
) -> dict[str, float]:
    mean_activation_dict = dict()
    n_pos_total = 0
    for prompt in prompts:
        activation_values_sum, n_pos = get_activation(prompt, model)
        n_pos_total += n_pos
        for key, val in activation_values_sum.items():
            try:
                current_val = mean_activation_dict[key]
                current_val = current_val.item() if isinstance(current_val, torch.Tensor) else current_val
                mean_activation_dict[key] = current_val + val
            except KeyError:
                mean_activation_dict[key] = val
        torch.cuda.empty_cache()
    
    for key, val in mean_activation_dict.items():
        mean_activation_dict[key] = val / n_pos_total
    return mean_activation_dict

def calculate_v(per_lang_mean_activation_dict: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = per_lang_mean_activation_dict.keys()
    v_dict = dict()
    for lang, sub_dict in per_lang_mean_activation_dict.items():
        v_dict[lang] = dict()
        for feature, val in sub_dict.items():
            gamma = 0
            for key in keys:
                if key == lang:
                    continue
                try:
                    v = per_lang_mean_activation_dict[key][feature]
                    gamma += v
                except KeyError:
                    gamma += 0
            gamma /= len(keys) - 1
            v = val - gamma
            v_dict[lang][feature] = v
    return v_dict

def histogram_v_values(data: dict[str, float], save_path: str):
    if not data:
        print("The dictionary is empty. No plot to generate.")
        return
    
    sorted_items = sorted(data.items(), key=lambda item: item[1], reverse=True)[:100]
    labels, values = zip(*sorted_items)
    if len(values) == 1:
        # If there's only one bar, it's both max and min
        colors = ['blue']
        max_val = values[0]
        min_val = values[0]
    else:
        # Default color 'gray' for all bars
        colors = ['#C0C0C0'] * len(labels)  # Using a hex code for standard gray
        # Set the first bar (max) to green
        colors[0] = '#2ca02c'  # Default matplotlib green
        # Set the last bar (min) to red
        colors[-1] = '#d62728' # Default matplotlib red
        
        max_val = values[0]
        min_val = values[-1]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=colors)
    plt.xlabel("Categories")
    plt.ylabel("Values")
    plt.title(
        f"Bar Chart in Decreasing Order\n"
        f"Max (Green): {max_val:.2f} | Min (Red): {min_val:.2f}"
    )
    if len(labels) > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    plt.savefig(save_path)
    print(f"Bar chart saved to {save_path}")
    return

def last_pos_feature_find(key: str, n_pos: int, active_features: torch.Tensor) -> int:
    pos = n_pos - 1
    layer, feature_idx = key.split('.')
    layer = int(layer)
    feature_idx = int(feature_idx)

    target_row = torch.tensor((layer, pos, feature_idx))
    matches = (active_features == target_row)
    row_matches_all = torch.all(matches, dim=1)
    indices = torch.nonzero(row_matches_all, as_tuple=False)

    if indices.numel() == 0:
        return -1
    else:
        return indices.item()


def steering_from_A_to_B(
        per_lang_mean_activation_dict: dict[str, dict[str, float]], 
        lang_A: str, # source
        lang_B: str, # target
        prompt: str,
        model: ReplacementModel,
        topk = 10,
        max_n_logits = 5,
        desired_logit_prob = 0.95,
        max_feature_nodes = None,
        batch_size = 64,
        offload = 'cpu',
        verbose = True,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    intervening_features1 = list(sorted(per_lang_mean_activation_dict[lang_A].items(), key=lambda item: item[1], reverse=True)[:topk])
    intervening_features2 = list(sorted(per_lang_mean_activation_dict[lang_B].items(), key=lambda item: item[1], reverse=True)[:topk])
    combined_intervening_features = intervening_features1 + intervening_features2

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
    active_features = active_features.detach().cpu()
    activation_values = graph.activation_values
    n_pos = graph.n_pos

    interventions = list()
    for key, _ in combined_intervening_features:
        index = last_pos_feature_find(key, n_pos, active_features)
        if index == -1:
            original_activation = 0
        else:
            original_activation = activation_values[index].detach().cpu()
            original_activation = original_activation.item() if isinstance(original_activation, torch.Tensor) else original_activation

        langA_activation = per_lang_mean_activation_dict[lang_A].get(key, 0)
        langB_activation = per_lang_mean_activation_dict[lang_B].get(key, 0)
        diff = langB_activation - langA_activation

        # tuple of layer, position, feature_idx, value
        intervention = (layer, pos, feature_idx, original_activation + diff)
        interventions.append(intervention)
    
    new_logits, new_activations = model.feature_intervention(prompt, interventions)
    return new_logits, new_activations

def code_switch_analysis(
        per_lang_mean_activation_dict: dict[str, dict[str, float]], 
        lang_A,
        lang_B,
        prompt_list,
        model: ReplacementModel,
        topk = 10,
        max_n_logits = 5,
        desired_logit_prob = 0.95,
        max_feature_nodes = None,
        batch_size = 64,
        offload = 'cpu',
        verbose = True,
        ) -> dict[str, tuple[float, float, float]]: 
        # lang_B noun, lang_A prefix lang_B noun, lang_A prefix lang_A noun
    top_features = list(sorted(per_lang_mean_activation_dict[lang_A].items(), key=lambda item: item[1], reverse=True)[:topk])
    activation_diff = dict()
    for key, _ in top_features:
        activation_diff[key] = [[], [], []]
    
    for prompt in prompt_list:
        ori_lan = prompt["ori_lan"]
        target_lan = prompt["target_lan"]
        ori_sentence = prompt["ori_sentence"]
        sentence = prompt["sentence"]
        if ori_lan != lang_A:
            continue
        if target_lan != lang_A and target_lan != lang_B:
            continue
        graph = attribute(
                prompt=sentence,
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
        if target_lan == lang_A:
            for key, val in activation_diff.items():
                index = last_pos_feature_find(key, n_pos, active_features)
                if index == -1:
                    val[2].append(0)
                else:
                    activation_value = activation_values[index]
                    activation_value = activation_value.item() if isinstance(activation_value, torch.Tensor) else activation_value
                    val[2].append(activation_value)
        elif target_lan == lang_B:
            for key, val in activation_diff.items():
                index = last_pos_feature_find(key, n_pos, active_features)
                if index == -1:
                    val[1].append(0)
                else:
                    activation_value = activation_values[index]
                    activation_value = activation_value.item() if isinstance(activation_value, torch.Tensor) else activation_value
                    val[1].append(activation_value)
            
            # lang_B noun
            prompt_inputs = model.tokenizer.encode(sentence)
            ori_prompt_inputs = model.tokenizer.encode(ori_sentence)
            noun = prompt_inputs[:1] + prompt_inputs[len(ori_prompt_inputs):]
            graph = attribute(
                prompt=sentence,
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

            for key, val in activation_diff.items():
                index = last_pos_feature_find(key, n_pos, active_features)
                if index == -1:
                    val[0].append(0)
                else:
                    activation_value = activation_values[index]
                    activation_value = activation_value.item() if isinstance(activation_value, torch.Tensor) else activation_value
                    val[0].append(activation_value)

    result = dict()
    for key, val in activation_diff.items():
        list1, list2, list3 = val
        mean1 = sum(list1) / len(list1)
        mean2 = sum(list2) / len(list2)
        mean3 = sum(list3) / len(list3)
        result[key] = (mean1, mean2, mean3)
    return result


if __name__ == "__main__":
    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    data_directory = os.path.join(absolute_directory, "data/multilingual_llm_features")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    model_name = 'google/gemma-2-2b'
    transcoder_name = "gemma"
    model = ReplacementModel.from_pretrained(model_name, transcoder_name, device=device, dtype=torch.bfloat16)

    lang_mean_activation_dict = dict()
    for lang, ds_key in lang_to_flores_key.items():
        file_name = f"{lang}.json"
        full_path = os.path.join(data_directory, file_name)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                mean_activation = json.load(f)
            lang_mean_activation_dict[lang] = mean_activation
        else:
            ds = load_dataset("openlanguagedata/flores_plus", ds_key, split="dev")
            ds = ds.shuffle(seed=42)
            df = ds.to_pandas()
            batch = df.loc[:100, 'text'].tolist()
            mean_activation = get_mean_activation(batch, model)
            with open(full_path, 'w') as f:
                json.dump(mean_activation, f)
            lang_mean_activation_dict[lang] = mean_activation
    
    full_path = os.path.join(data_directory, 'v_values.json')
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            v_dict = json.load(f)
    else:
        v_dict = calculate_v(lang_mean_activation_dict)
        with open(full_path, 'w') as f:
            json.dump(v_dict, f)

    for lang, data in lang_mean_activation_dict.items():
        file_name = f"{lang}_vplot.png"
        full_path = os.path.join(data_directory, file_name)
        if not os.path.exists(full_path):
            histogram_v_values(data, full_path)
    
    # steering experiments
    full_path = os.path.join(data_directory, 'forced_code_switch.jsonl')
    prompt_list = []
    with open(full_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                json_object = json.loads(line.strip())
                prompt_list.append(json_object)
    
    #data_list = list()
    topk = 50
    """
    for prompt in prompt_list:
        ori_sentence = prompt["ori_sentence"]
        ori_lan = prompt["ori_lan"]
        target_lan = prompt["target_lan"]
        if not (ori_lan in lang_to_flores_key.keys() and target_lan in lang_to_flores_key.keys()):
            continue
        logits, activation = steering_from_A_to_B(v_dict, ori_lan, target_lan, ori_sentence, model, topk)
        top_outputs = get_top_outputs(logits, model)
        record = {"sentence": ori_sentence, "source_lang": ori_lan, "target_lang": target_lan, "top_outputs": top_outputs}
        data_list.append(record)
    
    file_name = f"cross_lingual_continuation_{topk}.jsonl"
    full_path = os.path.join(data_directory, file_name)
    with open(full_path, 'w', encoding='utf-8') as file:
        for record in data_list:
            json_line = json.dumps(record, ensure_ascii=False)
            file.write(json_line + '\n')
    """

    for lang_A in lang_to_flores_key.keys():
        for lang_B in lang_to_flores_key.keys():
            if lang_A == lang_B:
                continue
            result = code_switch_analysis(v_dict, lang_A, lang_B, prompt_list, model, topk)
            file_name = f"code_switch_analysis_{lang_A}_{langB}.json"
            full_path = os.path.join(data_directory, file_name)
            with open(full_path, 'w') as f:
                json.dump(result, f)
