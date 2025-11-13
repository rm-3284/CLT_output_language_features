from datasets import load_dataset
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch

from circuit_tracer_import import Graph, attribute, ReplacementModel
from device_setup import device
from template import lang_to_flores_key

def get_activation(
        prompt: str, 
        model: ReplacementModel,
        max_n_logits = 5,
        desired_logit_prob = 0.95,
        max_feature_nodes = None,
        batch_size = 256,
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
    
    sorted_items = sorted(data.items(), key=lambda item: item[1], reverse=True)
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
        f"Max (Green): {max_val:.2f} | Min (Red): {min_val:.2f} | Difference: {difference:.2f}"
    )
    if len(labels) > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    plt.savefig(save_path)
    print(f"Bar chart saved to {save_path}")
    return

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
    
    v_dict = calculate_v(lang_mean_activation_dict)
    with open(os.path.join(data_directory, 'v_values'), 'w') as f:
        json.dump(v_dict, f)

    for lang, data in lang_mean_activation_dict.items():
        file_name = "{lang}_vplot.png"
        full_path = os.path.join(data_directory, )
    
