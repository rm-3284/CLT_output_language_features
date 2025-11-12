from datasets import load_dataset
import json
import torch
import os
import pandas as pd

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
    
    for key, val in mean_activation_dict.items():
        mean_activation_dict[key] = val / n_pos_total
    return mean_activation_dict

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
    for lang, ds_key in lang_to_flores_key.items():
        ds = load_dataset("openlanguagedata/flores_plus", ds_key, split="dev")
        ds = ds.shuffle(seed=42)
        df = ds.to_pandas()
        batch = df.loc[:100, 'text'].tolist()
        mean_activation = get_mean_activation(batch, model)
        file_name = f"{lang}.json"
        with open(os.path.join(data_directory, file_name), 'w') as f:
            json.dump(mean_activation, f)
