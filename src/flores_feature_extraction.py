from datasets import load_dataset
import json
import pandas as pd
import torch
import os

from circuit_tracer_import import ReplacementModel, attribute
from data.generic_sentences import alphabet_char, filter_sentences
from device_setup import device
from feature_extraction import distinct_path_max_bottleneck, prune_paths_by_first_last, pick_last_pos_features
from template import lang_to_flores_key

def iterate_through_sentences(
        model: ReplacementModel,
        sentences: list[str],
        logit_focus: list[int] = [0],
        throughput_threshold: float = 0.1,
        node_threshold: float = 0.8, edge_threshold: float = 0.98,
        MAX_ITERATIONS: int = 75,
        threshold_first = 0.5, threshold_last = 0.25,
        max_n_logits = 5, desired_logit_prob = 0.95,
        max_feature_nodes = None, batch_size = 256,
        offload = 'cpu', verbose = True,
        ) -> tuple[list[tuple[int, int]], dict[str, list[float]]]:
    features = []
    activation_dict = dict()
    for prompt in sentences:
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
        active_features = graph.active_features
        activation_values = graph.activation_values
        n_pos = graph.n_pos
        paths = []
        decoded = model.tokenizer.encode(prompt)
        for pos in range (1, len(decoded)): # the first one is <bos>
            path = []
            for logit in logit_focus:
                p = distinct_path_max_bottleneck(
                    graph, pos, logit, 
                    throughput_threshold=throughput_threshold, 
                    node_threshold=node_threshold, 
                    edge_threshold=edge_threshold, 
                    MAX_ITERATIONS=MAX_ITERATIONS)
                path.extend(p)
            paths.extend(path)
        pruned = prune_paths_by_first_last(graph, paths, threshold_first, threshold_last)
        last_pos_features = pick_last_pos_features(graph, pruned)
        features.extend(last_pos_features)

        pos = n_pos - 1
        for layer, feature_idx in last_pos_features:
            target_row = torch.tensor((layer, pos, feature_idx))
            matches = (active_features == target_row)
            row_matches_all = torch.all(matches, dim=1)
            indices = torch.nonzero(row_matches_all, as_tuple=False)
            activation_val = 0
            if indices.numel() > 0:
                index = indices.item()
                activation_val = activation_values[index].detach().cpu()
            key = f"{layer}.{feature_idx}"
            try:
                activation_dict[key].append(activation_val)
            except KeyError:
                activation_dict = [activation_val]
    
    return features, activation_dict

if __name__ == "__main__":
    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    data_directory = os.path.join(absolute_directory, "data/flores_features")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    model_name = 'google/gemma-2-2b'
    transcoder_name = "gemma"
    model = ReplacementModel.from_pretrained(model_name, transcoder_name, device=device, dtype=torch.bfloat16)
    for lang, ds_key in lang_to_flores_key.items():
        if lang != 'ko':
            continue
        print(f"Loading {ds_key}")
        ds = load_dataset("openlanguagedata/flores_plus", ds_key, split="dev")
        ds = ds.shuffle(seed=42)
        df = ds.to_pandas()
        batch = df.loc[:150, 'text'].tolist()
        sentences = filter_sentences(batch, alphabet_char[lang], model) # only returns 100 sentences
        features, activations = iterate_through_sentences(model, sentences)
        file_name = f'{lang}.json'
        with open(os.path.join(data_directory, file_name), 'w') as f:
            json.dump(features, f)
        file_name = f'{lang}_activations.json'
        with open(os.path.join(data_directory, file_name), 'w') as f:
            json.dump(activations, f)

