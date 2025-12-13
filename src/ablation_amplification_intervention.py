import json
import os
import torch
from typing import Literal

from circuit_tracer_import import Feature, ReplacementModel
from data.adjectives import big_data
from device_setup import device
from intervention import (
    ablation, amplification, get_best_base, get_best_rank,
    get_top_outputs, logit_diff_single, check_valid_meaning,
    visualize_bar_2ddict_outer_inter, bar_graph_visualize,
    create_multi_series_histogram,
    )
from template import lang_to_flores_key, base_strings

def description_based_features(dir_name: str, languages: list[str], threshold: float=0.1) -> dict[str, list[str]]:
    lang_features = dict()
    for lang in languages:
        file_name = f"{lang}_features.json"
        with open(os.path.join(dir_name, file_name), 'r') as f:
            features_freq = json.load(f)
        max_val = max(features_freq.values())
        features = list()
        for key, val in features_freq.items():
            if val >= max_val * threshold:
                features.append(key)
        lang_features[lang] = features
    return lang_features

def mean_value_based_features(dir_name: str, languages: list[str], topk: int=50) -> dict[str, list[str]]:
    lang_features = dict()
    for lang in languages:
        file_name = f"{lang}.json"
        with open(os.path.join(dir_name, file_name), 'r') as f:
            feature_val = json.load(f)
        sorted_features = sorted(feature_val.items(), key=lambda item: item[1], reverse=True)[:topk]
        features = list()
        for feature, _ in sorted_features:
            features.append(feature)
        lang_features[lang] = features
    return lang_features

def freq_based_features(dir_name: str, languages: list[str]) -> dict[str, list[str]]:
    lang_features = dict()
    file_name = "features_0.98.json"
    with open(os.path.join(dir_name, file_name), 'r') as f:
        feature_dict = json.load(f)
    for lang in languages:
        features = feature_dict[lang]
        features_str = list()
        for key, feature_idx in features:
            string = f"{key}.{feature_idx}"
            features_str.append(string)
        lang_features[lang] = features_str
    return lang_features

def model_run(prompt: str, model: ReplacementModel) -> tuple[list[tuple[str, float]], torch.Tensor]:
    logits, _ = model.get_activations(prompt)
    return get_top_outputs(logits, model), logits

def model_intervention(prompt: str, model: ReplacementModel, interventions: list[tuple[int, int, int, float]]) -> tuple[list[tuple[str, float]], torch.Tensor]:
    logits, _ = model.feature_intervention(prompt, interventions)
    return get_top_outputs(logits, model), logits

def map_from_mode_to_idx(mode: str) -> Literal[2, 3, 4, 5]:
    if mode == 'minimum':
        return 2
    elif mode == 'maximum':
        return 3
    elif mode == 'mean':
        return 4
    elif mode == 'median':
        return 5
    else:
        raise KeyError(f"{mode} is not a valid argument. Options are ['minimum', 'maximum', 'mean', 'median']")
    

def activation_dict(lang_feature_dict: dict[str, list[str]], dir_path: str, langs: list[str], mode: str = 'median') -> dict[str, list[tuple[Feature, float]]]:
    idx = map_from_mode_to_idx(mode)
    
    feature_val_by_langs = dict()
    for lang in langs:
        file_name = f"{lang}_pos_summary.json"
        with open(os.path.join(dir_path, file_name)) as f:
            feature_values = json.load(f)
        
        feature_val_list = list()
        for key in lang_feature_dict[lang]:
            layer, feature_idx = key.split('.')
            layer = int(layer)
            feature_idx = int(feature_idx)
            feature = Feature(layer=layer, pos=-1, feature_idx=feature_idx)
            val = feature_values[key][idx]
            if val == float('nan'):
                val = 0
            feature_val_list.append((feature, val))
        feature_val_by_langs[lang] = feature_val_list
    return feature_val_by_langs

def ablation_and_amplification(ablation, amplification):
    ablation_features = set()
    for layer, _, feature_idx, _ in ablation:
        ablation_features.add((layer, feature_idx))
    amplification_features = set()
    for layer, _, feature_idx, _ in amplification:
        amplification_features.add((layer, feature_idx))
    intersection = ablation_features.intersection(amplification_features)

    intervention_list = []
    for layer, pos, feature_idx, val in ablation:
        if (layer, feature_idx) not in intersection:
            intervention_list.append((layer, pos, feature_idx, val))
    for layer, pos, feature_idx, val in amplification:
        if (layer, feature_idx) not in intersection:
            intervention_list.append((layer, pos, feature_idx, val))
    return intervention_list

def perform_intervention(
        prompt: str, model: ReplacementModel, old_logits: torch.Tensor, 
        adj_lang: str, ans, ablation, amplification, langs):
    # info that should be stored is after each intervention, 
    # for answer in each language, how the logit changed and rank changed
    # the base should be the adjective language
    base = get_best_base(logits, ans[adj_lang], model)
    best_rank_before_intervention = dict()
    for lang in langs:
        best_rank_before_intervention[lang] = get_best_rank(logits, ans[lang], model)

    ablations_dict = dict()
    ablations_outputs = dict()
    for intervention_lang in langs:
        ablations_dict[intervention_lang] = dict()
        ablation_outputs, ablation_logits = model_intervention(prompt, model, ablation[intervention_lang])
        ablations_outputs[intervention_lang] = ablation_outputs
        for measure_lang in langs:
            target = get_best_base(ablation_logits, ans[measure_lang], model)
            o_diff, n_diff, _ = logit_diff_single(logits, ablation_logits, target, base, model)
            n_rank = get_best_rank(ablation_logits, ans[measure_lang], model)

            ablations_dict[intervention_lang][measure_lang] = (o_diff, n_diff, n_rank)

    amplifications_dict = dict()
    amplifications_outputs = dict()
    for intervention_lang in langs:
        amplifications_dict[intervention_lang] = dict()
        amplification_outputs, amplification_logits = model_intervention(prompt, model, amplification[intervention_lang])
        amplifications_outputs[intervention_lang] = amplification_outputs
        for measure_lang in langs:
            target = get_best_base(amplification_logits, ans[measure_lang], model)
            o_diff, n_diff, _ = logit_diff_single(logits, amplification_logits, target, base, model)
            n_rank = get_best_rank(amplification_logits, ans[measure_lang], model)

            amplifications_dict[intervention_lang][measure_lang] = (o_diff, n_diff, n_rank)

    interventions_dict = dict()
    interventions_outputs = dict()
    for ablation_lang in langs:
        interventions_dict[ablation_lang] = dict()
        interventions_outputs[ablation_lang] = dict()
        for amplification_lang in langs:
            intervention_list = ablation_and_amplification(ablation[ablation_lang], amplification[amplification_lang])
            interventions_dict[ablation_lang][amplification_lang] = dict()
            intervention_outputs, intervention_logits = model_intervention(prompt, model, intervention_list)
            interventions_outputs[amplification_lang] = intervention_outputs
            for measure_lang in langs:
                target = get_best_base(intervention_logits, ans[measure_lang], model)
                o_diff, n_diff, _ = logit_diff_single(logits, intervention_logits, target, base, model)
                n_rank = get_best_rank(intervention_logits, ans[measure_lang], model)
                interventions_dict[ablation_lang][amplification_lang][measure_lang] = (o_diff, n_diff, n_rank)
    
    return best_rank_before_intervention, ablations_dict, ablations_outputs, amplifications_dict, amplifications_outputs, interventions_outputs, interventions_dict
        

if __name__ == "__main__":
    # load the model
    model_name = 'google/gemma-2-2b'
    transcoder_name = "gemma"
    model = ReplacementModel.from_pretrained(model_name, transcoder_name, device=device, dtype=torch.bfloat16)

    # relevant directories
    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    data_directory = os.path.join(absolute_directory, "data")
    flores_directory = os.path.join(data_directory, "flores_features")
    lang_specific_directory = os.path.join(data_directory, "language_specific_features")
    multilingual_features_directory = os.path.join(data_directory, "multilingual_llm_features")
    amplification_values_directory = os.path.join(data_directory, "amplification_values")

    langs = list(lang_to_flores_key.keys())

    # get the features + amplification values
    desc_features = description_based_features(flores_directory, langs, 0.1)
    val_features = mean_value_based_features(multilingual_features_directory, langs, 50)
    freq_features = freq_based_features(lang_specific_directory, langs)

    desc_interventions = activation_dict(desc_features, amplification_values_directory, langs)
    val_interventions = activation_dict(val_features, amplification_values_directory, langs)
    freq_interventions = activation_dict(freq_features, amplification_values_directory, langs)

    desc_ablations = dict()
    desc_amplifications = dict()
    val_ablations = dict()
    val_amplifications = dict()
    freq_ablations = dict()
    freq_amplifications = dict()
    for lang in langs:
        desc_ablations[lang] = ablation(desc_interventions, lang)
        desc_amplifications[lang] = amplification(desc_interventions, lang)
        val_ablations[lang] = ablation(val_interventions, lang)
        val_amplifications[lang] = amplification(val_interventions, lang)
        freq_ablations[lang] = ablation(freq_interventions, lang)
        freq_amplifications[lang] = amplification(freq_interventions, lang)

    # ablation + amplification experiments
    output_dir = os.path.join(data_directory, "interventions")
    for prompt_lang in langs:
        if prompt_lang != 'zh':
            continue
        lang_out_dir = os.path.join(output_dir, prompt_lang)

        base = base_strings[prompt_lang]
        for adj_lang in langs:
            adj_lang_out_dir = os.path.join(lang_out_dir, adj_lang)
            os.makedirs(adj_lang_out_dir, exist_ok=True)

            desc_based = {'ablation': {'outputs': {}, 'logits_and_ranks': {}}, 'amplification': {'outputs': {}, 'logits_and_ranks': {}}, 'intervention': {'outputs': {}, 'logits_and_ranks': {}}}
            val_based = {'ablation': {'outputs': {}, 'logits_and_ranks': {}}, 'amplification': {'outputs': {}, 'logits_and_ranks': {}}, 'intervention': {'outputs': {}, 'logits_and_ranks': {}}}
            freq_based = {'ablation': {'outputs': {}, 'logits_and_ranks': {}}, 'amplification': {'outputs': {}, 'logits_and_ranks': {}}, 'intervention': {'outputs': {}, 'logits_and_ranks': {}}}

            best_rank_without_intervention = dict()
            for adj, ans in big_data:
                adjective = adj[adj_lang]
                prompt = base.format(adj=adjective)
                
                base_line, logits = model_run(prompt, model)
                
                if (get_best_rank(logits, ans['en'], model) >= 10 and 
                    get_best_rank(logits, ans[adj_lang], model) >= 10 and
                    get_best_rank(logits, ans[prompt_lang], model) >= 10):
                    # the model does not understand the correct meaning of the adjective
                    print(f"skipping prompt {prompt}")
                    continue
                
                (best_rank_before_intervention, 
                 ablations_dict, ablations_outputs, 
                 amplifications_dict, amplifications_outputs, 
                 interventions_outputs, interventions_dict
                 ) = perform_intervention(prompt, model, logits, adj_lang, ans, desc_ablations, desc_amplifications, langs)
                
                best_rank_without_intervention[prompt] = best_rank_before_intervention
                desc_based['ablation']['outputs'][prompt] = ablations_outputs
                desc_based['ablation']['logits_and_ranks'][prompt] = ablations_dict
                desc_based['amplification']['outputs'][prompt] = amplifications_outputs
                desc_based['amplification']['logits_and_ranks'][prompt] = amplifications_dict
                desc_based['intervention']['outputs'][prompt] = interventions_outputs
                desc_based['intervention']['logits_and_ranks'][prompt] = interventions_dict
                
                (_, 
                 ablations_dict, ablations_outputs, 
                 amplifications_dict, amplifications_outputs, 
                 interventions_outputs, interventions_dict
                 ) = perform_intervention(prompt, model, logits, adj_lang, ans, val_ablations, val_amplifications, langs)
                
                val_based['ablation']['outputs'][prompt] = ablations_outputs
                val_based['ablation']['logits_and_ranks'][prompt] = ablations_dict
                val_based['amplification']['outputs'][prompt] = amplifications_outputs
                val_based['amplification']['logits_and_ranks'][prompt] = amplifications_dict
                val_based['intervention']['outputs'][prompt] = interventions_outputs
                val_based['intervention']['logits_and_ranks'][prompt] = interventions_dict
                
                (_, 
                 ablations_dict, ablations_outputs, 
                 amplifications_dict, amplifications_outputs, 
                 interventions_outputs, interventions_dict
                 ) = perform_intervention(prompt, model, logits, adj_lang, ans, freq_ablations, freq_amplifications, langs)
                
                freq_based['ablation']['outputs'][prompt] = ablations_outputs
                freq_based['ablation']['logits_and_ranks'][prompt] = ablations_dict
                freq_based['amplification']['outputs'][prompt] = amplifications_outputs
                freq_based['amplification']['logits_and_ranks'][prompt] = amplifications_dict
                freq_based['intervention']['outputs'][prompt] = interventions_outputs
                freq_based['intervention']['logits_and_ranks'][prompt] = interventions_dict

            for key, val in desc_based.items():
                for key2, val2 in val.items():
                    file_name = f'description_based_{key}_{key2}.json'
                    with open(os.path.join(adj_lang_out_dir, file_name)) as f:
                        json.dump(val2, f, indent=4)

            for key, val in val_based.items():
                for key2, val2 in val.items():
                    file_name = f'value_based_{key}_{key2}.json'
                    with open(os.path.join(adj_lang_out_dir, file_name)) as f:
                        json.dump(val2, f, indent=4)

            for key, val in freq_based.items():
                for key2, val2 in val.items():
                    file_name = f'frequency_based_{key}_{key2}.json'
                    with open(os.path.join(adj_lang_out_dir, file_name)) as f:
                        json.dump(val2, f, indent=4)


