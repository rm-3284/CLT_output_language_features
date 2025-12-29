import argparse
import json
import nnsight
import os
import torch
import torch.nn.functional as F

from ablation_amplification_intervention import (
    activation_dict,
    ablation_and_amplification,
    combine_except_one,
    description_based_features,
    direction_ablation_layer_determine,
    direction_ablation_helper,
    freq_based_features,
    mean_value_based_features,
    model_intervention,
    model_run,
)
from circuit_tracer_import import ReplacementModel
from data.adjectives import big_data
from device_setup import device
from direction_ablation import (
    interventions_to_dict, interventions_to_dict_everything_ablation, 
    run_ablation_experiment
    )
from intervention import (
    ablation, amplification, get_top_outputs,
    )
from template import lang_to_flores_key, base_strings

def get_logit_and_rank(logits: torch.Tensor, target: str, model: ReplacementModel) -> tuple[float, int, float]:
    # returns logit, rank, prob
    l = logits.squeeze(0)[-1]
    t = model.tokenizer.encode(target)[1]
    lg = l[t]
    lg = lg.item() if isinstance(lg, torch.Tensor) else lg

    _, indices = torch.sort(l, dim=-1, descending=True)
    mask = (indices == t)
    rank = torch.argmax(mask.int(), dim=-1)
    rank = rank.item() if isinstance(rank, torch.Tensor) else rank

    probs = F.softmax(l, dim=-1)
    prob = probs[t]
    prob = prob.item() if isinstance(prob, torch.Tensor) else prob
    return lg, rank, prob

def get_logits_and_ranks(logit: torch.Tensor, ans: dict[str, list[str]], model: ReplacementModel) -> dict[str, dict[str, tuple[float, int, float]]]:
    result = dict()
    for key, value in ans.items():
        result[key] = dict()
        for v in value:
            logit_and_rank = get_logit_and_rank(logit, v, model)
            result[key][v] = logit_and_rank
    return result

def feature_interventions(prompt: str, model: ReplacementModel, ans: dict[str, list[str]], intervention: dict[str, list[tuple[int, int, int, float]]], langs: list[str]):
    results = dict()
    for intervened_lang in langs:
        results[intervened_lang] = dict()

        new_outputs, new_logits = model_intervention(prompt, model, intervention[intervened_lang])
        result = get_logits_and_ranks(new_logits, ans, model)
        results[intervened_lang]['output'] = new_outputs
        results[intervened_lang]['langs'] = result
    return results

def direction_ablate(prompt: str, model: ReplacementModel,
    ans, interventions, langs, nnsight_model
    ):
    results = dict()
    for intervention_lang in langs:
        ablation_logits = run_ablation_experiment(nnsight_model, prompt, interventions[intervention_lang])
        result = get_logits_and_ranks(ablation_logits, ans, model)
        outputs = get_top_outputs(ablation_logits, model)
        results[intervention_lang]['output'] = outputs
        results[intervention_lang]['langs'] = result
    return results

def feature_ablation_and_amplification(prompt: str, model: ReplacementModel, ans: dict[str, list[str]], ablation, amplification, langs):
    results = dict()
    for ablation_lang in langs:
        results[ablation_lang] = dict()
        for amplification_lang in langs:
            results[ablation_lang][amplification_lang] = dict()
            interventions = ablation_and_amplification(ablation, amplification)
            new_outputs, logits = model_intervention(prompt, model, interventions)
            result = get_logits_and_ranks(logits, ans, model)
            results[ablation_lang][amplification_lang]['output'] = new_outputs
            results[ablation_lang][amplification_lang]['langs'] = result
    return results

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prompt language",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # 2. Add a string argument
    # 'message' is the variable name inside the script
    # '--input-string' is the flag used on the command line
    parser.add_argument(
        '--lang',
        '-l',
        type=str,
        default=None,
        help='Prompt language',
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # load the model
    model_name = 'google/gemma-2-2b'
    transcoder_name = "gemma"
    model = ReplacementModel.from_pretrained(model_name, transcoder_name, device=device, dtype=torch.bfloat16)

    nnsight_model = nnsight.LanguageModel(model_name, device_map=device)

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

    desc_ablations = {'feature': dict(), 'one-layer-direction': dict(), 'feature_everything': dict(), 'one-layer-direction_everything': dict(), 'direction-ablation': dict(), 'direction-ablation-everything': dict()}
    desc_amplifications = {'normal': dict(), 'everything': dict()}
    val_ablations = {'feature': dict(), 'one-layer-direction': dict(), 'feature_everything': dict(), 'one-layer-direction_everything': dict(), 'direction-ablation': dict(), 'direction-ablation-everything': dict()}
    val_amplifications = {'normal': dict(), 'everything': dict()}
    freq_ablations = {'feature': dict(), 'one-layer-direction': dict(), 'feature_everything': dict(), 'one-layer-direction_everything': dict(), 'direction-ablation': dict(), 'direction-ablation-everything': dict()}
    freq_amplifications = {'normal': dict(), 'everything': dict()}
    for lang in langs:
        desc_ablations['feature'][lang] = ablation(desc_interventions, lang)
        desc_amplifications['normal'][lang] = amplification(desc_interventions, lang)
        val_ablations['feature'][lang] = ablation(val_interventions, lang)
        val_amplifications['normal'][lang] = amplification(val_interventions, lang)
        freq_ablations['feature'][lang] = ablation(freq_interventions, lang)
        freq_amplifications['normal'][lang] = amplification(freq_interventions, lang)

    for lang in langs:
        desc_ablations['feature_everything'][lang] = combine_except_one(desc_ablations['feature'], lang)
        desc_amplifications['everything'][lang] = combine_except_one(desc_amplifications['normal'], lang)
        val_ablations['feature_everything'][lang] = combine_except_one(val_ablations['feature'], lang)
        val_amplifications['everything'][lang] = combine_except_one(val_amplifications['normal'], lang)
        freq_ablations['feature_everything'][lang] = combine_except_one(freq_ablations['feature'], lang)
        freq_amplifications['everything'][lang] = combine_except_one(freq_amplifications['normal'], lang)

    for lang in langs:
        desc_ablations['direction-ablation'][lang] = interventions_to_dict(desc_interventions, lang, model)
        desc_ablations['direction-ablation-everything'][lang] = interventions_to_dict_everything_ablation(desc_interventions, lang, model)
        val_ablations['direction-ablation'][lang] = interventions_to_dict(val_interventions, lang, model)
        val_ablations['direction-ablation-everything'][lang] = interventions_to_dict_everything_ablation(val_interventions, lang, model)
        freq_ablations['direction-ablation'][lang] = interventions_to_dict(freq_interventions, lang, model)
        freq_ablations['direction-ablation-everything'][lang] = interventions_to_dict_everything_ablation(freq_interventions, lang, model)

    one_layer_ablation = {'desc': dict(), 'val': dict(), 'freq': dict()}
    for lang in langs:
        one_layer_ablation['desc'][lang] = direction_ablation_layer_determine(desc_interventions, lang)
        one_layer_ablation['val'][lang] = direction_ablation_layer_determine(val_interventions, lang)
        one_layer_ablation['freq'][lang] = direction_ablation_layer_determine(freq_interventions, lang)


    # ablation + amplification experiments
    output_dir = os.path.join(data_directory, "interventions")
    for prompt_lang in langs:
        if args.lang != None:
            if prompt_lang != args.lang:
                continue

        lang_out_dir = os.path.join(output_dir, prompt_lang)

        base = base_strings[prompt_lang]
        for adj_lang in langs:
            adj_lang_out_dir = os.path.join(lang_out_dir, adj_lang)
            os.makedirs(adj_lang_out_dir, exist_ok=True)

            desc_based = {
                'original': {}, 'distractor ablation': {}, 'ablation': {}, 'distractor one-layer direction ablation': {}, 
                'one-layer direction ablation': {}, 'distractor multi-layer direction ablation': {}, 
                'multi-layer direction ablation': {}, 'amplification': {}, 'non-distractor amplification': {}, 
                'feature-intervention': {}, 'one-layer direction intervention': {}, }
            val_based = {
                'original': {}, 'distractor ablation': {}, 'ablation': {}, 'distractor one-layer direction ablation': {}, 
                'one-layer direction ablation': {}, 'distractor multi-layer direction ablation': {}, 
                'multi-layer direction ablation': {}, 'amplification': {}, 'non-distractor amplification': {}, 
                'feature-intervention': {}, 'one-layer direction intervention': {}, }
            freq_based = {
                'original': {}, 'distractor ablation': {}, 'ablation': {}, 'distractor one-layer direction ablation': {}, 
                'one-layer direction ablation': {}, 'distractor multi-layer direction ablation': {}, 
                'multi-layer direction ablation': {}, 'amplification': {}, 'non-distractor amplification': {}, 
                'feature-intervention': {}, 'one-layer direction intervention': {}, }

            for adj, ans in big_data:
                adjective = adj[adj_lang]
                prompt = base.format(adj=adjective)

                # calculate the direction ablations
                for lang in langs:
                    layer, features = one_layer_ablation['desc'][lang]
                    interventions = direction_ablation_helper(model, layer, features, prompt)
                    desc_ablations['one-layer-direction'][lang] = interventions
                for lang in langs:
                    desc_ablations['one-layer-direction_everything'][lang] = combine_except_one(desc_ablations['one-layer-direction'], lang)
                
                for lang in langs:
                    layer, features = one_layer_ablation['val'][lang]
                    interventions = direction_ablation_helper(model, layer, features, prompt)
                    val_ablations['one-layer-direction'][lang] = interventions
                for lang in langs:
                    val_ablations['one-layer-direction_everything'][lang] = combine_except_one(val_ablations['one-layer-direction'], lang)
                
                for lang in langs:
                    layer, features = one_layer_ablation['freq'][lang]
                    interventions = direction_ablation_helper(model, layer, features, prompt)
                    freq_ablations['one-layer-direction'][lang] = interventions
                for lang in langs:
                    freq_ablations['one-layer-direction_everything'][lang] = combine_except_one(freq_ablations['one-layer-direction'], lang)
    
                
                # original
                base_line, logits = model_run(prompt, model)
                before_intervention = get_logits_and_ranks(logits, ans, model)
                desc_based['original'][prompt] = {'output': base_line, 'langs': before_intervention}
                val_based['original'][prompt] = {'output': base_line, 'langs': before_intervention}
                freq_based['original'][prompt] = {'output': base_line, 'langs': before_intervention}
                
                # distractor ablation
                desc_based['distractor ablation'][prompt] = feature_interventions(prompt, model, ans, desc_ablations['feature'], langs)
                val_based['distractor ablation'][prompt] = feature_interventions(prompt, model, ans, val_ablations['feature'], langs)
                freq_based['distractor ablation'][prompt] = feature_interventions(prompt, model, ans, freq_ablations['feature'], langs)

                # ablation
                desc_based['ablation'][prompt] = feature_interventions(prompt, model, ans, desc_ablations['feature_everything'], langs)
                val_based['ablation'][prompt] = feature_interventions(prompt, model, ans, val_ablations['feature_everything'], langs)
                freq_based['ablation'][prompt] = feature_interventions(prompt, model, ans, freq_ablations['feature_everything'], langs)
                
                # distractor one-layer direction
                desc_based['distractor one-layer direction ablation'][prompt] = feature_interventions(prompt, model, ans, desc_ablations['one-layer-direction'], langs)
                val_based['distractor one-layer direction ablation'][prompt] = feature_interventions(prompt, model, ans, val_ablations['one-layer-direction'], langs)
                freq_based['distractor one-layer direction ablation'][prompt] = feature_interventions(prompt, model, ans, freq_ablations['one-layer-direction'], langs)

                # one-layer direction ablation
                desc_based['one-layer direction ablation'][prompt] = feature_interventions(prompt, model, ans, desc_ablations['one-layer-direction_everything'], langs)
                val_based['one-layer direction ablation'][prompt] = feature_interventions(prompt, model, ans, val_ablations['one-layer-direction_everything'], langs)
                freq_based['one-layer direction ablation'][prompt] = feature_interventions(prompt, model, ans, freq_ablations['one-layer-direction_everything'], langs)

                # distractor multi-layer direction ablation
                desc_based['distractor multi-layer direction ablation'][prompt] = direction_ablate(prompt, model, ans, desc_ablations['direction-ablation'], langs, nnsight_model)
                val_based['distractor multi-layer direction ablation'][prompt] = direction_ablate(prompt, model, ans, val_ablations['direction-ablation'], langs, nnsight_model)
                freq_based['distractor multi-layer direction ablation'][prompt]= direction_ablate(prompt, model, ans, freq_ablations['direction-ablation'], langs, nnsight_model)

                # multi-layer direction ablaiton
                desc_based['multi-layer direction ablation'][prompt] = direction_ablate(prompt, model, ans, desc_ablations['direction-ablation-everything'], langs, nnsight_model)
                val_based['multi-layer direction ablation'][prompt] = direction_ablate(prompt, model, ans, val_ablations['direction-ablation-everything'], langs, nnsight_model)
                freq_based['multi-layer direction ablation'][prompt] = direction_ablate(prompt, model, ans, freq_ablations['direction-ablation-everything'], langs, nnsight_model)

                # amplification
                desc_based['amplification'][prompt] = feature_interventions(prompt, model, ans, desc_amplifications['normal'], langs)
                val_based['amplification'][prompt] = feature_interventions(prompt, model, ans, val_amplifications['normal'], langs)
                freq_based['amplification'][prompt] = feature_interventions(prompt, model, ans, freq_amplifications['normal'], langs)

                # non-distractor amplification
                desc_based['non-distractor amplification'][prompt] = feature_interventions(prompt, model, ans, desc_amplifications['everything'], langs)
                val_based['non-distractor amplification'][prompt] = feature_interventions(prompt, model, ans, val_amplifications['everything'], langs)
                freq_based['non-distractor amplification'][prompt] = feature_interventions(prompt, model, ans, freq_amplifications['everything'], langs)

                # feature-intervention
                desc_based['feature-intervention'][prompt] = feature_ablation_and_amplification(prompt, model, ans, desc_ablations['feature'], desc_amplifications['normal'], langs)
                val_based['feature-intervention'][prompt] = feature_ablation_and_amplification(prompt, model, ans, val_ablations['feature'], val_amplifications['normal'], langs)
                freq_based['feature-intervention'][prompt] = feature_ablation_and_amplification(prompt, model, ans, freq_ablations['feature'], freq_amplifications['normal'], langs)

                # one-layer direction intervention
                desc_based['one-layer direction intervention'][prompt] = feature_ablation_and_amplification(prompt, model, ans, desc_ablations['one-layer-direction'], desc_amplifications['normal'], langs)
                val_based['one-layer direction intervention'][prompt] = feature_ablation_and_amplification(prompt, model, ans, val_ablations['one-layer-direction'], val_amplifications['normal'], langs)
                freq_based['one-layer direction intervention'][prompt] = feature_ablation_and_amplification(prompt, model, ans, freq_ablations['one-layer-direction'], freq_amplifications['normal'], langs)

            filename = "interventions_and_results_description.json"
            with open(os.path.join(adj_lang_out_dir, filename), 'w') as f:
                json.dump(desc_based, f, indent=4)
            
            filename = "interventions_and_results_value.json"
            with open(os.path.join(adj_lang_out_dir, filename), 'w') as f:
                json.dump(val_based, f, indent=4)

            filename = "interventions_and_results_frequency.json"
            with open(os.path.join(adj_lang_out_dir, filename), 'w') as f:
                json.dump(freq_based, f, indent=4)
