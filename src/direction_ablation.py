# ==============================================================================
# ROBUST PATCH: Fixes Gemma 2 issues regardless of import order
# ==============================================================================
import torch
import transformers.models.gemma2.modeling_gemma2

# 1. FIX THE "check_model_inputs" CRASH
#    Instead of patching the decorator, we unwrap the already-decorated method.
#    This removes the strict check from the class itself.
try:
    model_class = transformers.models.gemma2.modeling_gemma2.Gemma2Model
    # Check if the forward method is wrapped (decorated)
    if hasattr(model_class.forward, "__wrapped__"):
        print(" [Patch] Removing strict signature check from Gemma2Model...")
        # Restore the original, unchecked forward method
        model_class.forward = model_class.forward.__wrapped__
except Exception as e:
    print(f" [Patch Warning] Could not unwrap Gemma2Model: {e}")

# 2. FIX THE "RoPE" SHAPE MISMATCH (8 vs 256)
#    This replaces the rotary embedding function with a shape-safe version.
def fixed_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # Detect if q is transposed (Heads at the end instead of HeadDim)
    # e.g. [..., 256, 8] instead of [..., 8, 256]
    head_dim = cos.shape[-1]
    is_transposed = False
    
    if q.shape[-1] != head_dim and q.shape[-2] == head_dim:
        q = q.transpose(-1, -2)
        k = k.transpose(-1, -2)
        is_transposed = True

    # Fix broadcasting for cos/sin
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)

    # Apply RoPE
    q_embed = (q * cos) + (transformers.models.gemma2.modeling_gemma2.rotate_half(q) * sin)
    k_embed = (k * cos) + (transformers.models.gemma2.modeling_gemma2.rotate_half(k) * sin)

    # Restore original shape if we transposed
    if is_transposed:
        q_embed = q_embed.transpose(-1, -2)
        k_embed = k_embed.transpose(-1, -2)

    return q_embed, k_embed

print(" [Patch] Applying RoPE shape fix...")
transformers.models.gemma2.modeling_gemma2.apply_rotary_pos_emb = fixed_apply_rotary_pos_emb
# ==============================================================================

import argparse
import json
import nnsight
import os
import torch
import torch.nn.functional as F

from ablation_amplification_intervention import (
    activation_dict, description_based_features, mean_value_based_features,
    freq_based_features, model_run,
    )
from circuit_tracer_import import ReplacementModel, Feature
from data.adjectives import big_data
from device_setup import device
from intervention import (
    get_top_outputs, get_best_base, logit_diff_single, get_best_rank
)
from template import lang_to_flores_key, base_strings

def extract_directions(transcoder_model, layer_feature_map):
    """
    extract direction from the ReplacementModel
    
    Args:
        transcoder_model: ReplacementModel
        layer_feature_map: {layer_idx: [feat_idx1, feat_idx2,...]} dict
        
    Returns:
        direction_dict: {layer_idx: torch.Tensor(shape=[k, d_model])}
    """
    direction_dict = {}
    
    for layer_idx, feature_indices in layer_feature_map.items():
        if hasattr(transcoder_model, 'transcoders'):
            transcoder = transcoder_model.transcoders[layer_idx]
        else:
            raise ValueError("Transcoder attributes not found in the model wrapper.")

        W_dec = transcoder.W_dec # (n_features, d_model)
            
        indices = torch.tensor(feature_indices, device=W_dec.device)
        selected_vectors = torch.index_select(W_dec, 0, indices) # Shape: [k, d_model]
        
        normalized_vectors = F.normalize(selected_vectors, p=2, dim=1)
        
        direction_dict[layer_idx] = normalized_vectors
        
    return direction_dict

def project_orthogonally(residual_stream, directions):
    """
    project residual_stream on the specified directions
    
    Args:
        residual_stream: [batch, seq_len, d_model]
        directions: [num_features, d_model] (Normalized)
        
    Returns:
        ablated_stream: [batch, seq_len, d_model]
    """
    # projection matrix P = D (D^T D)^-1 D^T を計算
    
    # Gram matrix
    G = torch.matmul(directions, directions.T) # [K, K]
    
    # Tikhonov Regularization
    try:
        G_inv = torch.linalg.inv(G + 1e-6 * torch.eye(G.shape[0], device=G.device))
    except RuntimeError:
        G_inv = torch.linalg.pinv(G)
        
    # alpha = (D^T D)^-1 D^T x
    # x:, D^T: -> x @ D^T :
    # projections = (x @ D^T) @ G_inv
    directions = directions.to(dtype=residual_stream.dtype)
    inner_products = torch.matmul(residual_stream, directions.T)
    coefficients = torch.matmul(inner_products, G_inv)
    
    removal_component = torch.matmul(coefficients, directions)
    
    return residual_stream - removal_component

def run_ablation_experiment(model, prompt, directions_map):
    """
    Args:
        model: model from nnsight
        prompt: input prompt
        directions_map: {layer_idx: tensor}
    """
    
    target_layers = set(directions_map.keys())
    
    with model.trace(prompt) as tracer:
        
        # Access to layer depending on the model architecture
        if hasattr(model, "transformer"):
            layers = model.transformer.h
        elif hasattr(model, "model"):
            layers = model.model.layers
        else:
            layers = [m for m in model.modules() if isinstance(m, torch.nn.ModuleList)]

        for i, layer in enumerate(layers):
            if i in target_layers:
                dirs = directions_map[i].to(device)
                
                # access to the residual stream
                # layer.output is usually a tuple (hidden_states, cache, attentions...)
                hidden_states = layer.output[0]
                
                ablated_states = project_orthogonally(hidden_states, dirs)
                
                layer_output = (ablated_states, ) + layer.output[1:]
                layer.output = layer_output
                
                # log for debug
                print(f"Layer {i}: Ablated {dirs.shape} directions.")

        output_logits = model.lm_head.output.save()
        
    return output_logits

def perform_intervention(
    prompt: str, model: ReplacementModel, logits: torch.Tensor,
    adj_lang: str, ans, interventions, langs, nnsight_model
):
    base = get_best_base(logits, ans[adj_lang], model)

    result_output = dict()
    result_logits = dict()

    for intervention_lang in langs:
        ablation_logits = run_ablation_experiment(nnsight_model, prompt, interventions[intervention_lang])
        outputs = get_top_outputs(ablation_logits, model)
        result_output[intervention_lang] = outputs
        result_logits[intervention_lang] = dict()
        for measure_lang in langs:
            target = get_best_base(ablation_logits, ans[measure_lang], model)
            o_diff, n_diff, _ = logit_diff_single(logits, ablation_logits, target, base, model)
            n_rank = get_best_rank(ablation_logits, ans[measure_lang], model)
            result_logits[intervention_lang][measure_lang] = (o_diff, n_diff, n_rank)

    return result_logits, result_output
        

def interventions_to_dict(interventions: dict[str, list[tuple[Feature, float]]], lang: str, model) -> dict[int, list[int]]:
    feature_dict = dict()
    for f, _ in interventions[lang]:
        layer = f.layer
        feature = f.feature_idx
        try:
            feature_dict[layer].append(feature)
        except KeyError:
            feature_dict[layer] = [feature]
    
    return extract_directions(model, feature_dict)

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
    model = ReplacementModel.from_pretrained(model_name, transcoder_name, device=device, dtype=torch.bfloat16, attn_implementation="sdpa")

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

    desc_ablations = dict()
    desc_amplifications = dict()
    val_ablations = dict()
    val_amplifications = dict()
    freq_ablations = dict()
    freq_amplifications = dict()
    for lang in langs:
        desc_ablations[lang] = interventions_to_dict(desc_interventions, lang, model)
        val_ablations[lang] = interventions_to_dict(val_interventions, lang, model)
        freq_ablations[lang] = interventions_to_dict(freq_interventions, lang, model)

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

            desc_based = {'outputs': {}, 'logits_and_ranks': {}}
            val_based = {'outputs': {}, 'logits_and_ranks': {}}
            freq_based = {'outputs': {}, 'logits_and_ranks': {}}


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


                
                ( 
                 ablations_dict, ablations_outputs, 
                 ) = perform_intervention(prompt, model, logits, adj_lang, ans, desc_ablations, langs, nnsight_model)
                
                
                desc_based['outputs'][prompt] = ablations_outputs
                desc_based['logits_and_ranks'][prompt] = ablations_dict

                
                (
                 ablations_dict, ablations_outputs, 
                 ) = perform_intervention(prompt, model, logits, adj_lang, ans, val_ablations, langs, nnsight_model)
                
                val_based['outputs'][prompt] = ablations_outputs
                val_based['logits_and_ranks'][prompt] = ablations_dict
    
                
                (
                 ablations_dict, ablations_outputs, 
                 ) = perform_intervention(prompt, model, logits, adj_lang, ans, freq_ablations, langs, nnsight_model)
                
                freq_based['outputs'][prompt] = ablations_outputs
                freq_based['logits_and_ranks'][prompt] = ablations_dict
                

            for key, val in desc_based.items():
                file_name = f'description_based_direction_ablation_across_layers_{key}.json'
                with open(os.path.join(adj_lang_out_dir, file_name), 'w') as f:
                    json.dump(val2, f, indent=4)

            for key, val in val_based.items():
                file_name = f'value_based_direction_ablation_across_layers_{key}.json'
                with open(os.path.join(adj_lang_out_dir, file_name), 'w') as f:
                    json.dump(val2, f, indent=4)

            for key, val in freq_based.items():
                file_name = f'frequency_based_direction_ablation_across_layers_{key}.json'
                with open(os.path.join(adj_lang_out_dir, file_name), 'w') as f:
                    json.dump(val2, f, indent=4)


