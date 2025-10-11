import torch

from intervention import (
    get_best_base, get_best_rank, get_top_outputs, logit_diff_single,
    visualize_bar_2ddict_outer_inter, create_multi_series_histogram
    )
from template import langs, Feature, ReplacementModel, base_strings

def ablation_test_strength(supernode_dict: dict[str, list[tuple[Feature, float]]], lang: str, alpha: float) -> list[tuple[int, int, int, torch.Tensor]]:
    if lang not in supernode_dict.keys():
        raise KeyError(f'{lang} is not a valid intervention language.')
    intervention_values = []
    for feature, val in supernode_dict[lang]:
        layer = feature.layer
        pos = feature.pos
        feature_idx = feature.feature_idx

        layer = layer.item() if isinstance(layer, torch.Tensor) else layer
        pos = pos.item() if isinstance(pos, torch.Tensor) else pos
        feature_idx = feature_idx.item() if isinstance(feature_idx, torch.Tensor) else feature_idx

        ablation_value = torch.tensor(val * alpha)
        intervention_values.append((layer, pos, feature_idx, ablation_value))
    return intervention_values

def ablation_test_chinese(
        prompt: str, ans: dict[str, list[str]],
        supernodes_dict: dict[str, list[tuple[Feature, float]]], alpha: float, model: ReplacementModel, k: int=10) -> dict[str, tuple[float, float, float, int]]:

    og_logits, og_activations = model.get_activations(prompt)
    outputs_og = get_top_outputs(og_logits, k)
    print(outputs_og)

    intervention_lst = ablation_test_strength(supernodes_dict, 'zh', alpha)

    new_logits, new_activations = model.feature_intervention(prompt, intervention_lst)
    new_outputs = get_top_outputs(new_logits, k)
    print(new_outputs)

    base = get_best_base(og_logits, ans['zh'], model)
    result_dict = dict()

    for lang2 in langs:
        target = get_best_base(og_logits, ans[lang2], model)
        o_diff, n_diff, diff = logit_diff_single(og_logits, new_logits, target, base, model)
        best_rank = get_best_rank(new_logits, ans[lang2], model)

        result_dict[lang2] = (o_diff, n_diff, diff, best_rank)
    return result_dict

def chinese_ablation_iterate(alpha: float, supernodes_dict: dict[str, list[tuple[Feature, float]]], model: ReplacementModel):
    result = dict()
    for lang in langs:
        result[lang] = {'old_logit_diff': [], 'new_logit_diff': [], 'logit_diff_diff': [], 'best_rank': []}

    for adj, ans in data:
        prompt = base_strings['zh'].format(adj=adj['zh'])
        d = ablation_test_chinese(prompt, ans, supernodes_dict, alpha, model)
        for lang in langs:
            o_diff, n_diff, diff, best_rank = d[lang]
            result[lang]['old_logit_diff'].append(o_diff)
            result[lang]['new_logit_diff'].append(n_diff)
            result[lang]['logit_diff_diff'].append(diff)
            result[lang]['best_rank'].append(best_rank)

    for lang in langs:
        for key, value in result[lang].items():
            if key == 'best_rank':
                continue
            if len(value) != 0:
                result[lang][key] = sum(value) / len(value)
            else:
                result[lang][key] = 0
    
    categories = ['old_logit_diff', 'new_logit_diff', 'logit_diff_diff', 'best_rank']
    small_dict = dict()
    for category in categories:
        if category == 'best_rank':
            continue
        small_dict[category] = dict()
        for lang in langs:
            small_dict[category][lang] = result[lang][category]

    visualize_bar_2ddict_outer_inter(small_dict)

    rank_dict = dict()
    for lang in langs:
        rank_dict[lang] = result[lang]['best_rank']

    title = f"Histogram of best_rank with zh as the adjective lang after ablation"
    create_multi_series_histogram(rank_dict, title=title)
