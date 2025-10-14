import matplotlib.pyplot as plt
import os
import torch

from intervention import get_top_outputs
from template import (
    ReplacementModel, Feature, Supernode, 
    InterventionGraph, Intervention
    )
from template import base_strings, langs


def choose_lang_node(lang_features: dict[str, int], threshold: int = 10) -> list[Feature]:
    feature_lst = []
    for key, value in lang_features.items():
        if value > threshold:
            layer, feature_idx = key.split('.')
            layer = int(layer) if isinstance(layer, str) else layer
            feature_idx = int(feature_idx) if isinstance(feature_idx, str) else feature_idx
            feature_lst.append(Feature(layer=layer, pos=-1, feature_idx=feature_idx))
    return feature_lst

## metrics
def prob_diff(
        old_logits: torch.Tensor, new_logits: torch.Tensor, 
        targets: list[str], 
        model: ReplacementModel, 
        verbose=True
    ) -> list[float]:
    old_probs = old_logits.squeeze(0)[-1].softmax(-1)
    new_probs = new_logits.squeeze(0)[-1].softmax(-1)

    diffs = []
    for target in targets:
        token = model.tokenizer.encode(target)[1]
        old_prob = old_probs[token].item()
        new_prob = new_probs[token].item()
        if verbose:
            print(f'Probability of "{target}": old {old_prob}, new {new_prob}, diff {new_prob - old_prob}')
        diffs.append(new_prob - old_prob)
    return diffs

def rank_diff(
        old_logits: torch.Tensor, new_logits: torch.Tensor, 
        targets: list[str], 
        model: ReplacementModel, 
        verbose=True
    ) -> list[int]:
    o_logits = old_logits.squeeze(0)[-1]
    _, o_indices = torch.sort(o_logits, dim=-1, descending=True)
    n_logits = new_logits.squeeze(0)[-1]
    _, n_indices = torch.sort(n_logits, dim=-1, descending=True)

    diffs = []
    for target in targets:
        token = model.tokenizer.encode(target)[1]

        o_mask = (o_indices == token)
        o_rank = torch.argmax(o_mask.int(), dim=-1)

        n_mask = (n_indices == token)
        n_rank = torch.argmax(n_mask.int(), dim=-1)
        if verbose:
            print(f'Rank of "{target}": old {o_rank}, new {n_rank}, diff {n_rank - o_rank}')
            
        diffs.append(n_rank - o_rank)
    return diffs

def logit_diff(
        old_logits: torch.Tensor, new_logits: torch.Tensor, 
        targets: list[str], 
        base: str, 
        model: ReplacementModel, 
        verbose=True
    ) -> list[float]:
    o_logits = old_logits.squeeze(0)[-1]
    n_logits = new_logits.squeeze(0)[-1]
    s = model.tokenizer.encode(base)[1]

    diffs = []
    for target in targets:
        t = model.tokenizer.encode(target)[1]
        o_diff = o_logits[t] - o_logits[s]
        n_diff = n_logits[t] - n_logits[s]
        if verbose:
            print(f'Logit difference of "{target}" to "{base}": old {o_diff}, new {n_diff}, diff {n_diff - o_diff}')
        diffs.append(n_diff - o_diff)
    return diffs

def get_best_rank(
        logits: torch.Tensor, 
        targets: list[str], 
        model: ReplacementModel
    ) -> int:
    last_logits = logits.squeeze(0)[-1]
    _, indices = torch.sort(last_logits, dim=-1, descending=True)
    ranks = []
    for target in targets:
        token = model.tokenizer.encode(target)[1]
        mask = (indices == token)
        rank = torch.argmax(mask.int(), dim=-1)
        rank = rank.item() if isinstance(rank, torch.Tensor) else rank
        ranks.append(rank)
    return min(ranks)


def ablation_test_with_metrics(
        prompt_lang: str, adj_lang: str, 
        language_node: Supernode, factor: float, 
        adj_dict: dict[str, str], ans_dict: dict[str, list[str]], 
        model: ReplacementModel, 
        k: int = 10, verbose=True, 
        target_prompt: bool = False
        ) -> tuple[list[float], list[int], list[float], int, int]:
    if prompt_lang not in langs or adj_lang not in langs:
        raise KeyError("Invalid lang")
    
    prompt = base_strings[prompt_lang].format(adj=adj_dict[adj_lang])
    if verbose:
        print(prompt)
    og_logits, og_activations = model.get_activations(prompt)
    outputs_og = get_top_outputs(og_logits, model, k)
    if verbose:
        print(outputs_og)

    graph = InterventionGraph([language_node], prompt=prompt)
    graph.initialize_node(language_node, og_activations)
    interventions = [Intervention(language_node, factor)]
    intervention_values = [(*feature, scaling_factor * default_act) for intervened_supernode, scaling_factor in interventions 
                           for feature, default_act in zip(intervened_supernode.features, intervened_supernode.default_activations)]
    new_logits, new_activations = model.feature_intervention(graph.prompt, intervention_values)
    new_outputs = get_top_outputs(new_logits, model, k)
    if verbose:
        print(new_outputs)

    if not target_prompt:
        target = ans_dict[adj_lang]
        bases = ans_dict[prompt_lang]
    else:
        target = ans_dict[prompt_lang]
        bases = ans_dict['en']

    prob_diffs = prob_diff(og_logits, new_logits, target, model, verbose=verbose)
    rank_diffs = rank_diff(og_logits, new_logits, target, model, verbose=verbose)
    logit_diffs = []
    for base in bases:
        logit_diffs.extend(logit_diff(og_logits, new_logits, target, base, model, verbose=verbose))

    original_rank = get_best_rank(og_logits, target, model)
    if original_rank < 20:
        base_rank = get_best_rank(new_logits, bases, model)
        target_rank = get_best_rank(new_logits, target, model)
    else:
        base_rank = -1
        target_rank = -1

    return prob_diffs, rank_diffs, logit_diffs, base_rank, target_rank

def threshold_prune(lang_features: dict[str, int]) -> list[int]:
    options_set = set()
    for value in lang_features.values():
        options_set.add(value)
    options = [0]
    options.extend(list(options_set))
    options.sort()
    return options

def threshold_determine(
        lang_features: dict[str, int], 
        model: ReplacementModel,
        data: list[tuple[dict[str, str], dict[str, list[str]]]],
        prompt_lang: str = 'en',
        adj_langs: list[str] = list(langs), 
    ) -> tuple[list[int], list[float], list[float], list[float], list[float], list[float]]:
    if prompt_lang not in langs:
        raise KeyError('Invalid prompt language')
    for l in adj_langs:
        if l not in langs:
            raise KeyError('Invalid adj lang')
    
    threshold_options = threshold_prune(lang_features)
    return_lists = {'p': [], 'r': [], 'l': [], 'b': [], 't': []}
    for threshold in threshold_options:
        feature_lst = choose_lang_node(lang_features, threshold)
        tmp_lists = {'p': [], 'r': [], 'l': [], 'b': [], 't': []}
        if not feature_lst:
            idx = threshold_options.index(threshold)
            threshold_options = threshold_options[:idx]
            break
        for lang in adj_langs:
            for adj, ans in data:
                prob_diff, rank_diff, logit_diff, base_rank, target_rank = ablation_test_with_metrics(prompt_lang, lang, Supernode('', features=feature_lst), -2, adj, ans, model)
                tmp_lists['p'].extend(prob_diff)
                tmp_lists['r'].extend(rank_diff)
                tmp_lists['l'].extend(logit_diff)
                if base_rank >= 0:
                    tmp_lists['b'].append(base_rank)
                if target_rank >= 0:
                    tmp_lists['t'].append(target_rank)
        for key, value in tmp_lists.items():
            tmp = sum(value) / len(value)
            tmp = tmp.item() if isinstance(tmp, torch.Tensor) else tmp
            return_lists[key].append(tmp)

    p = return_lists['p']
    p = [-x for x in p]
    r = return_lists['r']
    l = return_lists['l']
    l = [-x for x in l]
    b = return_lists['b']
    t = return_lists['t']
    return threshold_options, p, r, l, b, t

def prob_rank_logit_visualize(
        x: list[int], 
        probs: list[float], 
        ranks: list[float], 
        logits: list[float],
        interactive: bool,
        data_directory: str = '',
        file_name: str = '',
        ) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 6)) # Create the main plot and first axis
    # Plot on the first Y-axis (left)
    ax1.plot(x, probs, color='blue', label='Prob Diff')
    ax1.set_xlabel('Threshold for language features')
    ax1.set_ylabel('Prob Diff', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    # Create a second Y-axis (right) sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(x, ranks, color='red', label='Rank Diff')
    ax2.set_ylabel('Rank Diff', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    ax3 = ax1.twinx()
    # Offset the spine of the third y-axis to the right
    ax3.spines['right'].set_position(('axes', 1.15)) # Adjust this value to move the axis
    ax3.spines['right'].set_visible(True) # Make sure the spine is visible
    # Plot on the third Y-axis
    ax3.plot(x, logits, color='green', label='Logit Diff')
    ax3.set_ylabel('Logit diff', color='green')
    ax3.tick_params(axis='y', labelcolor='green')
    # Add a title and legend
    plt.title('Change in each metric as the threshold changes')
    # Combine legends from all axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax3.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper left')
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    if interactive:
        plt.show()
    else:
        plt_path = os.path.join(data_directory, file_name)
        plt.savefig(plt_path)
        plt.close()
    return

def best_rank_visualize(
        x: list[int], 
        base_ranks: list[float], 
        target_ranks: list[float],
        interactive: bool,
        data_directory: str = '',
        file_name: str = '',
        ) -> None:
    # Create the main plot and the first (left) axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Plot on the first Y-axis (left side)
    ax1.plot(x, base_ranks, color='blue', label='Rank of English words')
    ax1.set_xlabel('Threshold for language features')
    ax1.set_ylabel('Rank (English words)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue') # Set tick label color to match line
    # Create a second Y-axis (right side) sharing the same x-axis
    ax2 = ax1.twinx()
    # Plot on the second Y-axis (right side)
    ax2.plot(x, target_ranks, color='red', label='Rank of adj lang words')
    ax2.set_ylabel('Rank (adj lang words)', color='red')
    ax2.tick_params(axis='y', labelcolor='red') # Set tick label color to match line
    # Add a title
    plt.title('Ranks after ablation')
    # Add a legend
    # It's good practice to combine legends from all axes for clarity
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    # Ensure layout is tight to prevent labels from overlapping
    fig.tight_layout()
    if interactive:
        plt.show()
    else:
        plt_path = os.path.join(data_directory, file_name)
        plt.savefig(plt_path)
        plt.close()
    return

def iterate_threshold_options(lang_features: dict[str, int], 
        model: ReplacementModel,
        data: list[tuple[dict[str, str], dict[str, list[str]]]],
        adjective_lang: str,
        interactive: bool=True,
        data_directory: str = '',
        prompt_lang = 'en',
        ) -> None:
    threshold_options, p, r, l, b, t = threshold_determine(lang_features, model, data, prompt_lang)
    file_name1 = f'prob_rank_logit_{prompt_lang}_{adjective_lang}'
    prob_rank_logit_visualize(threshold_options, p, r, l, interactive, data_directory, file_name1)
    file_name2 = f'best_rank_{prompt_lang}_{adjective_lang}'
    best_rank_visualize(threshold_options, b, t, interactive, data_directory, file_name2)
    return

if __name__ == '__main__':
    from device_setup import device
    model_name = 'google/gemma-2-2b'
    transcoder_name = 'gemma'
    model = ReplacementModel.from_pretrained(model_name, transcoder_name, device=device, dtype=torch.bfloat16)

    import json
    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    data_directory = os.path.join(absolute_directory, "plot/supernode_threshold")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    feature_directory = os.path.join(absolute_directory, "data/features")
    with open(os.path.join(feature_directory, 'en_features.json'), 'r') as f:
        en_features = json.load(f)
    with open(os.path.join(feature_directory, 'de_features.json'), 'r') as f:
        de_features = json.load(f)
    with open(os.path.join(feature_directory, 'fr_features.json'), 'r') as f:
        fr_features = json.load(f)
    with open(os.path.join(feature_directory, 'ja_features.json'), 'r') as f:
        ja_features = json.load(f)
    with open(os.path.join(feature_directory, 'zh_features.json'), 'r') as f:
        zh_features = json.load(f)

    from data.adjectives import train_data
    iterate_threshold_options(en_features, model, train_data, 'en', False, data_directory)
    iterate_threshold_options(de_features, model, train_data, 'de', False, data_directory)
    iterate_threshold_options(fr_features, model, train_data, 'fr', False, data_directory)
    iterate_threshold_options(ja_features, model, train_data, 'ja', False, data_directory)
    iterate_threshold_options(zh_features, model, train_data, 'zh', False, data_directory)

