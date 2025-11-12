import bisect
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import torch

from language_activations import load_sae
from sae_statistics import set_deterministic

def aggregate_activations_count(
    lang_to_activations_count, normalized_lang, activations_count
):
    prev_num_examples = lang_to_activations_count[normalized_lang].get(
        "num_examples", 0
    )
    prev_num_tokens = lang_to_activations_count[normalized_lang].get("num_tokens", 0)
    prev_over_zero_token = lang_to_activations_count[normalized_lang].get(
        "over_zero_token", 0
    )
    prev_over_zero_example = lang_to_activations_count[normalized_lang].get(
        "over_zero_example", 0
    )
    prev_over_zero_total = lang_to_activations_count[normalized_lang].get(
        "over_zero_total", 0
    )

    prev_max_active_over_zero = lang_to_activations_count[normalized_lang].get(
        "max_active_over_zero",
        activations_count["max_active_over_zero"].to_dense(),
    )

    current_min_active_over_zero = activations_count["min_active_over_zero"].to_dense()
    current_min_active_over_zero[current_min_active_over_zero == 0] = math.inf
    prev_min_active_over_zero = lang_to_activations_count[normalized_lang].get(
        "min_active_over_zero",
        current_min_active_over_zero,
    )

    lang_to_activations_count[normalized_lang] = {
        "num_examples": prev_num_examples + activations_count["num_examples"],
        "num_tokens": prev_num_tokens + activations_count["num_tokens"],
        "over_zero_token": prev_over_zero_token
        + activations_count["over_zero_token"].to_dense(),
        "over_zero_example": prev_over_zero_example
        + activations_count["over_zero_example"].to_dense(),
        "over_zero_total": prev_over_zero_total
        + activations_count["over_zero_total"].to_dense(),
        "max_active_over_zero": torch.max(
            prev_max_active_over_zero,
            activations_count["max_active_over_zero"].to_dense(),
        ),
        "min_active_over_zero": torch.min(
            prev_min_active_over_zero,
            current_min_active_over_zero,
        ),
    }

def stack_activations_count(
    lang_to_activations_count, sorted_lang
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack activations count for each language.

    Returns:
        tuple[list, list, torch.Tensor]: num_examples, num_tokens, over_zero_token
            num_examples: number of examples for each language
            num_tokens: number of tokens for each language
            over_zero_token: number of activations over zero for each language with shape (layer x hidden_dim x lang)
    """
    num_examples = []
    num_tokens = []
    over_zero_token = []
    over_zero_example = []
    global_max_active_over_zero = None
    global_min_active_over_zero = None

    global_over_zero_total = 0
    global_over_zero_token = 0

    for lang in sorted_lang:
        num_examples.append(lang_to_activations_count[lang]["num_examples"])
        num_tokens.append(lang_to_activations_count[lang]["num_tokens"])
        over_zero_token.append(lang_to_activations_count[lang]["over_zero_token"])
        over_zero_example.append(lang_to_activations_count[lang]["over_zero_example"])

        global_over_zero_total += lang_to_activations_count[lang]["over_zero_total"]
        global_over_zero_token += lang_to_activations_count[lang]["over_zero_token"]

        if global_max_active_over_zero is None:
            global_max_active_over_zero = lang_to_activations_count[lang][
                "max_active_over_zero"
            ]
        else:
            global_max_active_over_zero = torch.max(
                global_max_active_over_zero,
                lang_to_activations_count[lang]["max_active_over_zero"],
            )

        if global_min_active_over_zero is None:
            global_min_active_over_zero = lang_to_activations_count[lang][
                "min_active_over_zero"
            ]
        else:
            global_min_active_over_zero = torch.min(
                global_min_active_over_zero,
                lang_to_activations_count[lang]["min_active_over_zero"],
            )

    num_examples = torch.tensor(num_examples)
    num_tokens = torch.tensor(num_tokens)
    over_zero_token = torch.stack(over_zero_token, dim=-1)
    over_zero_example = torch.stack(over_zero_example, dim=-1)

    global_avg_active_over_zero = global_over_zero_total / global_over_zero_token

    return (
        num_examples,
        num_tokens,
        over_zero_token,
        over_zero_example,
        global_max_active_over_zero,
        global_min_active_over_zero,
        global_avg_active_over_zero,
    )

def sae_lape(
    num_examples,
    num_tokens,
    over_zero_token,
    over_zero_example,
    global_max_active_over_zero,
    global_min_active_over_zero,
    global_avg_active_over_zero,
    topk_threshold_ratio,
    example_rate,
    top,
    top_per_layer,
    sorted_lang,
    entropy_threshold,
    lang_specific,
    lang_shared,
    shared_count,
    top_by_frequency,
):
    num_layers, hidden_dim, lang = over_zero_token.size()
    activation_probs = over_zero_token / num_tokens
    # L1 normalization
    normed_activation_probs = activation_probs / activation_probs.sum(dim=-1, keepdim=True)
    normed_activation_probs[torch.isnan(normed_activation_probs)] = 0
    # entropy calculation
    log_probs = torch.where(normed_activation_probs > 0, normed_activation_probs.log(), 0)
    entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)
    if torch.isnan(entropy).sum():
        raise ValueError(f"NaN values in entropy tensor: {torch.isnan(entropy).sum()}")
    
    # Take the N smallest or largest values of the entropy tensor
    # If largest is True, take the N largest values, otherwise take the N smallest values.
    largest = False
    
    # dismiss the neuron if no language has a total activated examples over the threshold
    num_examples = num_examples.clone().detach().unsqueeze(0) * example_rate
    num_examples = num_examples.type(torch.int)
    over_zero_example_dismissed_neurons = (over_zero_example >= num_examples).any(dim=-1)
    # Dismiss the neuron if no language has a total activated tokens over the threshold
    # total activated tokens > 10% of the total tokens for HFLs
    hfl_rate = 0.1
    num_tokens = num_tokens.clone().detach().unsqueeze(0) * hfl_rate
    num_tokens = num_tokens.type(torch.int)
    over_zero_token_dismissed_neurons = (over_zero_token > num_tokens).any(dim=-1)

    # set the entropy of the dismissed neuron
    dismissed_neurons = over_zero_example_dismissed_neurons * over_zero_token_dismissed_neurons
    patched_val = -torch.inf if largest else torch.inf
    entropy[dismissed_neurons == 0] = patched_val

    min_entropy = entropy[entropy != patched_val].min() if largest else entropy.min()
    max_entropy = (
        entropy[entropy != patched_val].max()
        if not largest
        else (entropy != patched_val).min()
    )
    non_inf_entropy = (entropy != patched_val).sum().item()

    # Take the N smallest or largest values of the entropy tensor for each language
    flattened_entropy = entropy.flatten()
    non_inf_entropy = (entropy != patched_val).sum().item()

    # Filter non inf entropy
    values, indices = flattened_entropy.topk(non_inf_entropy, largest=largest)

    # Convert the index to the row and column index of the entropy tensor
    layer_index = indices // entropy.size(1)
    hidden_dim_index = indices % entropy.size(1)
    selected_probs = activation_probs[layer_index, hidden_dim_index]  # topk x lang
    merged_index = torch.stack((layer_index, hidden_dim_index), dim=-1)
    # selected_probs shape = (lang x topk)
    selected_probs = selected_probs.transpose(0, 1)

    # Create mask for elements >= N% of max in their column
    # True means features are considered as specific for that language
    max_selected_probs = selected_probs.max(dim=0, keepdim=True).values
    mask_selected_probs = selected_probs >= (max_selected_probs * topk_threshold_ratio)

    # Check shared features across languages
    count_shared_features = mask_selected_probs.sum(dim=0)
    count_shared_indices = torch.nonzero(count_shared_features > 1).squeeze()

    if lang_specific:
        mask_selected_probs[:, count_shared_indices] = False

        # Check entropies for each count of shared features
        for count in range(1, len(sorted_lang) + 1):
            count_shared_features_indices = torch.nonzero(
                count_shared_features == count
            ).squeeze()

            if count_shared_features_indices.numel() == 0:
                continue

            count_shared_features_indices = [
                tuple(row.tolist())
                for row in merged_index[count_shared_features_indices]
            ]

            row_idx, col_idx = zip(*count_shared_features_indices)
            count_entropies = entropy[row_idx, col_idx].flatten()
            sorted_count_entropies = count_entropies.sort().values.round(decimals=3)

    elif lang_shared:
        non_specific_count_indices = torch.nonzero(
            count_shared_features != shared_count
        ).squeeze()
        mask_selected_probs[:, non_specific_count_indices] = False

    # Get the indices of the selected features for each language
    lang, topk_indice = torch.where(mask_selected_probs)

    lang_bincount = torch.bincount(lang).tolist()
    lang_topk_indice = topk_indice.split(lang_bincount)

    # Store the indices of the selected neurons for each language
    final_indice = []
    final_indice_global_max_active = []
    final_indice_global_min_active = []
    final_indice_global_avg_active = []
    features_info = {}
    for i, indices in enumerate(lang_topk_indice):
        lang_indices = [tuple(row.tolist()) for row in merged_index[indices]]

        if top:
            if top_by_frequency:
                # Get the indices of the selected features for each language
                matches = (
                    merged_index[:, None, :] == torch.tensor(lang_indices)[None, :, :]
                ).all(dim=2)
                matching_rows = matches.any(dim=1)
                selected_indices = torch.nonzero(matching_rows).squeeze()

                lang_probs = selected_probs[i, selected_indices]
                lang_indices = sorted(
                    lang_indices,
                    key=lambda indices: lang_probs[lang_indices.index(indices)],
                    reverse=True,
                )

            if top_per_layer:
                count_layer = [0] * num_layers
                new_lang_indices = []

                for lang_index in lang_indices:
                    layer_index = lang_index[0]

                    if count_layer[layer_index] < top:
                        new_lang_indices.append(lang_index)
                        count_layer[layer_index] += 1

                lang_indices = new_lang_indices
            else:
                lang_indices = lang_indices[:top]

        if entropy_threshold:
            row_idx, col_idx = zip(*lang_indices)
            lang_index_entropy = entropy[row_idx, col_idx].tolist()
            lang_index_entropy_threshold_index = bisect.bisect_right(
                lang_index_entropy, entropy_threshold
            )
            lang_indices = lang_indices[:lang_index_entropy_threshold_index]

        lang_indices.sort()

        if len(lang_indices) == 0:
            continue

        row_idx, col_idx = zip(*lang_indices)
        entropy_values = entropy[row_idx, col_idx]
        min_entropy = entropy_values.min().item()
        max_entropy = entropy_values.max().item()

        # Get the indices of the selected features for each language
        matches = (
            merged_index[:, None, :] == torch.tensor(lang_indices)[None, :, :]
        ).all(dim=2)
        matching_rows = matches.any(dim=1)
        selected_indices = torch.nonzero(matching_rows).squeeze()

        features_info[sorted_lang[i]] = {
            "indicies": lang_indices,
            "selected_probs": selected_probs[i, selected_indices],
            "entropies": entropy_values,
        }

        layer_index = [[] for _ in range(num_layers)]
        layer_index_max_active = [[] for _ in range(num_layers)]
        layer_index_min_active = [[] for _ in range(num_layers)]
        layer_index_avg_active = [[] for _ in range(num_layers)]

        for l, h in lang_indices:
            layer_index[l].append(h)
            layer_index_max_active[l].append(global_max_active_over_zero[l, h].item())
            layer_index_min_active[l].append(global_min_active_over_zero[l, h].item())
            layer_index_avg_active[l].append(global_avg_active_over_zero[l, h].item())

        for l, h in enumerate(layer_index):
            layer_index[l] = torch.tensor(h).long()

        for l, h in enumerate(layer_index_max_active):
            layer_index_max_active[l] = torch.tensor(h)

        for l, h in enumerate(layer_index_min_active):
            layer_index_min_active[l] = torch.tensor(h)

        for l, h in enumerate(layer_index_avg_active):
            layer_index_avg_active[l] = torch.tensor(h)

        final_indice.append(layer_index)
        final_indice_global_max_active.append(layer_index_max_active)
        final_indice_global_min_active.append(layer_index_min_active)
        final_indice_global_avg_active.append(layer_index_avg_active)

    return (
        final_indice,
        final_indice_global_max_active,
        final_indice_global_min_active,
        final_indice_global_avg_active,
        features_info,
    )

def lape(
    over_zero_token,
    num_tokens,
    global_max_active_over_zero,
    global_min_active_over_zero,
    global_avg_active_over_zero,
    sorted_lang,
):
    top_rate = 0.01
    filter_rate = 0.95
    activation_bar_ratio = 0.95

    num_layers, intermediate_size, lang_num = over_zero_token.size()

    activation_probs = over_zero_token / num_tokens  # layer x inter x lang_num
    normed_activation_probs = activation_probs / activation_probs.sum(
        dim=-1, keepdim=True
    )
    normed_activation_probs[torch.isnan(normed_activation_probs)] = 0
    log_probs = torch.where(
        normed_activation_probs > 0, normed_activation_probs.log(), 0
    )
    entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)
    largest = False

    if torch.isnan(entropy).sum():
        print(torch.isnan(entropy).sum())
        raise ValueError

    flattened_probs = activation_probs.flatten()
    top_prob_value = flattened_probs.kthvalue(
        round(len(flattened_probs) * filter_rate)
    ).values.item()

    print(top_prob_value)

    # dismiss the neruon if no language has an activation value over top 90%
    top_position = (activation_probs > top_prob_value).sum(dim=-1)
    entropy[top_position == 0] = -torch.inf if largest else torch.inf

    flattened_entropy = entropy.flatten()
    top_entropy_value = round(len(flattened_entropy) * top_rate)
    _, index = flattened_entropy.topk(top_entropy_value, largest=largest)
    row_index = index // entropy.size(1)
    col_index = index % entropy.size(1)
    selected_probs = activation_probs[row_index, col_index]  # n x lang

    # for r, c in zip(row_index, col_index):
    #     print(r, c, activation_probs[r][c])

    print(selected_probs.size(0), torch.bincount(selected_probs.argmax(dim=-1)))

    selected_probs = selected_probs.transpose(0, 1)
    activation_bar = flattened_probs.kthvalue(
        round(len(flattened_probs) * activation_bar_ratio)
    ).values.item()

    print((selected_probs > activation_bar).sum(dim=1).tolist())

    lang, indice = torch.where(selected_probs > activation_bar)

    final_indice = []
    final_indice_global_max_active = []
    final_indice_global_min_active = []
    final_indice_global_avg_active = []
    features_info = {}

    merged_index = torch.stack((row_index, col_index), dim=-1)

    for i, index in enumerate(indice.split(torch.bincount(lang).tolist())):
        lang_index = [tuple(row.tolist()) for row in merged_index[index]]

        lang_index.sort()

        # Compare each row in tensor to each target
        # tensor[:, None, :] has shape [N, 1, D]
        # targets[None, :, :] has shape [1, M, D]
        # Broadcasting to [N, M, D]
        matches = (
            merged_index[:, None, :] == torch.tensor(lang_index)[None, :, :]
        ).all(dim=2)

        # Reduce across target rows (axis=1) to see if any matched
        matching_rows = matches.any(dim=1)

        # Get indices of matching rows
        merge_indices = torch.nonzero(matching_rows).squeeze()

        row_idx, col_idx = zip(*lang_index)

        features_info[sorted_lang[i]] = {
            "indicies": lang_index,
            "selected_probs": selected_probs[i, merge_indices],
            "entropies": entropy[row_idx, col_idx],
        }

        layer_index = [[] for _ in range(num_layers)]
        layer_index_max_active = [[] for _ in range(num_layers)]
        layer_index_min_active = [[] for _ in range(num_layers)]
        layer_index_avg_active = [[] for _ in range(num_layers)]

        for l, h in lang_index:
            layer_index[l].append(h)
            layer_index_max_active[l].append(global_max_active_over_zero[l, h].item())
            layer_index_min_active[l].append(global_min_active_over_zero[l, h].item())
            layer_index_avg_active[l].append(global_avg_active_over_zero[l, h].item())

        for l, h in enumerate(layer_index):
            layer_index[l] = torch.tensor(h).long()

        for l, h in enumerate(layer_index_max_active):
            layer_index_max_active[l] = torch.tensor(h)

        for l, h in enumerate(layer_index_min_active):
            layer_index_min_active[l] = torch.tensor(h)

        for l, h in enumerate(layer_index_avg_active):
            layer_index_avg_active[l] = torch.tensor(h)

        final_indice.append(layer_index)
        final_indice_global_max_active.append(layer_index_max_active)
        final_indice_global_min_active.append(layer_index_min_active)
        final_indice_global_avg_active.append(layer_index_avg_active)

    return (
        final_indice,
        final_indice_global_max_active,
        final_indice_global_min_active,
        final_indice_global_avg_active,
        features_info,
    )

def get_sae_features(final_indices, sae_model_name, layers):
    num_langs = len(final_indices)

    sae_features = [[] for _ in range(num_langs)]

    for layer in layers:
        sae = load_sae(sae_model_name, layer)

        for lang in range(num_langs):
            feature_indices = final_indices[lang][layer]
            features = sae.W_dec[feature_indices].detach()
            sae_features[lang].append(features)

    return sae_features

lang_choices_to_qualified_name = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "hi": "Hindi",
    "es": "Spanish",
    "th": "Thai",
    "bg": "Bulgarian",
    "ru": "Russian",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "eng_Latn": "English",
    "deu_Latn": "German",
    "fra_Latn": "French",
    "ita_Latn": "Italian",
    "por_Latn": "Portuguese",
    "hin_Deva": "Hindi",
    "spa_Latn": "Spanish",
    "tha_Thai": "Thai",
    "bul_Cyrl": "Bulgarian",
    "rus_Cyrl": "Russian",
    "tur_Latn": "Turkish",
    "vie_Latn": "Vietnamese",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",
    "cmn_Hans": "Chinese",
    "eng": "English",
    "deu": "German",
    "fra": "French",
    "ita": "Italian",
    "por": "Portuguese",
    "hin": "Hindi",
    "spa": "Spanish",
    "tha": "Thai",
    "bul": "Bulgarian",
    "rus": "Russian",
    "tur": "Turkish",
    "vie": "Vietnamese",
    "jpn": "Japanese",
    "kor": "Korean",
    "cmn": "Chinese",
}

lang_short_dict = {
    "English": "en",
    "German": "de",
    "French": "fr",
    "Italian": "it",
    "Portuguese": "pt",
    "Hindi": "hi",
    "Spanish": "es",
    "Thai": "th",
    "Bulgarian": "bg",
    "Russian": "ru",
    "Turkish": "tr",
    "Vietnamese": "vi",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh",
}

if __name__ == "__main__":
    set_deterministic()

    # args
    algorithm = "sae_lape"
    topk_threshold_ratio = 0.8
    example_rate = 0.98
    top = 0
    top_per_layer = True
    entropy_threshold = 10
    lang_specific = True
    lang_shared = True
    shared_count = 2
    top_by_frequency = True

    lang_to_activations_count = defaultdict(dict)
    langs = ['bg','zh','en','fr','de','ru','es','tr','vi',]
    sae_name = "gemma-scope-2b-pt-mlp-canonical"
    max_layers = 26

    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    out_dir = os.path.join(absolute_directory, "data/sae_features")
    out_dir = Path(out_dir)
    
    for lang in langs:
        normalized_lang = lang_choices_to_qualified_name[lang]

        output_dir = Path(absolute_directory) / "data/sae_features_count"
        file_path = output_dir / f"{lang}_extended.pt"

        activations_count = torch.load(file_path)
        aggregate_activations_count(lang_to_activations_count, normalized_lang, activations_count)

    # LAPE input
    sorted_lang = sorted(lang_to_activations_count.keys())
    (
        num_examples,
        num_tokens,
        over_zero_token,
        over_zero_example,
        global_max_active_over_zero,
        global_min_active_over_zero,
        global_avg_active_over_zero,
    ) = stack_activations_count(lang_to_activations_count, sorted_lang)

    # LAPE Output
    if algorithm == "sae_lape":
        (
            final_indice,
            final_indice_global_max_active,
            final_indice_global_min_active,
            final_indice_global_avg_active,
            features_info,
        ) = sae_lape(
            num_examples,
            num_tokens,
            over_zero_token,
            over_zero_example,
            global_max_active_over_zero,
            global_min_active_over_zero,
            global_avg_active_over_zero,
            topk_threshold_ratio,
            example_rate,
            top,
            top_per_layer,
            sorted_lang,
            entropy_threshold,
            lang_specific,
            lang_shared,
            shared_count,
            top_by_frequency,
        )
    elif algorithm == "lape":
        (
            final_indice,
            final_indice_global_max_active,
            final_indice_global_min_active,
            final_indice_global_avg_active,
            features_info,
        ) = lape(
            over_zero_token,
            num_tokens,
            global_max_active_over_zero,
            global_min_active_over_zero,
            global_avg_active_over_zero,
            sorted_lang,
        )
    
    sae_features = (
        {
            "sae_features": get_sae_features(
                final_indice, sae_name, [i for i in range(max_layers)]
            )
        }
    )

    # Save as a sparse tensor
    global_min_active_over_zero[global_min_active_over_zero == math.inf] = 0

    output = {
        "final_indice": final_indice,
        "final_indice_global_max_active": final_indice_global_max_active,
        "final_indice_global_min_active": final_indice_global_min_active,
        "final_indice_global_avg_active": final_indice_global_avg_active,
        "sorted_lang": sorted_lang,
        "num_examples": num_examples,
        "num_tokens": num_tokens,
        "over_zero_token": over_zero_token.to_sparse(),
        "over_zero_example": over_zero_example.to_sparse(),
        "features_info": features_info,
        **sae_features,
    }
    print(output)

    file_path = out_dir / "result_extended.pt"
    torch.save(output, file_path)

    for key, val in features_info.items():
        lang = lang_short_dict[key]
        indices = val["indicies"]
        probs = val["selected_probs"].tolist()
        probs = [probs] if isinstance(probs, float) else probs
        entropies = val["entropies"].tolist()
        entropies = [entropies] if isinstance(entropies, float) else entropies
        result_list = []
        for index, prob, entropy in zip(indices, probs, entropies):
            result_list.append((index, prob, entropy))
        file_path = out_dir / f"{lang}_extended.json"
        with open(file_path, "w") as f:
            json.dump(result_list, f)

