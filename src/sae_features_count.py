import math
import os
from pathlib import Path
import torch

from sae_statistics import load_activations, set_deterministic

def count_sae_features(
        activations_list: list[tuple[torch.Tensor, torch.Tensor]],
        layer_index: int,
        over_zero_token: torch.Tensor,
        over_zero_example: torch.Tensor,
        over_zero_total: torch.Tensor,
        num_examples: int,
        num_tokens: int,
        max_active_over_zero: torch.Tensor,
        min_active_over_zero: torch.Tensor,
        rounding_digit: int=3,
):
    for activations in activations_list:
        top_acts, top_indices = activations

        num_examples += 1
        num_tokens += top_indices.shape[1]

        flat_top_indices = top_indices.flatten()
        flat_top_acts = top_acts.flatten()

        unique_feature_indices = torch.unique(flat_top_indices)
        over_zero_example[layer_index, unique_feature_indices] += 1

        ones = torch.ones_like(flat_top_indices, dtype=over_zero_token.dtype)
        over_zero_token[layer_index].scatter_add_(0, flat_top_indices, ones)

        over_zero_total[layer_index].scatter_add_(0, flat_top_indices, flat_top_acts)

        rounded_acts = flat_top_acts.round(decimals=rounding_digit)

        max_active_over_zero[layer_index].scatter_reduce_(
            0,
            flat_top_indices,
            rounded_acts,
            reduce="amax",
            include_self=True,
        )

        min_active_over_zero[layer_index].scatter_reduce_(
            0,
            flat_top_indices,
            rounded_acts,
            reduce="amin",
            include_self=True,
        )

    return (
        over_zero_token,
        over_zero_example,
        over_zero_total,
        num_examples,
        num_tokens,
        max_active_over_zero,
        min_active_over_zero,
    )

if __name__ == "__main__":
    set_deterministic()

    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    out_dir = os.path.join(absolute_directory, "data/sae_features")
    out_dir = Path(out_dir)

    num_layers = 26
    hidden_dim = 16384
    langs = ['bg','zh','en','fr','de','ru','es','tr','vi',]

    for lang in langs:
        over_zero_token = torch.zeros((num_layers, hidden_dim))
        over_zero_example = torch.zeros((num_layers, hidden_dim))
        over_zero_total = torch.zeros((num_layers, hidden_dim))
        max_active_over_zero = torch.zeros((num_layers, hidden_dim))
        min_active_over_zero = torch.full((num_layers, hidden_dim), math.inf)

        for layer in range(num_layers):
            input_dir = (out_dir / lang)
            activations = load_activations(input_dir, layer)

            num_examples = 0
            num_tokens = 0

            (
                over_zero_token,
                over_zero_example,
                over_zero_total,
                num_examples,
                num_tokens,
                max_active_over_zero,
                min_active_over_zero,
            ) = count_sae_features(
                activations,
                layer,
                over_zero_token,
                over_zero_example,
                over_zero_total,
                num_examples,
                num_tokens,
                max_active_over_zero,
                min_active_over_zero,
            )

        output_dir = Path(absolute_directory) / "data/sae_features_count"
        file_path = output_dir / f"{lang}_extended.pt"

        os.makedirs(output_dir, exist_ok=True)

        # Save as a sparse tensor
        min_active_over_zero[min_active_over_zero == math.inf] = 0

        output = {
            "num_examples": num_examples,
            "num_tokens": num_tokens,
            "over_zero_token": over_zero_token.to_sparse(),
            "over_zero_example": over_zero_example.to_sparse(),
            "over_zero_total": over_zero_total.to_sparse(),
            "max_active_over_zero": max_active_over_zero.to_sparse(),
            "min_active_over_zero": min_active_over_zero.to_sparse(),
        }

        torch.save(output, file_path)
