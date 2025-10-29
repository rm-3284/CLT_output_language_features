from collections import defaultdict
import numpy as np
import os
from pathlib import Path
import pandas as pd
import re
import torch
from tqdm.auto import tqdm
from transformers import set_seed


def set_deterministic(seed: int = 42):
    # Set seed for reproducibility
    set_seed(seed=42, deterministic=True)

    # NNsightError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2.
    # To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8.
    # For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def extract_range(file_path: Path):
    m = re.search(r"(\d+)-(\d+)", file_path.name)

    if m:
        start, end = int(m.group(1)), int(m.group(2))
        return start, end

    return (0, 0)

def load_activations(input_dir: Path, layer: str):
    layer_files = sorted(list(input_dir.glob(f"{layer}*.pt")), key=extract_range)
    activations = []
    for layer_file in layer_files:
        activations.extend(torch.load(layer_file, weights_only=False))
    return activations

def process_sae_features(
        sae_features_list: list[any],
        layer: str,
        lang: str,
        rounding_digit=3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sae_feature_index_to_activations = defaultdict(list)
    sae_feature_index_to_dataset_id_token_id_act_val = defaultdict(list)

    for dataset_row_index, sae_features in enumerate(sae_features_list):
        top_acts, top_indices = sae_features
        top_act_index_per_token = zip(top_acts.squeeze(0), top_indices.squeeze(0))

        for token_index, (top_act, top_index) in enumerate(top_act_index_per_token):
            top_act, top_index = top_act.tolist(), top_index.tolist()
            if isinstance(top_act, float) or isinstance(top_act, int):
                top_act = [top_act]
            if isinstance(top_index, float) or isinstance(top_index, int):
                top_index = [top_index]
            for act_val, feature_index in zip(top_act, top_index):
                sae_feature_index_to_activations[feature_index].append(act_val)
                dataset_id_token_id_act_val = (
                    dataset_row_index,
                    token_index,
                    round(act_val, rounding_digit),
                )
                sae_feature_index_to_dataset_id_token_id_act_val[feature_index].append(
                    dataset_id_token_id_act_val
                )

    sae_features_count = {}
    sae_features_avg = {}
    sae_features_q1 = {}
    sae_features_median = {}
    sae_features_q3 = {}
    sae_features_min_active = {}
    sae_features_max_active = {}
    sae_features_std = {}

    for feature_index, activations in sae_feature_index_to_activations.items():
        sae_features_count[feature_index] = len(activations)
        sae_features_avg[feature_index] = round(
            np.mean(activations).item(), rounding_digit
        )
        sae_features_q1[feature_index] = round(
            np.percentile(activations, 25).item(), rounding_digit
        )
        sae_features_median[feature_index] = round(
            np.median(activations).item(), rounding_digit
        )
        sae_features_q3[feature_index] = round(
            np.percentile(activations, 75).item(), rounding_digit
        )
        sae_features_min_active[feature_index] = round(
            np.min(activations).item(), rounding_digit
        )
        sae_features_max_active[feature_index] = round(
            np.max(activations).item(), rounding_digit
        )
        sae_features_std[feature_index] = round(
            np.std(activations).item(), rounding_digit
        )

    # Create a dataframe from the statistics
    statistics = {
        "count": sae_features_count,
        "avg": sae_features_avg,
        "q1": sae_features_q1,
        "median": sae_features_median,
        "q3": sae_features_q3,
        "min_active": sae_features_min_active,
        "max_active": sae_features_max_active,
        "std": sae_features_std,
        "lang": lang,
        "layer": layer,
    }

    df_statistics = pd.DataFrame(statistics)
    df_statistics.sort_index(inplace=True)
    df_statistics.reset_index(inplace=True)

    # Create a dataframe from the dataset_token_activations
    dataset_token_activations = {
        "count": sae_features_count,
        "dataset_row_id_token_id_act_val": sae_feature_index_to_dataset_id_token_id_act_val,
    }

    df_dataset_token_activations = pd.DataFrame(dataset_token_activations)
    df_dataset_token_activations.sort_index(inplace=True)
    df_dataset_token_activations.reset_index(inplace=True)

    return df_statistics, df_dataset_token_activations

if __name__ == "__main__":
    set_deterministic()

    langs = ['bg','zh','en','fr','de','hi','it','ja','ko','pt','ru','es','th','tr','vi',]
    max_layers = 26
    layer_template = "model.layers.{l}.mlp"
    layers = [layer_template.format(l=i) for i in range(max_layers)]

    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    out_dir = os.path.join(absolute_directory, "data/sae_features")
    out_dir = Path(out_dir)

    for lang in tqdm(langs, desc="Processing languages"):
        for layer in tqdm(layers, desc="Processing layers", leave=False):
            input_dir = (out_dir / lang)
            sae_features = load_activations(input_dir, layer)
            df_statistics, df_dataset_token_activations = process_sae_features(
                sae_features, layer, lang
            )

            output_dir = (Path(absolute_directory) / "statistics" / "summary" / layer)
            os.makedirs(output_dir, exist_ok=True)
            df_statistics.to_csv(output_dir / f"{lang}.csv", index=False)

            output_dir = (Path(absolute_directory) / "statistics" / "dataset_token_activation" / layer)
            os.makedirs(output_dir, exist_ok=True)
            df_dataset_token_activations.to_csv(output_dir / f"{lang}.csv", index=False)
