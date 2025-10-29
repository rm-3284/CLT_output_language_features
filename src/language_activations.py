from datasets import Dataset, load_dataset
from functools import reduce
from nnsight import LanguageModel
import os
from pathlib import Path
from sae_lens import SAE
import torch
from tqdm.auto import tqdm

from device_setup import get_device

def get_nested_attr(obj, attr_path):
    return reduce(getattr, attr_path.split("."), obj)

def collect_activations(
        llm: LanguageModel,
        layers: list[int],
        layer_template: str,
        prompt: str
) -> dict[int, list[torch.Tensor]]:
    layers_modules = {}
    for layer in layers:
        layers_modules[layer] = get_nested_attr(llm, layer_template.format(l=layer))

    layers_activations = {layer: None for layer in layers}

    with llm.trace(prompt):
        for layer in layers:
            layers_activations[layer] = layers_modules[layer].output.cpu().save()

    layers_activations_processed = {
        layer: (
            activations[0].value
            if isinstance(activations, tuple)
            else activations.value
        )
        for layer, activations in layers_activations.items()
    }

    return layers_activations_processed

def collect_all_activations(
        llm: LanguageModel, 
        max_layers: int, 
        dataset: Dataset, 
        prompt_template: str,
        layer_template: str,
    ) -> dict[int, list[torch.Tensor]]:
    layers = [i for i in range(max_layers)]
    all_activations = {layer: [] for layer in layers}

    for row in tqdm(dataset, desc="Processing Samples", leave=False):
        prompt = prompt_template.format_map(row)
        activations = collect_activations(llm, layers, layer_template, prompt)

        for layer, layer_activations in activations.items():
            all_activations[layer].append(layer_activations)
    # the key is layer, the value is a list of activations (by tokens)
    return all_activations

def load_sae(
    sae_model_name: str, layer: int,
):
    if sae_model_name.startswith("gemma-scope"):
        sae_model_layer_name = f"layer_{layer}/width_16k/canonical"
        sae = SAE.from_pretrained(sae_model_name, sae_model_layer_name)[0]
        return sae
    
def sae_features_from_activations(
    activations_list: list[torch.Tensor],
    sae: SAE,
    device: torch.device,
    batch: int = 100,
):
    activations_size = [
        activations.shape[1] for activations in activations_list
    ]  # [a, b, ...]
    activations_list = torch.cat(activations_list, dim=1)  # tensor(1, a+b+..., 2048)
    top_acts = []
    top_indices = []
    
    chunks = torch.split(activations_list, batch, dim=1)
    
    for chunk in chunks:
        sae.eval()
        input_activation = chunk.to(device)
        print(chunk.shape, input_activation.shape)
        with torch.no_grad():
            # (batch, seq_len, d_sae)
            _, feature_acts, *args = sae(input_activation)
            print(feature_acts.shape)
            K = 100
            top_values, top_indices_batch = torch.topk(feature_acts, k=K, dim=-1)
        top_acts.append(top_values.unsqueeze(0).cpu())
        top_indices.append(top_indices_batch.unsqueeze(0).cpu())
    
    top_acts = torch.cat(top_acts, dim=1)
    top_acts = torch.split(top_acts, activations_size, dim=1)

    top_indices = torch.cat(top_indices, dim=1)
    top_indices = torch.split(top_indices, activations_size, dim=1)

    all_sae_features = []

    for top_acts, top_indices in zip(top_acts, top_indices):
        all_sae_features.append((top_acts, top_indices))

    return all_sae_features

def collect_all_sae_features(
        all_activations: dict[int, list[torch.Tensor]],
        max_layer: int ,
        sae_model: str,
        batch: int,
) -> dict[int, list[torch.Tensor]]:
    all_sae_features = {layer: [] for layer in range(max_layer)}
    device = get_device()
    for layer, layer_activations in all_activations.items():
        sae = load_sae(sae_model, layer).to(device)
        all_sae_features[layer] = sae_features_from_activations(
            layer_activations, sae, device, batch
        )

    return all_sae_features # val is tuple of top32 activations, indices

def save_activations(
    output_dir: Path,
    activations: dict[int, list[torch.Tensor]],
    start: int,
    end: int | float,
):
    os.makedirs(output_dir, exist_ok=True)

    for layer, layer_activations in activations.items():
        file_path = output_dir / f"{layer}.{start}-{end}.pt"
        torch.save(layer_activations, file_path)


if __name__ == "__main__":
    llm = LanguageModel("google/gemma-2-2b", device_map="auto", dispatch=True)
    sae_name = "gemma-scope-2b-pt-mlp-canonical"
    start, end = 0, 100
    max_layer = 26

    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    out_dir = os.path.join(absolute_directory, "data/sae_features")
    out_dir = Path(out_dir)

    for lang in ['bg','zh','en','fr','de','hi','it','ja','ko','pt','ru','es','th','tr','vi',]:
        try:
            dataset = load_dataset("facebook/xnli", lang, split=f"train[{start}:{end}]", trust_remote_code=True)
        except ValueError:
            print(f"{lang} is not in the list of languages available")
            continue
        # this depends on the dataset and model
        prompt_template = "{premise} {hypothesis}"
        layer_template = "model.layers.{l}.mlp"

        all_activations = collect_all_activations(llm, max_layer, dataset, prompt_template, layer_template)
        all_sae_features = collect_all_sae_features(all_activations, max_layer, sae_name, 100)

        output_dir = (out_dir / lang)
        save_activations(
            output_dir,
            all_sae_features,
            start,
            end
        )

