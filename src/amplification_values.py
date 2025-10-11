import matplotlib.pyplot as plt
import torch
from typing import Optional

from device_setup import device
from template import Graph, Supernode, Feature, ReplacementModel, attribute

def set_features_from_supernodes(*supernodes: Supernode) -> list[Feature]:
    feature_lst = []
    for supernode in supernodes:
        feature_lst.extend(supernode.features)
    return list(set(feature_lst))

def feature_find(graph: Graph, feature: Feature) -> Optional[int]:
    layer = feature.layer
    pos = feature.pos
    if pos < 0:
        pos = graph.n_pos + pos
    feature_idx = feature.feature_idx
    feature_tensor = torch.tensor([layer, pos, feature_idx], device=device)

    element_wise_match = (graph.active_features == feature_tensor)
    row_match = torch.all(element_wise_match, dim=1)
    matching_indices = torch.where(row_match)[0]
    if matching_indices.numel() > 1:
        raise ValueError('Multiple matching rows')
    elif matching_indices.numel() == 0:
        return None
    return matching_indices.item()

def get_feature_activation_from_prompt(
        prompt: str, feature_list: list[Feature], model: ReplacementModel,
        max_n_logits = 5, desired_logit_prob = 0.95,
        max_feature_nodes = None, batch_size = 256,
        offload = 'cpu', verbose = True,
        ) -> list[float]:
    graph = attribute(
        prompt=prompt,
        model=model,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
        batch_size=batch_size,
        max_feature_nodes=max_feature_nodes,
        offload=offload,
        verbose=verbose
    )
    activation_list = []
    for feature in feature_list:
        idx = feature_find(graph, feature)
        if idx == None:
            activation_list.append(float('nan'))
        else:
            activation_value = graph.activation_values[idx]
            activation_value = activation_value.item() if isinstance(activation_value, torch.Tensor) else activation_value
            activation_list.append(activation_value)
    del graph
    return activation_list

def iterate_over_sentences(prompts: list[str], feature_list: list[Feature], model: ReplacementModel) -> dict[Feature, list[float]]:
    activation_values_dict = dict()
    for feature in feature_list:
        activation_values_dict[feature] = []
    for prompt in prompts:
        activation_list = get_feature_activation_from_prompt(prompt, feature_list, model)
        for i, feature in enumerate(feature_list):
            activation_values_dict[feature].append(activation_list[i])
    return activation_values_dict

def make_histogram_from_values_dict(data: list[float], bins: int = 30, title: str = "Histogram of Data (NaNs Excluded)", xlabel: str = "Value", ylabel: str = "Frequency") -> None:
    clean_data = []
    nan_count = 0
    
    for x in data:
        if math.isnan(x):
            nan_count += 1
        else:
            clean_data.append(x)

    print(f"Total NaN values found: {nan_count}")
    if not clean_data:
        print("No valid data points to plot after excluding NaNs.")
        return
    
    maximum = max(clean_data)
    minimum = min(clean_data)
    mean = statistics.mean(clean_data)
    median = statistics.median(clean_data)
    print(f'Max {maximum}, Min {minimum}, Mean {mean}, Median {median}')
    
    plt.figure(figsize=(10, 6)) # Set figure size for better readability
    plt.hist(clean_data, bins=bins, edgecolor='black', alpha=0.7) # 'edgecolor' for bin borders, 'alpha' for transparency
    
    # Add labels and title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Add a text box to display NaN count on the plot
    plt.text(0.95, 0.95, f'NaNs: {nan_count}, Max: {maximum}, Min: {minimum}, Mean: {mean}, Median: {median}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
             
    plt.grid(axis='y', alpha=0.75) # Add a grid for better readability
    plt.show()
    return

def print_feature(feature: Feature) -> str:
    layer = feature.layer
    feature_idx = feature.feature_idx
    layer = layer.item() if isinstance(layer, torch.Tensor) else layer
    feature_idx = feature_idx.item() if isinstance(feature_idx, torch.Tensor) else feature_idx
    return f'Layer {layer}, feature_idx {feature_idx}'

if __name__ == "__main__":
    from device_setup import device

    model_name = 'google/gemma-2-2b'
    transcoder_name = "gemma"
    model = ReplacementModel.from_pretrained(model_name, transcoder_name, device=device, dtype=torch.bfloat16)

    


