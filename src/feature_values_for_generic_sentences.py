import torch
from typing import Optional

from template import Graph, Feature, attribute, ReplacementModel

def feature_find(graph: Graph, feature: Feature, device) -> Optional[int]:
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
        prompt: str, feature_list: list[list[int]],
        model: ReplacementModel, device,
        max_n_logits:int=5, desired_logit_prob:float=0.95,
        max_feature_nodes=None, batch_size:int=256,
        offload:str='cpu', verbose:bool=True,
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
        layer, feature_idx = feature
        f = Feature(layer=layer, pos=-1, feature_idx = feature_idx)
        idx = feature_find(graph, f, device)
        if idx == None:
            activation_list.append(float('nan'))
        else:
            activation_value = graph.activation_values[idx]
            activation_value = activation_value.item() if isinstance(activation_value, torch.Tensor) else activation_value
            activation_list.append(activation_value)
    del graph
    return activation_list

def iterate_over_sentences(
        prompts: list[str], feature_list: list[list[int]], 
        model: ReplacementModel, device,
        ) -> dict[str, list[float]]:

    activation_values_dict = dict()
    for feature in feature_list:
        activation_values_dict[feature] = []
    for prompt in prompts:
        activation_list = get_feature_activation_from_prompt(prompt, feature_list, model, device)
        for i, feature in enumerate(feature_list):
            activation_values_dict[feature].append(activation_list[i])
    return activation_values_dict

if __name__ == "__main__":
    from device_setup import device
    model_name = 'google/gemma-2-2b'
    transcoder_name = 'gemma'
    model = ReplacementModel.from_pretrained(model_name, transcoder_name, device=device, dtype=torch.bfloat16)

    import os, json
    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    data_directory = os.path.join(absolute_directory, "data/generic_sentences")

    with open(os.path.join(data_directory, 'en_sentences.json'), 'r') as f:
        en_sentences = json.load(f)
    with open(os.path.join(data_directory, 'de_sentences.json'), 'r') as f:
        de_sentences = json.load(f)
    with open(os.path.join(data_directory, 'fr_sentences.json'), 'r') as f:
        fr_sentences = json.load(f)
    with open(os.path.join(data_directory, 'ja_sentences.json'), 'r') as f:
        ja_sentences = json.load(f)
    with open(os.path.join(data_directory, 'zh_sentences.json'), 'r') as f:
        zh_sentences = json.load(f)
    
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

    def flatten(xss):
        return [x for xs in xss for x in xs]

    features = flatten([list(en_features.keys()), list(de_features.keys()), list(fr_features.keys()), list(ja_features.keys()), list(zh_features.keys())])
    
    en_values = iterate_over_sentences(en_sentences, features, model, device)
    de_values = iterate_over_sentences(de_sentences, features, model, device)
    fr_values = iterate_over_sentences(fr_sentences, features, model, device)
    ja_values = iterate_over_sentences(ja_sentences, features, model, device)
    zh_values = iterate_over_sentences(zh_sentences, features, model, device)

    values_directory = os.path.join(absolute_directory, 'data/values')
    if not os.path.exists(values_directory):
        os.makedirs(values_directory)

    with open(os.path.join(values_directory, 'en_values.json'), 'w') as f:
        json.dump(en_values, f)
    with open(os.path.join(values_directory, 'de_values.json'), 'w') as f:
        json.dump(de_values, f)
    with open(os.path.join(values_directory, 'fr_values.json'), 'w') as f:
        json.dump(fr_values, f)
    with open(os.path.join(values_directory, 'ja_values.json'), 'w') as f:
        json.dump(ja_values, f)
    with open(os.path.join(values_directory, 'zh_values.json'), 'w') as f:
        json.dump(zh_values, f)
    