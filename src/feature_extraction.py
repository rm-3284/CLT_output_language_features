import copy
import heapq
import json
import requests
import torch
from typing import Optional

from template import Graph, ReplacementModel, attribute, prune_graph
from template import base_strings, langs

## helper functions
def token_to_idx(graph: Graph, token: int) -> int:
    features = len(graph.selected_features)
    errors = graph.cfg.n_layers * graph.n_pos
    if token >= 0:
        return features + errors + token
    else:
        return features + errors + graph.n_pos + token

def logit_to_idx(graph: Graph, logit: int) -> int:
    features = len(graph.selected_features)
    errors = graph.cfg.n_layers * graph.n_pos
    tokens = graph.n_pos
    return features + errors + tokens + logit

def path_reconstruct(start, last, step_dict) -> list[int]:
    curr = last
    path = [last]
    while curr != start:
        try:
            curr = step_dict[curr]
            path.insert(0, curr)
        except KeyError:
            raise KeyError('Path is disconnected')
    return path

def path_to_edge_weights(graph: Graph, path: list[int]) -> list[float]:
    path_copy = copy.deepcopy(path)
    last = path[-1]
    if last < len(graph.logit_tokens):
        path_copy[-1] = logit_to_idx(graph, path[-1])
    n = len(path_copy)
    weights = []
    for i in range(n - 1):
        weight = (graph.adjacency_matrix[path_copy[i+1], path_copy[i]]).item()
        weights.append(weight)
    return weights

# find the distinct paths that have high edge weights
def distinct_path_max_bottleneck(
        graph: Graph, token_idx: int, logit_idx: int,
        throughput_threshold: float = 0.1,
        node_threshold: float = 0.8, edge_threshold: float = 0.98,
        MAX_ITERATIONS: int = 75,
        ) -> list[list[int]]:

    if throughput_threshold < 0:
        raise ValueError('The throughput threshold cannot be negative')
    if MAX_ITERATIONS <= 0:
        raise ValueError('The maximum number of iterations have to be positive')

    pruned_adjacency_matrix = copy.deepcopy(graph.adjacency_matrix)
    node_mask, edge_mask, _ = prune_graph(graph, node_threshold, edge_threshold)
    n, _ = pruned_adjacency_matrix.shape

    for i in range(n):
        if not node_mask[i]:
            pruned_adjacency_matrix[i, :] = 0.0
            pruned_adjacency_matrix[:, i] = 0.0
    pruned_adjacency_matrix = pruned_adjacency_matrix * edge_mask.float()

    negative_values = (pruned_adjacency_matrix > 0).float() # 0 means it is negative
    pruned_adjacency_matrix = pruned_adjacency_matrix * negative_values

    start_node_idx = token_to_idx(graph, token_idx)
    target_node_idx = logit_to_idx(graph, logit_idx)

    # exclude the direct path
    pruned_adjacency_matrix[target_node_idx, start_node_idx] = 0

    matrix_for_distinct_paths = copy.deepcopy(pruned_adjacency_matrix) # This matrix gets nodes removed after each path

    paths = []
    iteration_counter = 0

    while True:
        iteration_counter += 1
        if iteration_counter > MAX_ITERATIONS:
            print(f"Hit MAX_ITERATIONS ({MAX_ITERATIONS}). Forced termination.")
            return paths

        # Step 1: Find the max-throughput path in the *current* network state (matrix_for_distinct_paths)
        # using a modified Dijkstra's algorithm (or similar widest path algorithm)

        # bottleneck_capacity[v] stores the maximum bottleneck capacity found so far to reach v from start_node_idx
        # predecessor[v] stores the predecessor node on that max-bottleneck path
        bottleneck_capacity = torch.full((n,), float('-inf')) # Use -inf for min-heap (will store negative for max-heap)
        bottleneck_capacity[start_node_idx] = float('inf') # Source has infinite capacity to itself (bottleneck is min of path)
        predecessor = {} # Stores predecessor for path reconstruction

        # Max-heap (Python's heapq is a min-heap, so store negative capacities to simulate a max-heap)
        # Priority Queue: ( -bottleneck_capacity, node_idx )
        pq = [(-float('inf'), start_node_idx)] # (negative_capacity_to_node, node_idx)

        current_path_predecessors = {} # For this iteration's path reconstruction

        path_found_in_iteration = False

        while pq:
            # Pop node with current highest bottleneck capacity
            # Use -w to convert back to positive for comparison
            neg_current_bottleneck, u = heapq.heappop(pq)
            current_bottleneck = -neg_current_bottleneck

            # If we've already found a better or equal bottleneck path to 'u', skip
            if current_bottleneck < bottleneck_capacity[u].item(): # Compare with stored value
                continue

            if u == target_node_idx:
                path_found_in_iteration = True
                break # Found a path to the target

            # Iterate over neighbors v of u (outgoing edges from u)
            # Assuming adjacency_matrix[destination, source] = weight
            outgoing_edges_from_u = matrix_for_distinct_paths[:, u]
            valid_neighbors = (outgoing_edges_from_u > throughput_threshold).nonzero().squeeze(1) # Find neighbors with positive capacity

            for v_tensor in valid_neighbors:
                v = v_tensor.item()
                edge_capacity_u_v = outgoing_edges_from_u[v].item()

                new_bottleneck = min(current_bottleneck, edge_capacity_u_v)

                if new_bottleneck > bottleneck_capacity[v].item(): # If this path offers a better bottleneck to v
                    bottleneck_capacity[v] = new_bottleneck
                    current_path_predecessors[v] = u # Store predecessor for this path
                    heapq.heappush(pq, (-new_bottleneck, v)) # Add to PQ (negated for max-heap behavior)

        if not path_found_in_iteration or bottleneck_capacity[target_node_idx].item() <= throughput_threshold:
            # No path found to target_node_idx, or the best path found is below the threshold
            return paths

        # Step 2: Reconstruct the path found
        reconstructed_path = path_reconstruct(start_node_idx, target_node_idx, current_path_predecessors)
        paths.append(reconstructed_path)

        # Step 3: Remove internal nodes of the found path for the next iteration (to find distinct paths)
        # Ensure not to remove source or target nodes
        for node_in_path in reconstructed_path[1:-1]: # Exclude start and end nodes
            matrix_for_distinct_paths[node_in_path, :] = 0.0 # Remove all outgoing edges from this node
            matrix_for_distinct_paths[:, node_in_path] = 0.0 # Remove all incoming edges to this node

# helper functions for the graph visualization
def paths_list(*paths_lsts: list[list[int]]) -> list[list[int]]:
    path_sets = []
    for paths_lst in paths_lsts:
        for path in paths_lst:
            path_sets.append(path)
    return path_sets

def create_feature_dict(graph: Graph, paths: list[list[int]]) -> dict[str, str]:
    feature_dict = dict()

    for path in paths:
        features = path[1:-1]
        for feature in features:
            layer, pos, feature_idx = graph.active_features[graph.selected_features[feature]]
            key = f"{layer.item()}.{feature_idx.item()}"
            if feature_dict.get(key) == None:
                response = requests.get(f"https://www.neuronpedia.org/api/feature/gemma-2-2b/{layer}-gemmascope-transcoder-16k/{feature_idx}")
                explanations = response.json()['explanations']
                description = explanations[0]['description']
                feature_dict[key] = description

    return feature_dict

def prune_paths_by_first_last(graph: Graph, paths: list[list[int]], threshold_first: float, threshold_last: float) -> list[list[int]]:
    pruned_paths = []
    for path in paths:
        weights = path_to_edge_weights(graph, path)
        if weights[0] > threshold_first and weights[-1] > threshold_last:
            pruned_paths.append(path)
    return pruned_paths

def paths_set_to_json(graph: Graph, paths: list[list[int]], filename: Optional[str] = None, feature_dict: Optional[dict[str, str]] = None) -> dict[str, str]:
    feature_set = set()
    for path in paths:
        middle = path[1:-1]
        for feature in middle:
            feature_set.add(feature)

    description_dict = dict()
    if feature_dict == None:
        feature_dict = dict()

    for feature in feature_set:
        layer, pos, feature_idx = graph.active_features[graph.selected_features[feature]]
        layer = layer.item() if isinstance(layer, torch.Tensor) else layer
        pos = pos.item() if isinstance(pos, torch.Tensor) else pos
        feature_idx = feature_idx.item() if isinstance(feature_idx, torch.Tensor) else feature_idx

        description_key = f'{layer}.{pos}.{feature_idx}'
        feature_dict_key = f'{layer}.{feature_idx}'
        try:
            description = feature_dict[feature_dict_key]
            description_dict[description_key] = description
        except KeyError:
            try:
                response = requests.get(f"https://www.neuronpedia.org/api/feature/gemma-2-2b/{layer}-gemmascope-transcoder-16k/{feature_idx}")
                explanations = response.json()['explanations']
                description = explanations[0]['description']
                feature_dict[feature_dict_key] = description
                description_dict[description_key] = description
            except TypeError:
                raise TypeError(f"Layer {layer}, position {pos}, feature {feature_idx} does not exist")
                
    if filename != None:
        with open(filename, 'w') as file:
            json.dump(description_dict, file, indent=4)

    return description_dict
    
def find_substring(hay: str, needles: list[str]) -> bool:
    for needle in needles:
        if needle in hay:
            return True
    return False

def pick_last_pos_features(graph: Graph, paths: list[list[int]]) -> list[tuple[int, int]]:
    feature_list = []
    n_pos = graph.n_pos
    for path in paths:
        middle = path[1:-1]
        for feature in middle:
            layer, pos, feature_idx = graph.active_features[graph.selected_features[feature]]
            layer = layer.item() if isinstance(layer, torch.Tensor) else layer
            pos = pos.item() if isinstance(pos, torch.Tensor) else pos
            feature_idx = feature_idx.item() if isinstance(feature_idx, torch.Tensor) else feature_idx

            if pos == n_pos - 1:
                feature_list.append((layer, feature_idx))

    return feature_list

def choose_language_features(features: list[tuple[int, int]], language_identifiers: list[str], feature_dict: Optional[dict[str, str]] = None) -> dict[str, int]:
    lang_feature_dict = dict()
    for layer, feature_idx in features:
        key = f'{layer}.{feature_idx}'
        if feature_dict == None:
            try:
                response = requests.get(f"https://www.neuronpedia.org/api/feature/gemma-2-2b/{layer}-gemmascope-transcoder-16k/{feature_idx}")
                explanations = response.json()['explanations']
                description = explanations[0]['description']
            except TypeError:
                raise TypeError(f"Layer {layer}, feature {feature_idx} does not exist")
        else:
            try:
                description = feature_dict[key]
            except KeyError:
                try:
                    response = requests.get(f"https://www.neuronpedia.org/api/feature/gemma-2-2b/{layer}-gemmascope-transcoder-16k/{feature_idx}")
                    explanations = response.json()['explanations']
                    description = explanations[0]['description']
                    feature_dict[key] = description
                except TypeError:
                    raise TypeError(f"Layer {layer}, feature {feature_idx} does not exist")
                
        if find_substring(description, language_identifiers):
            if lang_feature_dict.get(key) == None:
                lang_feature_dict[key] = 1
            else:
                lang_feature_dict[key] = lang_feature_dict[key] + 1

    return lang_feature_dict

def iterate_through_data(
        train_data: list[tuple[dict[str, str], dict[str, list[str]]]],
        model: ReplacementModel,
        lang: str,
        base_prompt: str = base_strings['en'],
        important_pos: list[int] = [2, -4, -1],
        logit_focus: list[int] = [0],
        throughput_threshold: float = 0.1,
        node_threshold: float = 0.8, edge_threshold: float = 0.98,
        MAX_ITERATIONS: int = 75,
        threshold_first = 0.5, threshold_last = 0.25,
        max_n_logits = 5, desired_logit_prob = 0.95,
        max_feature_nodes = None, batch_size = 256,
        offload = 'cpu', verbose = True,
        ) -> list[tuple[int, int]]:
    features = []
    if lang not in langs:
        raise KeyError(f"{lang} is not a valid language for this experiment")
    for adj, ans in train_data:
        prompt = base_prompt.format(adj=adj[lang])
        graph = attribute(
            prompt=prompt,
            model=model,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            verbose=verbose,
        )
        paths = []
        for pos in important_pos:
            path = []
            for logit in logit_focus:
                p = distinct_path_max_bottleneck(
                    graph, pos, logit, 
                    throughput_threshold=throughput_threshold, 
                    node_threshold=node_threshold, 
                    edge_threshold=edge_threshold, 
                    MAX_ITERATIONS=MAX_ITERATIONS)
                path.extend(p)
            paths.extend(path)
        pruned = prune_paths_by_first_last(graph, paths, threshold_first, threshold_last)
        last_pos_features = pick_last_pos_features(graph, pruned)
        features.extend(last_pos_features)
    return features

if __name__ == '__main__':
    from device_setup import device

    model_name = 'google/gemma-2-2b'
    transcoder_name = 'gemma'
    model = ReplacementModel.from_pretrained(model_name, transcoder_name, device=device, dtype=torch.bfloat16)

    from data.adjectives import train_data
    import os
    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    data_directory = os.path.join(absolute_directory, "data/features")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    feature_dict = dict()
    for lang in langs:
        features = iterate_through_data(train_data, model, lang)
        file_name = "{lang}_features.json"
        file_path = os.path.join(data_directory, file_name)
        with open(file_path, 'w') as f:
            json.dump(features, f)
        