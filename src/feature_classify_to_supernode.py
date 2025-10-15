import math
import statistics

from template import Feature, langs

def count_NaN(lst: list[float]) -> tuple[int, list[float]]:
    n_lst = []
    count = len(lst)
    for value in lst:
        if math.isnan(value):
            count -= 1
        else:
            n_lst.append(value)
    return count, n_lst

def classify_feature(lang_values: dict[str, tuple[int, float, float]]) -> list[str]:
    # input is dict[lang, [nan_count, median, mean]]
    lst = []
    for key, value in lang_values.items():
        if value[0] == 0:
            continue
        else:
            lst.append(key)
    if len(lst) == 1:
        return lst
    else:
        val_lst = []
        for key in lst:
            val_lst.append(lang_values[key][0])
        if not val_lst:
            return val_lst
        threshold = 0.1 * max(val_lst)
        idx_lst = [index for index, value in enumerate(val_lst) if value > threshold]
        n_lst = []
        for idx in idx_lst:
            n_lst.append(lst[idx])
        # returns which language it should belong to
        return n_lst

def classify_features_with_values(features: list[str], values_dict: dict[str, dict[str, list[float]]]) -> dict[str, list[tuple[str, float]]]:
    # return type is dictionary with key of language and the value of Feature, the value for amplification (ablation is just 0)
    feature_dict = dict()
    for feature in features:
        feature_dict[feature] = dict()
    for key, value in values_dict.items():
        for feature, float_list in value.items():
            nan_count, filtered_list = count_NaN(float_list)
            if filtered_list:
                median = statistics.median(filtered_list)
                mean = statistics.mean(filtered_list)
            else:
                median = 0
                mean = 0
            feature_dict[feature][key] = (nan_count, median, mean)

    return_dict = dict()
    for lang in langs:
        return_dict[lang] = []
    for feature in features:
        which_supernode = classify_feature(feature_dict[feature])
        for lang in which_supernode:
            return_dict[lang].append((feature, feature_dict[feature][lang][1]))
    # median is the amplification value
    return return_dict

if __name__ == "__main__":
    import os, json
    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)

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

    value_directory = os.path.join(absolute_directory, "data/values")
    with open(os.path.join(value_directory, 'en_values.json'), 'r') as f:
        en_values = json.load(f)
    with open(os.path.join(value_directory, 'de_values.json'), 'r') as f:
        de_values = json.load(f)
    with open(os.path.join(value_directory, 'fr_values.json'), 'r') as f:
        fr_values = json.load(f)
    with open(os.path.join(value_directory, 'ja_values.json'), 'r') as f:
        ja_values = json.load(f)
    with open(os.path.join(value_directory, 'zh_values.json'), 'r') as f:
        zh_values = json.load(f)
    values_dict = {'en': en_values, 'de': de_values, 'fr': fr_values, 'ja': ja_values, 'zh': zh_values}

    supernodes_dict = classify_features_with_values(features, values_dict)
    supernode_directory = os.path.join(absolute_directory, "data/supernodes")
    if not os.path.exists(supernode_directory):
        os.makedirs(supernode_directory)
    for key, val in supernodes_dict.items():
        file_name = key + "_supernode.json"
        with open(os.path.join(supernode_directory, file_name), 'w') as f:
            json.dump(val, f)
    
