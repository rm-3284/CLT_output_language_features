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
        return n_lst

def classify_features_with_values(features: list[Feature], values_dict: dict[str, dict[Feature, list[float]]]) -> dict[str, list[tuple[Feature, float]]]:
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
    return return_dict
