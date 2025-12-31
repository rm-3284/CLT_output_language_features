import json
import os

from ablation_amplification_intervention import (
    description_based_features,
    mean_value_based_features,
    freq_based_features,
)
from template import lang_to_flores_key

def intersection_across_langs(d):
    keys = sorted(list(d.keys()))
    result = dict()
    for base_lang in keys:
        result[base_lang] = dict()
        for intersection_lang in keys:
            base = set(d[base_lang])
            comparison = set(d[intersection_lang])
            intersection = base.intersection(comparison)
            result[base_lang][intersection_lang] = (len(base), len(intersection))
    return result

def intersection_across_dictionary(annSel, valSel, freqSel):
    langs = sorted(list(annSel.keys()))
    dictionary = {"AnnSel": annSel, "ValSel": valSel, "FreqSel": freqSel}
    result = dict()
    for method1, d1 in dictionary.items():
        result[method1] = dict()
        for method2, d2 in dictionary.items():
            result[method1][method2] = dict()
            for lang1 in langs:
                result[method1][method2][lang1] = dict()
                for lang2 in langs:
                    set1 = set(d1[lang1])
                    set2 = set(d2[lang2])
                    intersection = set1.intersection(set2)
                    result[method1][method2][lang1][lang2] = (len(set1), len(intersection))
    return result


if __name__ == "__main__":
    # relevant directories
    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    data_directory = os.path.join(absolute_directory, "data")
    flores_directory = os.path.join(data_directory, "flores_features")
    lang_specific_directory = os.path.join(data_directory, "language_specific_features")
    multilingual_features_directory = os.path.join(data_directory, "multilingual_llm_features")
    amplification_values_directory = os.path.join(data_directory, "amplification_values")

    langs = list(lang_to_flores_key.keys())

    # get the features + amplification values (dict[lang, list[str]])
    desc_features = description_based_features(flores_directory, langs, 0.1)
    val_features = mean_value_based_features(multilingual_features_directory, langs, 50)
    freq_features = freq_based_features(lang_specific_directory, langs)

    annSelLang = intersection_across_langs(desc_features)
    valSelLang = intersection_across_langs(val_features)
    freqSelLang = intersection_across_langs(freq_features)
    across_dict = intersection_across_dictionary(desc_features, val_features, freq_features)
    with open(os.path.join(absolute_directory, "intersections.json"), 'w') as f:
        json.dump("AnnSel intersection between languages", f, indent=4)
        f.write('\n')
        json.dump(annSelLang, f, indent=4)
        f.write('\n')
        json.dump("ValSel intersection between languages", f, indent=4)
        f.write('\n')
        json.dump(valSelLang, f, indent=4)
        f.write('\n')
        json.dump("FreqSel intersection between languages", f, indent=4)
        f.write('\n')
        json.dump(freqSelLang, f, indent=4)
        f.write('\n')
        json.dump("Intersection between different methods and languages", f, indent=4)
        f.write('\n')
        json.dump(across_dict, f, indent=4)
        f.write('\n')

