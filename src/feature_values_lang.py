import torch
from typing import Optional

from template import Graph, Feature, attribute, ReplacementModel
from feature_values_for_generic_sentences import iterate_over_sentences

if __name__ == "__main__":
    import sys
    lang = sys.argv[1]

    from device_setup import device
    model_name = 'google/gemma-2-2b'
    transcoder_name = 'gemma'
    model = ReplacementModel.from_pretrained(model_name, transcoder_name, device=device, dtype=torch.bfloat16)

    import os, json
    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    data_directory = os.path.join(absolute_directory, "data/generic_sentences")

    with open(os.path.join(data_directory, lang + 'sentences.json'), 'r') as f:
        sentences = json.load(f)
    
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
    
    en_values = iterate_over_sentences(sentences, features, model, device)

    values_directory = os.path.join(absolute_directory, 'data/values')
    if not os.path.exists(values_directory):
        os.makedirs(values_directory)

    with open(os.path.join(values_directory, lang + '_values_individual.json'), 'w') as f:
        json.dump(en_values, f)
    