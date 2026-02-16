import argparse
import json
import os

from template import langs
from models import hf_model_names

def argsparse():
    parser = argparse.ArgumentParser(description='Check overlap of features between FLORES and antonym datasets')
    parser.add_argument('--model', type=str, default='gemma-2-2b', choices=hf_model_names.keys(), help='Model to use for the experiment')
    return parser.parse_args()

if __name__ == "__main__":
    args = argsparse()
    model_name = args.model

    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    data_directory = os.path.join(absolute_directory, "data")

    flores_directory = os.path.join(data_directory, "flores_features", model_name)
    antonym_features_directory = os.path.join(data_directory, "features")
    
    for lang in langs:
        with open(os.path.join(flores_directory, f"{lang}_features.json"), 'r') as f:
            flores_data = json.load(f)
        with open(os.path.join(antonym_features_directory, f"{lang}_features.json"), 'r') as f:
            antonym_data = json.load(f)
        flores_keys = flores_data.keys()
        antonym_keys = antonym_data.keys()
        print(lang)
        print(f"statistics: flores keys {len(flores_keys)}, antonym keys {len(antonym_keys)}")
        
        overlap_set = set(flores_data).intersection(set(antonym_data))
        print(f"overlap size {len(overlap_set)}")
        for duplicate in overlap_set:
            print(duplicate, flores_data[duplicate], antonym_data[duplicate])
