import json
import os
from pathlib import Path
import requests

from template import identifiers

if __name__ == "__main__":
    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    out_dir = os.path.join(absolute_directory, "data/sae_features")
    out_dir = Path(out_dir)

    os.makedirs((out_dir / "description"), exist_ok=True)

    for item in out_dir.glob("*.json"):
        lang = str(item.stem)
        with open(item, "r") as f:
            tuples = json.load(f)

        description_dict = dict()
        
        for (layer, feature_idx), _, _ in tuples:
            response = requests.get(f"https://www.neuronpedia.org/api/feature/gemma-2-2b/{layer}-gemmascope-mlp-16k/{feature_idx}")
            explanations = response.json()['explanations']
            try:
                description = explanations[0]['description']
            except IndexError:
                print(f"layer{layer}, feature{feature_idx}, unable to retrieve description")
                description = ""
            description_dict[f"{layer}.{feature_idx}"] = description

        file_path = (out_dir / f"description/{lang}.json")
        with open(file_path, "w") as f:
            json.dump(description_dict, f)
    
    description_included = dict()
    for item in (out_dir / "description").glob("??.json"):
        lang = str(item.stem)
        with open(item, 'r') as f:
            descriptions = json.load(f)
        
        total = 0
        included = 0
        
        for _, val in descriptions.items():
            ok = False
            for identifier in identifiers[lang]:
                if identifier in val:
                    ok = True
            total += 1
            if ok:
                included += 1
            
        description_included[lang] = (included, total)
    
    file_path = (out_dir / "description/summary.json")
    with open(file_path, "w") as f:
        json.dump(description_included, f)
