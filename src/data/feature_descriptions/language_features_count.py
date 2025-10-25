import json
import os
from pathlib import Path
import sys

current_file_path = __file__
current_directory = os.path.dirname(current_file_path)
absolute_directory = os.path.abspath(current_directory)

parent_directory = Path(absolute_directory).parent
grand_parent_directory = parent_directory.parent
sys.path.insert(0, str(grand_parent_directory))
from template import identifiers

all_entries = os.listdir(absolute_directory)
result = dict()
for file in all_entries:
    root, extention = os.path.splitext(file)
    if extention != '.json':
        continue
    lang = root.split('_')[-1]
    
    total = 0
    lang_included = 0
    with open(os.path.join(absolute_directory, file), 'r') as f:
        descriptions = json.load(f)
    
    for _, v in descriptions.items():
        for _, description in v.items():
            total += 1
            default = False
            for identifier in identifiers[lang]:
                included = identifier in description
                default = default or included
            if default:
                lang_included += 1
    
    result[file] = (lang_included, total)

with open(os.path.join(absolute_directory, 'lang_count.json'), 'w') as f:
    json.dump(result, f)
