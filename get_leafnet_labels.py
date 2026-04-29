from datasets import load_dataset_builder

ds_builder = load_dataset_builder("enalis/LeafNet")
features = ds_builder.info.features['label']
# print(features)
# print(features.names)

mapping = {}
for i, name in enumerate(features.names):
    mapping[i] = name

import json
with open("leafnet_mapping.json", "w") as f:
    json.dump(mapping, f, indent=4)
print("Saved leafnet_mapping.json")
