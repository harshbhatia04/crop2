import os
import json

# Order from ImageFolder (Alphabetical)
dataset_classes = sorted([d for d in os.listdir("dataset_final/train") if os.path.isdir(os.path.join("dataset_final/train", d))])

# Order from Symptoms JSON
with open("class_symptoms.json", "r", encoding="utf-8") as f:
    symptoms_data = json.load(f)
symptoms_classes = sorted(list(symptoms_data.keys()))

print("--- DATASET CLASSES ---")
for i, c in enumerate(dataset_classes):
    print(f"{i}: {c}")

print("\n--- SYMPTOMS JSON CLASSES ---")
for i, c in enumerate(symptoms_classes):
    print(f"{i}: {c}")

if dataset_classes == symptoms_classes:
    print("\n✅ Order Matches Perfectly!")
else:
    print("\n❌ ORDER MISMATCH FOUND!")
