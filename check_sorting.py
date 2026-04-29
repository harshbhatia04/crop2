import json
import os

def check_sorting():
    with open("class_symptoms.json", "r") as f:
        symptoms_data = json.load(f)
        
    json_keys = sorted(list(symptoms_data.keys()))
    
    train_dir = "train"
    dir_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    
    print("JSON Keys (Sorted):")
    for i, k in enumerate(json_keys):
        print(f"{i}: {k}")
        
    print("\nDirectory Names (Sorted):")
    for i, d in enumerate(dir_names):
        print(f"{i}: {d}")
        
    if json_keys == dir_names:
        print("\nSUCCESS: Orders match!")
    else:
        print("\nFAILURE: Orders do NOT match!")
        for i in range(min(len(json_keys), len(dir_names))):
            if json_keys[i] != dir_names[i]:
                print(f"Mismatch at index {i}: JSON='{json_keys[i]}', DIR='{dir_names[i]}'")
                break

if __name__ == "__main__":
    check_sorting()
