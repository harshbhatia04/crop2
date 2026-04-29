import torch
from inference import CropDiseaseInference
import os
import random

def debug_mismatch(image_dir, target_class, model_path):
    predictor = CropDiseaseInference(model_path, "class_symptoms.json", 17)
    
    # Get 5 random images from the folder the model IS getting wrong
    full_path = os.path.join(image_dir, target_class)
    images = [f for f in os.listdir(full_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    sample = random.sample(images, min(5, len(images)))
    
    print(f"\n--- DEBUGGING {target_class} ---")
    for img_name in sample:
        img_path = os.path.join(full_path, img_name)
        result = predictor.predict_image(img_path)
        print(f"File: {img_name} | Predicted: {result['disease']} | Confidence: {result['confidence']}")

if __name__ == "__main__":
    # Test images from the actual "Yellow Leaf Curl" folder
    debug_mismatch("dataset_final/val", "Tomato_Yellow_Leaf_Curl_Virus", "best_model_vision_autosave.pth")
