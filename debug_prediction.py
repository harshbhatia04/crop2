import torch
import json
import os
from inference import CropDiseaseInference

def debug():
    model_path = "best_model_vision.pth"
    symptoms_json = "class_symptoms.json"
    num_classes = 17
    
    predictor = CropDiseaseInference(model_path, symptoms_json, num_classes)
    
    print("Class names in inference:")
    for i, name in enumerate(predictor.class_names):
        print(f"{i}: {name}")
        
    test_images = [
        r"D:\crop\test\Corn_Common_Rust\ln_ln_003965_Corn_Common_Rust (960).JPG.jpg",
        r"D:\crop\test\Tomato_Healthy\000146ff-92a4-4db6-90ad-8fce2ae4fddd___GH_HL Leaf 259.1.JPG"
    ]
    
    for test_image in test_images:
        if os.path.exists(test_image):
            print(f"\nPredicting for image: {test_image}")
            result = predictor.predict_image(test_image)
            print(f"Result: {result}")
        else:
            print(f"\nTest image not found: {test_image}")

if __name__ == "__main__":
    debug()
