import torch
from inference import CropDiseaseInference
import os

predictor = CropDiseaseInference('best_model_vision_autosave.pth', 'class_symptoms.json', 17)
path = 'dataset_final/val/Tomato_Healthy'
imgs = [f for f in os.listdir(path) if f.endswith('.JPG')][:5]

print("\n--- TESTING HEALTHY IMAGES ---")
for img in imgs:
    res = predictor.predict_image(os.path.join(path, img))
    print(f"File: {img} | Pred: {res['disease']} | Conf: {res['confidence']}")
