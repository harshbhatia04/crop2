import torch
from inference import CropDiseaseInference
import os

def check_late_blight():
    p = CropDiseaseInference('best_model_vision.pth', 'class_symptoms.json', 17, device='cpu')
    test_dir = 'dataset_final/test/Tomato_Late_Blight'
    
    if not os.path.exists(test_dir):
        print(f"Directory not found: {test_dir}")
        return
        
    imgs = [f for f in os.listdir(test_dir) if f.lower().endswith('.jpg')][:10]
    print(f"Testing {len(imgs)} images of Tomato_Late_Blight...")
    
    for img in imgs:
        path = os.path.join(test_dir, img)
        res = p.predict_image(path)
        print(f"Image: {img}")
        print(f"  Result: {res['crop']} {res['disease']} (Conf: {res['confidence']})")

if __name__ == "__main__":
    check_late_blight()
