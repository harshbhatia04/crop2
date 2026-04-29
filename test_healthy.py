import torch
import os
from inference import CropDiseaseInference
from PIL import Image
import cv2

def test_healthy():
    predictor = CropDiseaseInference('best_model_vision.pth', 'class_symptoms.json', 17, device='cpu')
    
    classes_to_test = ['Potato_Healthy', 'Tomato_Healthy']
    test_root = 'test'
    
    for cls_name in classes_to_test:
        cls_path = os.path.join(test_root, cls_name)
        if not os.path.exists(cls_path):
            print(f"Directory not found: {cls_path}")
            continue
            
        imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"\nTesting {cls_name} ({len(imgs)} images):")
        
        correct = 0
        for img_name in imgs:
            img_path = os.path.join(cls_path, img_name)
            try:
                # Use the actual predictor logic
                result = predictor.predict_image(img_path)
                pred_class = f"{result['crop']}_{result['disease'].replace(' ', '_')}"
                
                if pred_class.lower() == cls_name.lower():
                    correct += 1
                else:
                    print(f"  FAILED: {img_name} -> Predicted: {pred_class} (Conf: {result['confidence']})")
            except Exception as e:
                print(f"  ERROR: {img_name} -> {e}")
                
        print(f"Result for {cls_name}: {correct}/{len(imgs)}")

if __name__ == "__main__":
    test_healthy()
