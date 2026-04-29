import torch
import os
from inference import CropDiseaseInference
import json

def test():
    predictor = CropDiseaseInference('best_model_vision.pth', 'class_symptoms.json', 17, device='cpu')
    test_dir = 'test'
    
    # Get ground truth classes from folders
    classes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    
    print(f"Classes found in test dir: {classes}")
    print(f"Classes in predictor: {predictor.class_names}")
    
    correct = 0
    total = 0
    
    for cls in classes:
        cls_path = os.path.join(test_dir, cls)
        # Get up to 5 images per class
        imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]
        for img_name in imgs:
            img_path = os.path.join(cls_path, img_name)
            try:
                result = predictor.predict_image(img_path)
                # Reconstruct class name from result
                pred_crop = result["crop"]
                pred_disease = result["disease"].replace(" ", "_")
                pred_class = f"{pred_crop}_{pred_disease}"
                
                # Check for match (handling potential naming differences like Septoria_leaf_spot vs Septoria_Leaf_Spot)
                if pred_class.lower() == cls.lower():
                    correct += 1
                else:
                    print(f"Mismatch: Actual={cls}, Predicted={pred_class} (Confidence: {result['confidence']})")
                total += 1
            except Exception as e:
                print(f"Error predicting {img_path}: {e}")

    if total > 0:
        print(f"\nFinal Sample Accuracy: {correct}/{total} ({correct/total*100:.2f}%)")
    else:
        print("No images found to test.")

if __name__ == "__main__":
    test()
