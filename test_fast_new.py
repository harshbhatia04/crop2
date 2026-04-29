import torch
import os
from inference import CropDiseaseInference
import json

def test():
    predictor = CropDiseaseInference('best_model_vision.pth', 'class_symptoms.json', 17, device='cpu')
    test_dir = 'test'
    
    classes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    
    correct = 0
    total = 0
    
    print(f"Starting fast test with new transforms...")
    
    for cls in classes:
        cls_path = os.path.join(test_dir, cls)
        imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]
        for img_name in imgs:
            img_path = os.path.join(cls_path, img_name)
            try:
                # Bypass Grad-CAM for speed
                import cv2
                from PIL import Image
                img_orig = cv2.imread(img_path)
                img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_orig)
                img_tensor = predictor.transform(img_pil).unsqueeze(0).to(predictor.device)
                
                with torch.no_grad():
                    outputs = predictor.model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    _, pred_idx = torch.max(probs, 1)
                
                pred_class = predictor.class_names[pred_idx.item()]
                
                if pred_class.lower() == cls.lower():
                    correct += 1
                else:
                    print(f"Mismatch: Actual={cls}, Predicted={pred_class}")
                total += 1
            except Exception as e:
                print(f"Error predicting {img_path}: {e}")

    if total > 0:
        print(f"\nFinal Accuracy: {correct}/{total} ({correct/total*100:.2f}%)")
    else:
        print("No images found.")

if __name__ == "__main__":
    test()
