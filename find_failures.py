import torch
import os
from inference import CropDiseaseInference
from PIL import Image
import cv2

def find_failures():
    predictor = CropDiseaseInference('best_model_vision.pth', 'class_symptoms.json', 17, device='cpu')
    
    classes_to_test = ['Potato_Healthy', 'Tomato_Healthy']
    test_root = 'val'
    
    for cls_name in classes_to_test:
        cls_path = os.path.join(test_root, cls_name)
        if not os.path.exists(cls_path):
            print(f"Directory not found: {cls_path}")
            continue
            
        imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Testing {len(imgs)} images in {cls_name}...")
        
        failures = 0
        for img_name in imgs:
            img_path = os.path.join(cls_path, img_name)
            try:
                # Direct prediction without Grad-CAM for speed
                img_orig = cv2.imread(img_path)
                img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_orig)
                img_tensor = predictor.transform(img_pil).unsqueeze(0).to(predictor.device)
                
                with torch.no_grad():
                    outputs = predictor.model(img_tensor)
                    _, pred_idx = torch.max(outputs, 1)
                
                pred_class = predictor.class_names[pred_idx.item()]
                
                if pred_class.lower() != cls_name.lower():
                    print(f"  FAILURE: {img_name} -> Predicted: {pred_class}")
                    failures += 1
            except Exception as e:
                pass
                
        print(f"Total failures for {cls_name}: {failures}/{len(imgs)}")

if __name__ == "__main__":
    find_failures()
