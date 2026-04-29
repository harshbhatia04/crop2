import torch
import cv2
from PIL import Image
from inference import CropDiseaseInference
import json

def debug_one_image():
    predictor = CropDiseaseInference('best_model_vision.pth', 'class_symptoms.json', 17, device='cpu')
    
    # Use an image that definitely exists
    img_path = r"test\Tomato_Healthy\000146ff-92a4-4db6-90ad-8fce2ae4fddd___GH_HL Leaf 259.1.JPG"
    
    print(f"Testing image: {img_path}")
    
    img_orig = cv2.imread(img_path)
    if img_orig is None:
        print("FAILED TO READ IMAGE WITH CV2")
        # Try PIL
        try:
            img_pil = Image.open(img_path).convert('RGB')
            print("READ IMAGE WITH PIL SUCCESS")
        except Exception as e:
            print(f"FAILED TO READ IMAGE WITH PIL: {e}")
            return
    else:
        print("READ IMAGE WITH CV2 SUCCESS")
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_orig)

    img_tensor = predictor.transform(img_pil).unsqueeze(0).to(predictor.device)
    
    with torch.no_grad():
        outputs = predictor.model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)
        
    print(f"Probabilities: {probs[0].tolist()}")
    print(f"Predicted index: {pred_idx.item()}")
    print(f"Predicted class: {predictor.class_names[pred_idx.item()]}")
    print(f"Confidence: {conf.item()}")

if __name__ == "__main__":
    debug_one_image()
