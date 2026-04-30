import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import json

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class CropDiseaseInference:
    def __init__(self, model_path, symptoms_json, num_classes, device='cuda'):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        self.model = models.efficientnet_b3()
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
                print(f"Model loaded successfully from {model_path}")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load weights: {e}")
            
        self.model.to(self.device)
        self.model.eval()
        
        with open(symptoms_json, 'r', encoding='utf-8') as f:
            self.symptoms_data = json.load(f)
            
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.class_names = sorted(list(self.symptoms_data.keys()))
        self.target_layers = [self.model.features[-1]]

    def predict_image(self, img_path, visual_output="static/gradcam.jpg"):
        img_orig = cv2.imread(img_path)
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_orig)
        
        tta_transforms = [
            lambda x: x,
            lambda x: transforms.functional.hflip(x),
            lambda x: transforms.functional.vflip(x),
            lambda x: transforms.functional.rotate(x, 15),
            lambda x: transforms.functional.rotate(x, -15)
        ]
        
        all_probs = []
        with torch.no_grad():
            for t in tta_transforms:
                augmented_img = t(img_pil)
                img_tensor = self.transform(augmented_img).unsqueeze(0).to(self.device)
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs)
        
        avg_probs = torch.mean(torch.stack(all_probs), dim=0)
        probs = avg_probs
        
        conf, pred_idx = torch.max(probs, 1)
        class_name = self.class_names[pred_idx.item()]
        
        class_data = self.symptoms_data.get(class_name, {})
        symptom = class_data.get("symptom", "No symptoms found.")
        organic = class_data.get("organic", "No organic treatment found.")
        chemical = class_data.get("chemical", "No chemical treatment found.")
        danger = class_data.get("danger", "UNKNOWN")
        
        cam = GradCAM(model=self.model, target_layers=self.target_layers)
        targets = [ClassifierOutputTarget(pred_idx.item())]
        
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]
        
        img_for_overlay = cv2.resize(img_orig, (300, 300)) / 255.0
        visualization = show_cam_on_image(img_for_overlay, grayscale_cam, use_rgb=True)
        
        import os
        os.makedirs(os.path.dirname(visual_output), exist_ok=True)
        cv2.imwrite(visual_output, cv2.cvtColor((visualization * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        split = class_name.split("_", 1)
        web_path = f"/{visual_output}".replace('\\', '/')
        
        return {
            "crop": split[0],
            "disease": split[1].replace("_", " "),
            "confidence": f"{conf.item() * 100:.2f}%",
            "symptoms": symptom,
            "organic_treatment": organic,
            "chemical_treatment": chemical,
            "danger_level": danger,
            "visual_explanation": web_path
        }

    def predict_video(self, video_path, output_path="output_detected.mp4", skip_frames=5):
        cap = cv2.VideoCapture(video_path)
        width, height = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps / (skip_frames + 1), (width, height))
        
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if count % (skip_frames + 1) == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    _, pred_idx = torch.max(probs, 1)
                
                class_name = self.class_names[pred_idx.item()]
                label = class_name.replace("_", " ")
                cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(frame)
            count += 1
            
        cap.release()
        out.release()
        return output_path
