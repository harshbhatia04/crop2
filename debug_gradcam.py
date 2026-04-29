import torch
import cv2
from PIL import Image
from inference import CropDiseaseInference
import os

def check_gradcam_healthy():
    predictor = CropDiseaseInference('best_model_vision.pth', 'class_symptoms.json', 17, device='cpu')
    
    img_path = r"test\Potato_Healthy\00fc2ee5-729f-4757-8aeb-65c3355874f2___RS_HL 1864.JPG"
    output_path = "static/debug_healthy_gradcam.jpg"
    
    print(f"Generating Grad-CAM for: {img_path}")
    result = predictor.predict_image(img_path, visual_output=output_path)
    print(f"Result: {result}")
    print(f"Grad-CAM saved to {output_path}")

if __name__ == "__main__":
    check_gradcam_healthy()
