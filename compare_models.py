import torch
import os
from inference import CropDiseaseInference
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def evaluate_model(model_path, data_dir):
    print(f"\n--- Testing Model: {model_path} ---")
    predictor = CropDiseaseInference(model_path, "class_symptoms.json", 17)
    
    val_transforms = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder(data_dir, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    correct = 0
    total = 0
    
    predictor.model.eval()
    with torch.no_grad():
        count = 0
        for inputs, labels in val_loader:
            if count > 500: break # Quick sample check
            inputs = inputs.to(predictor.device)
            labels = labels.to(predictor.device)
            
            outputs = predictor.model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            count += labels.size(0)
            
    accuracy = 100 * correct / total
    print(f"Final Accuracy on Validation Set: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    val_dir = "dataset_final/val"
    
    old_model = "best_model_vision.pth"
    new_model = "best_model_vision_autosave.pth"
    
    acc_old = evaluate_model(old_model, val_dir)
    acc_new = evaluate_model(new_model, val_dir)
    
    print("\n" + "="*30)
    print(f"OLD MODEL ACCURACY: {acc_old:.2f}%")
    print(f"NEW MODEL ACCURACY: {acc_new:.2f}%")
    print("="*30)
    
    if acc_new > acc_old:
        print("WINNER: NEW MODEL! Pushing to production...")
    else:
        print("WINNER: OLD MODEL! Reverting to keep performance.")
