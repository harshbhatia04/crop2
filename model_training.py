import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import json
import os
from PIL import Image

# Configuration
CONFIG = {
    "img_size": 300,
    "batch_size": 32,
    "lr": 1e-3,
    "epochs": 20,
    "num_classes": 13, 
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Image Transformations
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(CONFIG["img_size"]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(CONFIG["img_size"]),
    transforms.CenterCrop(CONFIG["img_size"]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom Dataset for VLM (Image + Symptom Text)
class CropVLMDataset(Dataset):
    def __init__(self, root_dir, symptoms_map, transform=None):
        self.dataset = datasets.ImageFolder(root=root_dir)
        self.symptoms_map = symptoms_map
        self.transform = transform
        self.classes = self.dataset.classes
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        img, label_idx = self.dataset[idx]
        class_name = self.classes[label_idx]
        symptom_text = self.symptoms_map.get(class_name, "Healthy plant")
        
        if self.transform:
            img = self.transform(img)
            
        return img, label_idx, symptom_text

# Vision Model (EfficientNet-B3)
def get_efficientnet_b3(num_classes):
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    # Modify the classifier
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

from sentence_transformers import SentenceTransformer
import time
import copy
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# VLM Architecture (Option A: Feature Fusion)
class CropVLMModel(nn.Module):
    def __init__(self, num_classes, text_embed_dim=384):
        super(CropVLMModel, self).__init__()
        # Vision Backbone
        self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        self.num_vision_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(self.num_vision_ftrs + text_embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, img, text_embed):
        vision_features = self.backbone(img)
        # Flatten vision features if needed (EfficientNet backbone output is usually already flat after GlobalAvgPool)
        fused_features = torch.cat((vision_features, text_embed), dim=1)
        logits = self.fusion(fused_features)
        return logits

# Training Function
def train_model(model, dataloaders, text_encoder, symptoms_map, criterion, optimizer, num_epochs=25, device='cuda'):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = torch.tensor(0).to(device)

            for inputs, labels, text_queries in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Get text embeddings for the symptoms
                with torch.no_grad():
                    # We encode the symptom descriptions into embeddings
                    text_embeds = text_encoder.encode(text_queries, convert_to_tensor=True).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, text_embeds)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def evaluate_model(model, dataloader, text_encoder, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels, text_queries in dataloader:
            inputs = inputs.to(device)
            text_embeds = text_encoder.encode(text_queries, convert_to_tensor=True).to(device)
            outputs = model(inputs, text_embeds)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print(classification_report(all_labels, all_preds))
    return all_labels, all_preds

if __name__ == "__main__":
    # Load symptoms map
    with open("class_symptoms.json", "r") as f:
        symptoms_full = json.load(f)
    
    # Initialize Dataset and Dataloaders
    train_dir = "dataset/train"
    val_dir = "dataset/val"
    
    # Get symptoms map for training classes ONLY
    dummy_dataset = datasets.ImageFolder(train_dir)
    train_classes = dummy_dataset.classes
    symptoms_map = {cls: symptoms_full.get(cls, "Healthy plant") for cls in train_classes}
    
    train_dataset = CropVLMDataset(train_dir, symptoms_map, transform=train_transforms)
    val_dataset = CropVLMDataset(val_dir, symptoms_map, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    dataloaders = {"train": train_loader, "val": val_loader}
    
    # Initialize Model
    num_classes = len(train_classes)
    device = CONFIG["device"]
    model = CropVLMModel(num_classes).to(device)
    
    # Initialize Text Encoder
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    
    # Train
    print(f"Starting training for {num_classes} classes...")
    model, history = train_model(model, dataloaders, text_encoder, symptoms_map, criterion, optimizer, num_epochs=CONFIG["epochs"], device=device)
    
    # Save Model
    torch.save(model.state_dict(), "best_model.pth")
    print("Model saved to best_model.pth")
    
    # Evaluate
    evaluate_model(model, val_loader, text_encoder, device=device)
