import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image, ImageFile
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import GradScaler, autocast
import os
import joblib
import boto3
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

# Enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# SageMaker paths
input_data_path = '/opt/ml/input/data/training/'
csv_path = os.path.join(input_data_path, 'the name of your dataset.csv') # replace with your value
image_folder = input_data_path
model_dir = '/opt/ml/model/'

# Hyperparameters
batch_size = 256
num_epochs_phase1 = 5
num_epochs_phase2 = 15
lr_phase1 = 1e-3
lr_phase2 = 3e-5
val_split = 0.2
patience = 5

# Dataset class
class LocalImageDataset(Dataset):
    def __init__(self, csv_path, image_folder, transform=None, label_encoder=None):
        self.image_folder = image_folder
        self.transform = transform
        self.label_encoder = label_encoder
        self.df = pd.read_csv(csv_path)
        self.image_keys = self.df['image_path'].tolist() # replace the name of column to yours
        self.labels = self.df['music_tag'].tolist() # replace the name of class column to yours 
        if self.label_encoder:
            self.labels = self.label_encoder.transform(self.labels)
    def __len__(self):
        return len(self.image_keys)
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_keys[idx])
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                if self.transform:
                    img = self.transform(img)
        except Exception:
            img = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                img = self.transform(img)
        label = self.labels[idx]
        return img, label

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load CSV and encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(pd.read_csv(csv_path)['music_tag']) # replace the name of class column to yours 
num_classes = len(label_encoder.classes_)
joblib.dump(label_encoder, 'label_encoder.pkl')
boto3.client('s3').upload_file('label_encoder.pkl', 'your bucket name', 'data/label_encoder.pkl') # replace with your true bucket name

# Dataset and Dataloaders
full_dataset = LocalImageDataset(csv_path, image_folder, label_encoder=label_encoder)
train_size = int((1 - val_split) * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# Define model
model = models.resnet34(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, num_classes))
for param in model.fc.parameters():
    param.requires_grad = True
model.to(device)

# Loss, optimizer, scheduler
class_weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(encoded_labels), y=encoded_labels), dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
scaler = GradScaler()


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, phase):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses, val_losses, val_accuracies = [], [], []
    val_f1_scores = []


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss, all_preds, all_labels = 0, [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1_scores.append(val_f1)

        print(f"Phase {phase} - Epoch [{epoch+1}/{num_epochs}]  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f} Val F1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(model_dir, f'best_model_phase{phase}.pth'))
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step(val_loss)

    return train_losses, val_losses, val_accuracies

# Phase 1: train classifier only
losses1, val_losses1, accs1 = train_model(model, train_loader, val_loader, num_epochs_phase1, lr_phase1, phase=1)

# Phase 2: unfreeze deeper layers and fine-tune
model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model_phase1.pth')))
for name, param in model.named_parameters():
    if any(layer in name for layer in ["layer2", "layer3", "layer4", "fc"]):
        param.requires_grad = True
    else:
        param.requires_grad = False

losses2, val_losses2, accs2 = train_model(model, train_loader, val_loader, num_epochs_phase2, lr_phase2, phase=2)

# Save final model
torch.save(model.state_dict(), os.path.join(model_dir, 'model_final.pth'))

# Plot
plt.plot(losses1 + losses2, label='Train Loss')
plt.plot(val_losses1 + val_losses2, label='Val Loss')
plt.legend()
plt.title('Loss Curves')
plt.savefig(os.path.join(model_dir, 'loss_plot.png'))
plt.show()

plt.plot(accs1 + accs2, label='Val Accuracy', color='green')
plt.title('Validation Accuracy')
plt.legend()
plt.savefig(os.path.join(model_dir, 'val_accuracy_plot.png'))
plt.show()
