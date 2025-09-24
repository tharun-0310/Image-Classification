import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
from scipy.fftpack import dct
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}, name: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")

# Custom transform to apply DCT
class DCTTransform:
    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, img):
        img = img.resize(self.size)
        img_np = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Apply DCT on each channel
        dct_channels = []
        for c in range(3):
            dct_channel = dct(dct(img_np[:, :, c], axis=0, norm='ortho'), axis=1, norm='ortho')
            dct_channels.append(dct_channel)

        dct_image = np.stack(dct_channels, axis=0)  # Shape: (3, H, W)
        return torch.tensor(dct_image, dtype=torch.float32)

# Dataset loading
transform = DCTTransform(size=(256, 256))

train_data = datasets.ImageFolder(r"Dataset\dataset_split\train", transform=transform)
val_data   = datasets.ImageFolder(r"Dataset\dataset_split\val", transform=transform)
test_data  = datasets.ImageFolder(r"Dataset\dataset_split\test", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=32)
test_loader  = DataLoader(test_data, batch_size=32)

num_classes = len(train_data.classes)

# Simple CNN model (you can replace with a better one if needed)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 64 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = SimpleCNN(num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Tracking metrics
train_acc_history = []
val_acc_history = []
train_loss_history = []

# Evaluation function
def evaluate_model(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

# Training function
def train_model(model, train_loader, val_loader, epochs):
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=correct / total)

        train_acc = correct / total
        val_acc = evaluate_model(model, val_loader)

        print(f"✅ Epoch {epoch+1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_loss_history.append(total_loss / len(train_loader))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_dct_model.pth")
            print("💾 Best model saved!")

# Train the model
train_model(model, train_loader, val_loader, epochs=10)

# Test evaluation
model.load_state_dict(torch.load("best_dct_model.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

print("📊 Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=train_data.classes))

# 📈 Plotting graphs
plt.figure(figsize=(10, 5))
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs (DCT)')
plt.legend()
plt.grid()
plt.savefig("accuracy_curve_dct.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label='Train Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs (DCT)')
plt.legend()
plt.grid()
plt.savefig("loss_curve_dct.png")
plt.close()

print("✅ Plots saved: 'accuracy_curve_dct.png' and 'loss_curve_dct.png'")