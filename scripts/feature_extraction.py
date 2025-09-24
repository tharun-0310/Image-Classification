import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import classification_report
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}, name: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")

# Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Dataset loading
train_data = datasets.ImageFolder(r"Dataset\dataset_split\train", transform=transform)
val_data   = datasets.ImageFolder(r"Dataset\dataset_split\val", transform=transform)
test_data  = datasets.ImageFolder(r"Dataset\dataset_split\test", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=32)
test_loader  = DataLoader(test_data, batch_size=32)

num_classes = len(train_data.classes)

# Model setup
model = timm.create_model("mobilevitv2_050", pretrained=True, num_classes=0)
model.reset_classifier(num_classes=num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# For graph tracking
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
            torch.save(model.state_dict(), "best_mobilevitv2_model.pth")
            print("💾 Best model saved!")

# Train model
train_model(model, train_loader, val_loader, epochs=10)

# Test evaluation
model.load_state_dict(torch.load("best_mobilevitv2_model.pth"))
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
plt.title('Accuracy over Epochs')
plt.legend()
plt.grid()
plt.savefig("accuracy_curve.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label='Train Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid()
plt.savefig("loss_curve.png")
plt.close()

print("✅ Plots saved: 'accuracy_curve.png' and 'loss_curve.png'")
