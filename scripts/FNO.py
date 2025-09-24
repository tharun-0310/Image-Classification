import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}, name: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")

# === Dataset ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(r"Dataset\dataset_split\train", transform=transform)
val_data   = datasets.ImageFolder(r"Dataset\dataset_split\val", transform=transform)
test_data  = datasets.ImageFolder(r"Dataset\dataset_split\test", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=32)
test_loader  = DataLoader(test_data, batch_size=32)

num_classes = len(train_data.classes)

# === SpectralConv2d (Fixed) ===
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixyk,ioxyk->boxyk", input, weights)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm='ortho')  # [B, C, H, W//2+1]
        x_ft_real = torch.view_as_real(x_ft[:, :, :self.modes1, :self.modes2]).contiguous()

        out_ft = torch.zeros(B, self.out_channels, H, W//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.view_as_complex(
            self.compl_mul2d(x_ft_real, self.weights).contiguous()
        )

        x = torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')
        return x

# === FNO Model ===
class FNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, width=32, modes1=16, modes2=16):
        super().__init__()
        self.width = width
        self.fc0 = nn.Linear(in_channels, width)
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Linear(width * 128 * 128, 256)
        self.fc2 = nn.Linear(256, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # [B, width, H, W]

        x = self.conv0(x) + self.w0(x)
        x = self.conv1(x) + self.w1(x)
        x = self.conv2(x) + self.w2(x)
        x = self.conv3(x) + self.w3(x)

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# === Model Setup ===
model = FNO2d(in_channels=3, out_channels=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_acc_history = []
val_acc_history = []
train_loss_history = []

# === Evaluation ===
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

# === Training Loop ===
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
            torch.save(model.state_dict(), "best_fno_model.pth")
            print("📂 Best model saved!")

# === Run Training ===
train_model(model, train_loader, val_loader, epochs=10)

# === Test Model ===
model.load_state_dict(torch.load("best_fno_model.pth"))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

print("📊 Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=train_data.classes))

# === Accuracy Plot ===
plt.figure(figsize=(10, 5))
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs (FNO)')
plt.legend()
plt.grid()
plt.savefig("accuracy_curve_fno.png")
plt.close()

# === Loss Plot ===
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label='Train Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs (FNO)')
plt.legend()
plt.grid()
plt.savefig("loss_curve_fno.png")
plt.close()

print("✅ Plots saved: 'accuracy_curve_fno.png' and 'loss_curve_fno.png'")
