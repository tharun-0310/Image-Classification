# convnext_train.py
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import platform

# -----------------------
# Config
# -----------------------
DATA_DIR = r"Dataset\dataset_split"  # expects train/val/test subfolders
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-5
MODEL_SAVE_PATH = "best_convnext_tiny.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True if torch.cuda.is_available() else False

# Set NUM_WORKERS safely (on Windows you can still use >0 but wrapper will protect it).
# You can set to 0 to avoid multiprocessing while debugging.
NUM_WORKERS = 4

# If True: freeze backbone and only train classifier head for a few epochs (useful for small datasets)
FREEZE_BACKBONE = False

# For reproducibility (optional)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Platform: {platform.system()}, Using device: {DEVICE}, name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# -----------------------
# Transforms / Data
# -----------------------
# ImageNet mean/std for pretrained convnext
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# -----------------------
# Evaluation function
# -----------------------
def evaluate_model(model, loader, criterion, device):
    model.eval()
    correct, total = 0, 0
    loss_sum = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = loss_sum / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return acc, avg_loss

# -----------------------
# Training loop
# -----------------------
def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, scheduler, device):
    best_val_acc = 0.0
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=(correct/total))

        epoch_train_loss = total_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0
        val_acc, val_loss = evaluate_model(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        print(f"✅ Epoch {epoch+1}: Train Loss={epoch_train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_loss_history.append(epoch_train_loss)

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_acc": val_acc
            }, MODEL_SAVE_PATH)
            print(f"💾 New best model saved (val_acc={val_acc:.4f}) -> {MODEL_SAVE_PATH}")

    print("Training complete.")
    return train_acc_history, val_acc_history, train_loss_history

# -----------------------
# Main entry (safe for Windows multiprocessing)
# -----------------------
if __name__ == "__main__":
    # Create datasets & loaders inside main to avoid multiprocessing import issues
    train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_data   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_test_transform)
    test_data  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_test_transform)

    # If dataset folders not found, raise clearer error
    if not os.path.isdir(os.path.join(DATA_DIR, "train")):
        raise FileNotFoundError(f"Train folder not found: {os.path.join(DATA_DIR, 'train')}")
    if not os.path.isdir(os.path.join(DATA_DIR, "val")):
        raise FileNotFoundError(f"Val folder not found: {os.path.join(DATA_DIR, 'val')}")
    if not os.path.isdir(os.path.join(DATA_DIR, "test")):
        raise FileNotFoundError(f"Test folder not found: {os.path.join(DATA_DIR, 'test')}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    num_classes = len(train_data.classes)
    print(f"Found {num_classes} classes: {train_data.classes}")

    # -----------------------
    # Model: ConvNeXt-Tiny
    # -----------------------
    model = timm.create_model("convnext_tiny", pretrained=True, num_classes=num_classes)
    model = model.to(DEVICE)

    # Optional: Freeze backbone (everything except classifier head)
    if FREEZE_BACKBONE:
        for name, param in model.named_parameters():
            if "head" not in name and "fc" not in name and "classifier" not in name:
                param.requires_grad = False
        print("Backbone frozen — training classifier head only.")
    else:
        print("Finetuning entire ConvNeXt-Tiny model.")

    # -----------------------
    # Loss, optimizer, scheduler
    # -----------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # -----------------------
    # Train
    # -----------------------
    train_acc_history, val_acc_history, train_loss_history = train_model(
        model, train_loader, val_loader, NUM_EPOCHS, criterion, optimizer, scheduler, DEVICE
    )

    # -----------------------
    # Load best model & Test / Final evaluation
    # -----------------------
    if os.path.exists(MODEL_SAVE_PATH):
        ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best model from epoch {ckpt.get('epoch', '?')} with val_acc={ckpt.get('val_acc', '?'):.4f}")
    else:
        print("Warning: saved model not found, using current model weights for evaluation.")

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    print("\n📊 Classification Report (test set):\n")
    print(classification_report(all_labels, all_preds, target_names=train_data.classes))

    # -----------------------
    # Save training plots
    # -----------------------
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(val_acc_history, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs (ConvNeXt-Tiny)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/accuracy_curve.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(train_loss_history, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss over Epochs (ConvNeXt-Tiny)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/loss_curve.png")
    plt.close()

    print("✅ Plots saved to ./plots/, model saved to", MODEL_SAVE_PATH)
