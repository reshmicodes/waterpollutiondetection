import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from model import OilModel, DebrisModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# TRANSFORMS
# -------------------------
transform = transforms.Compose([
    transforms.Resize((256,256)),        # 👈 bigger image
    transforms.CenterCrop(224),          # 👈 crop to model size
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------
# OIL DATASET
# -------------------------
oil_train = datasets.ImageFolder("data/oil/train", transform=transform)
oil_loader = DataLoader(oil_train, batch_size=8, shuffle=True)

val_data = datasets.ImageFolder("data/oil/validation", transform=transform)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

# -------------------------
# MODEL
# -------------------------
oil_model = OilModel().to(device)

# -------------------------
# WEIGHTED LOSS (IMPORTANT)
# -------------------------
from collections import Counter

labels = oil_train.targets
count = Counter(labels)

total = sum(count.values())
class_weights = [total / count[i] for i in range(len(count))]

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

oil_criterion = nn.CrossEntropyLoss(weight=class_weights)

# -------------------------
# OPTIMIZER (LOW LR BETTER)
# -------------------------
oil_optimizer = torch.optim.Adam(oil_model.parameters(), lr=0.0003)

print("Training Oil Model...")

# -------------------------
# TRAINING LOOP
# -------------------------
for epoch in range(25):
    total_loss = 0

    # ---- TRAIN ----
    oil_model.train()
    for img, label in oil_loader:
        img, label = img.to(device), label.to(device)

        pred = oil_model(img)
        loss = oil_criterion(pred, label)

        oil_optimizer.zero_grad()
        loss.backward()
        oil_optimizer.step()

        total_loss += loss.item()

    # ---- VALIDATION ----
    oil_model.eval()
    correct = 0
    total_val = 0

    with torch.no_grad():
        for img, label in val_loader:
            img, label = img.to(device), label.to(device)

            pred = oil_model(img)
            _, predicted = torch.max(pred, 1)

            total_val += label.size(0)
            correct += (predicted == label).sum().item()

    accuracy = 100 * correct / total_val

    print(f"Epoch {epoch}: Loss = {total_loss:.2f}, Val Accuracy = {accuracy:.2f}%")

# -------------------------
# SAVE MODEL
# -------------------------
torch.save(oil_model.state_dict(), "oil_model.pth")
# -------------------------
# DEBRIS DATASET
# -------------------------
debris_train = datasets.ImageFolder("data/debris", transform=transform)
debris_loader = DataLoader(debris_train, batch_size=8, shuffle=True)

debris_model = DebrisModel(num_classes=6).to(device)
debris_criterion = nn.CrossEntropyLoss()
debris_optimizer = torch.optim.Adam(debris_model.parameters(), lr=0.001)

print("Training Debris Model...")

for epoch in range(25):
    total_loss = 0
    for img, label in debris_loader:
        img, label = img.to(device), label.to(device)

        pred = debris_model(img)
        loss = debris_criterion(pred, label)

        debris_optimizer.zero_grad()
        loss.backward()
        debris_optimizer.step()

        total_loss += loss.item()

    print(f"Debris Epoch {epoch}: Loss = {total_loss:.4f}")

torch.save(debris_model.state_dict(), "debris_model.pth")

print("✅ Training Complete!")