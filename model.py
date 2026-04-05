import torch
import torch.nn as nn
import torchvision.models as models

# -------------------------
# Oil Model
# -------------------------
class OilModel(nn.Module):
    def __init__(self):
        super(OilModel, self).__init__()

        self.model = models.resnet18(pretrained=True)

        # 🔥 FREEZE ALL LAYERS
        for param in self.model.parameters():
            param.requires_grad = False

        # 🔥 REPLACE FINAL LAYER
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)


# -------------------------
# Debris Model
# -------------------------
class DebrisModel(nn.Module):
    def __init__(self, num_classes=6):
        super(DebrisModel, self).__init__()

        self.model = models.resnet18(pretrained=True)

        # 🔥 FREEZE ALL LAYERS
        for param in self.model.parameters():
            param.requires_grad = False

        # 🔥 REPLACE FINAL LAYER
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)