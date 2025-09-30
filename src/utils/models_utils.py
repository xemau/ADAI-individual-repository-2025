import torch
from torch import nn
import torchvision.models as models

def build_resnet18_binary(num_classes=2, use_pretrained=True):
    try:
        if use_pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        else:
            model = models.resnet18(weights=None)
    except Exception:
        model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def load_checkpoint(model, checkpoint_path, device):
    data = torch.load(checkpoint_path, map_location=device)
    state = data.get("state_dict", data)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model

def build_resnet18_multiclass(num_classes, use_pretrained=True):
    try:
        if use_pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        else:
            model = models.resnet18(weights=None)
    except Exception:
        model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model