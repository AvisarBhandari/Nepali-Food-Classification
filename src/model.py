import torchvision.models as models
import torch.nn as nn
import torch


def make_model(class_names=None, class_len=None):
    if class_names:
        class_len = len(class_names)
    model = models.resnet18(weights="IMAGENET1K_V1", progress=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=model.fc.in_features, out_features=class_len),
    )
    return model
