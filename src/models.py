import torch.nn as nn
from torchvision import models

def build_model(pretrained=True):
    weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model=models.mobilenet_v3_large(weights=weights)
    model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,2)
    return model

def freeze_backbone(model):
    for name,param in model.named_parameters():
        param.requires_grad=name.startswith('classifier.')
