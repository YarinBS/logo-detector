"""
Model module
"""


import os

import torch
import torch.nn as nn
import torchvision.models as models

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
torch.hub.set_dir(os.path.join(PROJECT_ROOT, "models", "downloaded"))

def get_model(load_pretrained: bool = True) -> nn.Module:
    """
    Loads a pre-trained ResNet-18 model and modifies the final FC layer to output 1 binary class.

    Parameters:
    - load_pretrained (bool): Whether to load pre-trained weights. Defaults to True.

    Returns:
    - nn.Module: The modified ResNet-18 model ready for training or inference.
    """

    weights = models.ResNet18_Weights.DEFAULT if load_pretrained else None
    model = models.resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 1),
    )

    return model   
