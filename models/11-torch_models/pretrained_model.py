"""
We will use the FCN ResNet50 from the PyTorch model. We will not
use any pretrained weights. Training from scratch.
"""

import torchvision.models as models
import torch.nn as nn

def fcn_resnet50_model(pretrained=True, requires_grad=True):
    model = models.segmentation.fcn_resnet50(weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT, progress=True)

    if requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True
    elif requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False

    # change the classification FCNHead and make it learnable
    model.classifier[4] = nn.Conv2d(512, 32, kernel_size=(1, 1))

    # change the aux_classification FCNHead and make it learnable
    model.aux_classifier[4] = nn.Conv2d(256, 32, kernel_size=(1, 1))
    
    return model


