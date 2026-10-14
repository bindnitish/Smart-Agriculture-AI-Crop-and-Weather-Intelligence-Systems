import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights
)


# ==========================================
# EFFICIENTNETV2-S MODEL
# ==========================================

def create_model(num_classes, pretrained=True):

    # --------------------------------------
    # Load EfficientNetV2-S
    # --------------------------------------

    if pretrained:

        weights = EfficientNet_V2_S_Weights.DEFAULT

    else:

        weights = None


    model = efficientnet_v2_s(
        weights=weights
    )


    # --------------------------------------
    # Replace Final Classifier
    # --------------------------------------

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )


    return model