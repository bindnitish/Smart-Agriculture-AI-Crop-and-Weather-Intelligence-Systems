import torch

from models.efficientnet_model import create_model
from train.loss import criterion
from train.optimizer import get_optimizer


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==========================================
# CREATE CROP MODEL
# ==========================================

model = create_model(
    num_classes=5,
    pretrained=True
)

model = model.to(device)


# ==========================================
# LOSS
# ==========================================

print("\nLoss Function:")
print(criterion)


# ==========================================
# OPTIMIZER
# ==========================================

optimizer = get_optimizer(model)

print("\nOptimizer:")
print(optimizer)


# ==========================================
# CHECK SETTINGS
# ==========================================

print("\nLearning Rate:")

for param_group in optimizer.param_groups:

    print(param_group["lr"])


print("\nWeight Decay:")

for param_group in optimizer.param_groups:

    print(param_group["weight_decay"])