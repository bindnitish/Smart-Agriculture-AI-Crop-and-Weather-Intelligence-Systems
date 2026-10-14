import torch


# ==========================================
# OPTIMIZER SETTINGS
# ==========================================

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5


# ==========================================
# GET OPTIMIZER
# ==========================================

def get_optimizer(model):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    return optimizer