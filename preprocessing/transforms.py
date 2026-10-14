from torchvision import transforms


# ==========================================
# PAPER SETTINGS
# ==========================================

IMAGE_SIZE = 128

MEAN = [0.48, 0.47, 0.42]

STD = [0.226, 0.225, 0.227]


# ==========================================
# TRAIN TRANSFORM
# ==========================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])


# ==========================================
# VALIDATION TRANSFORM
# ==========================================

val_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])


# ==========================================
# TEST TRANSFORM
# ==========================================

test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])