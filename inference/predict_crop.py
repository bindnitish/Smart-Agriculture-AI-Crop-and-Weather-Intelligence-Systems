from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from models.efficientnet_model import create_model


# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "crop_efficientnetv2s_best.pth"
)


# ==========================================
# CLASS NAMES
# ==========================================

CLASS_NAMES = [
    "jute",
    "maize",
    "rice",
    "sugarcane",
    "wheat"
]


# ==========================================
# DEVICE
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ==========================================
# TRANSFORM
# ==========================================

transform = transforms.Compose([

    transforms.Resize(
        (128, 128)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.48, 0.47, 0.42],
        std=[0.226, 0.225, 0.227]
    )
])


# ==========================================
# LOAD MODEL
# ==========================================

model = create_model(
    num_classes=5,
    pretrained=False
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


# ==========================================
# CROP PREDICTION FUNCTION
# ==========================================

def predict_crop(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")


    # ======================================
    # PREPROCESS
    # ======================================

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )


    # ======================================
    # PREDICTION
    # ======================================

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )


    # ======================================
    # RESULT
    # ======================================

    predicted_index = (
        predicted_class.item()
    )

    predicted_name = (
        CLASS_NAMES[predicted_index]
    )

    confidence_value = (
        confidence.item() * 100
    )


    return (
        predicted_name,
        confidence_value
    )


# ==========================================
# DIRECT TEST
# ==========================================

if __name__ == "__main__":

    IMAGE_PATH = (
        PROJECT_ROOT
        / "test_image.jpg"
    )

    predicted_name, confidence_value = predict_crop(
        IMAGE_PATH
    )


    print("\n" + "=" * 60)
    print("CROP IMAGE PREDICTION")
    print("=" * 60)

    print(
        f"Image      : {IMAGE_PATH.name}"
    )

    print(
        f"Prediction : {predicted_name}"
    )

    print(
        f"Confidence : {confidence_value:.2f}%"
    )

    print("=" * 60)