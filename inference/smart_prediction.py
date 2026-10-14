from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from models.efficientnet_model import create_model
from agentic_ai.decision import make_decision


# ==========================================
# PROJECT ROOT
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ==========================================
# MODEL PATHS
# ==========================================

CROP_MODEL_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "crop_efficientnetv2s_best.pth"
)

WEATHER_MODEL_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "weather_efficientnetv2s_best.pth"
)


# ==========================================
# CLASS NAMES
# ==========================================

CROP_CLASSES = [
    "jute",
    "maize",
    "rice",
    "sugarcane",
    "wheat"
]


WEATHER_CLASSES = [
    "dew",
    "fogsmog",
    "frost",
    "glaze",
    "hail",
    "lightning",
    "rain",
    "rainbow",
    "rime",
    "sandstorm",
    "snow"
]


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
# LOAD CROP MODEL
# ==========================================

crop_model = create_model(
    num_classes=5,
    pretrained=False
)

crop_checkpoint = torch.load(
    CROP_MODEL_PATH,
    map_location=DEVICE
)

crop_model.load_state_dict(
    crop_checkpoint["model_state_dict"]
)

crop_model = crop_model.to(DEVICE)

crop_model.eval()


# ==========================================
# LOAD WEATHER MODEL
# ==========================================

weather_model = create_model(
    num_classes=11,
    pretrained=False
)

weather_checkpoint = torch.load(
    WEATHER_MODEL_PATH,
    map_location=DEVICE
)

weather_model.load_state_dict(
    weather_checkpoint["model_state_dict"]
)

weather_model = weather_model.to(DEVICE)

weather_model.eval()


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict(model, image_path, class_names):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)


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


    predicted_index = (
        predicted_class.item()
    )

    predicted_name = (
        class_names[predicted_index]
    )

    confidence_value = (
        confidence.item() * 100
    )

    return (
        predicted_name,
        confidence_value
    )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("SMART AGRICULTURE - FULL PIPELINE")
    print("=" * 60)

    print(
        f"Device : {DEVICE}"
    )


    # ======================================
    # IMAGE PATHS
    # ======================================

    crop_image = (
        PROJECT_ROOT
        / "test_image.jpg"
    )

    weather_image = (
        PROJECT_ROOT
        / "test_weather.jpg"
    )


    # ======================================
    # CROP PREDICTION
    # ======================================

    crop_prediction, crop_confidence = predict(
        crop_model,
        crop_image,
        CROP_CLASSES
    )


    # ======================================
    # WEATHER PREDICTION
    # ======================================

    weather_prediction, weather_confidence = predict(
        weather_model,
        weather_image,
        WEATHER_CLASSES
    )


    # ======================================
    # DISPLAY PREDICTIONS
    # ======================================

    print("\n" + "-" * 60)

    print(
        f"Crop Image       : "
        f"{crop_image.name}"
    )

    print(
        f"Predicted Crop   : "
        f"{crop_prediction}"
    )

    print(
        f"Crop Confidence  : "
        f"{crop_confidence:.2f}%"
    )


    print("\n" + "-" * 60)

    print(
        f"Weather Image    : "
        f"{weather_image.name}"
    )

    print(
        f"Predicted Weather: "
        f"{weather_prediction}"
    )

    print(
        f"Weather Confidence: "
        f"{weather_confidence:.2f}%"
    )


    # ======================================
    # AGENTIC AI DECISION
    # ======================================

    result = make_decision(
        crop_prediction,
        weather_prediction
    )