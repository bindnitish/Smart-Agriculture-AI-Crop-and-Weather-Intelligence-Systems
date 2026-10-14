from pathlib import Path
from agentic_ai.decision import make_decision

import torch
from PIL import Image
from torchvision import transforms

from models.efficientnet_model import create_model


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

def predict_image(
    image_path,
    model_type
):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )


    # --------------------------------------
    # SELECT MODEL
    # --------------------------------------

    if model_type == "crop":

        model = crop_model
        class_names = CROP_CLASSES

    elif model_type == "weather":

        model = weather_model
        class_names = WEATHER_CLASSES

    else:

        raise ValueError(
            "model_type must be "
            "'crop' or 'weather'"
        )


    # --------------------------------------
    # PREDICTION
    # --------------------------------------

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = (
            torch.max(
                probabilities,
                dim=1
            )
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
    print("SMART AGRICULTURE - IMAGE PREDICTION")
    print("=" * 60)

    print(
        f"Device : {DEVICE}"
    )


    # --------------------------------------
    # IMAGE PATH
    # --------------------------------------

    image_path = (
        PROJECT_ROOT
        / "test_weather.jpg"
    )


    # --------------------------------------
    # SELECT MODEL
    # --------------------------------------

    print("\nSelect prediction type:")

    print(
        "1. Crop Classification"
    )

    print(
        "2. Weather Classification"
    )


    choice = input(
        "\nEnter choice (1/2): "
    )


    if choice == "1":

        model_type = "crop"

    elif choice == "2":

        model_type = "weather"

    else:

        print(
            "\nInvalid choice."
        )

        raise SystemExit


    # --------------------------------------
    # PREDICT
    # --------------------------------------

    prediction, confidence = (
        predict_image(
            image_path,
            model_type
        )
    )


    # --------------------------------------
    # RESULT
    # --------------------------------------

    print("\n" + "=" * 60)

    print(
        f"Image      : "
        f"{image_path.name}"
    )

    print(
        f"Model      : "
        f"{model_type}"
    )

    print(
        f"Prediction : "
        f"{prediction}"
    )

    print(
        f"Confidence : "
        f"{confidence:.2f}%"
    )

    print("=" * 60)