import torch

from models.efficientnet_model import create_model


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==========================================
# CROP MODEL
# ==========================================

crop_model = create_model(
    num_classes=5,
    pretrained=True
)

crop_model = crop_model.to(device)

print("\nCrop Model")
print("----------")
print(crop_model.classifier)


# ==========================================
# WEATHER MODEL
# ==========================================

weather_model = create_model(
    num_classes=11,
    pretrained=True
)

weather_model = weather_model.to(device)

print("\nWeather Model")
print("-------------")
print(weather_model.classifier)


# ==========================================
# TEST FORWARD PASS
# ==========================================

dummy_input = torch.randn(
    2,
    3,
    128,
    128
).to(device)


# Crop
crop_output = crop_model(dummy_input)

print("\nCrop output shape   :", crop_output.shape)


# Weather
weather_output = weather_model(dummy_input)

print("Weather output shape:", weather_output.shape)