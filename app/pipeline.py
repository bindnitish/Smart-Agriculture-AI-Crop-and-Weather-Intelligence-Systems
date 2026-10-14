from pathlib import Path

from inference.predict_crop import predict_crop
from inference.predict_weather import predict_weather
from agentic_ai.decision import make_decision


# ==========================================
# FINAL SMART AGRICULTURE PIPELINE
# ==========================================

def run_smart_agriculture(
    crop_image,
    weather_image
):

    print("\n" + "=" * 60)
    print("SMART AGRICULTURE - FINAL PIPELINE")
    print("=" * 60)


    # ======================================
    # IMAGE PATHS
    # ======================================

    crop_image = Path(crop_image)
    weather_image = Path(weather_image)


    print(
        f"\nCrop Image    : {crop_image.name}"
    )

    print(
        f"Weather Image : {weather_image.name}"
    )


    # ======================================
    # 1. CROP PREDICTION
    # ======================================

    print("\n[1] Crop prediction")

    crop, crop_confidence = predict_crop(
        crop_image
    )

    print(
        f"Crop Prediction     : {crop}"
    )

    print(
        f"Crop Confidence     : "
        f"{crop_confidence:.2f}%"
    )


    # ======================================
    # 2. WEATHER PREDICTION
    # ======================================

    print("\n[2] Weather prediction")

    weather, weather_confidence = predict_weather(
        weather_image
    )

    print(
        f"Weather Prediction  : {weather}"
    )

    print(
        f"Weather Confidence  : "
        f"{weather_confidence:.2f}%"
    )


    # ======================================
    # 3. AGENTIC AI DECISION
    # ======================================

    print("\n[3] Agentic AI decision")

    decision = make_decision(
        crop,
        weather
    )


    # ======================================
    # 4. FINAL RESULT
    # ======================================

    print("\n[4] IoT action execution")

    print(
        "\nFinal Decision Generated Successfully."
    )


    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)


    return {
        "crop": crop,
        "crop_confidence": crop_confidence,

        "weather": weather,
        "weather_confidence": weather_confidence,

        "decision": decision
    }