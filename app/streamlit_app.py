import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import streamlit as st
import tempfile

from inference.predict_crop import predict_crop
from inference.predict_weather import predict_weather
from agentic_ai.decision import make_decision




# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Smart Agriculture AI",
    page_icon="🌱",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("🌱 Smart Agriculture AI System")

st.write(
    "Crop and weather image classification "
    "with Agentic AI based decision making."
)


st.divider()


# ==========================================
# IMAGE UPLOAD
# ==========================================

col1, col2 = st.columns(2)


with col1:

    st.subheader("🌾 Crop Image")

    crop_file = st.file_uploader(
        "Upload crop image",
        type=["jpg", "jpeg", "png"],
        key="crop"
    )


with col2:

    st.subheader("🌦️ Weather Image")

    weather_file = st.file_uploader(
        "Upload weather image",
        type=["jpg", "jpeg", "png"],
        key="weather"
    )


st.divider()


# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button(
    "🔍 Predict",
    use_container_width=True
):

    if crop_file is None or weather_file is None:

        st.warning(
            "Please upload both crop and weather images."
        )

    else:

        # ==================================
        # SAVE TEMPORARY CROP IMAGE
        # ==================================

        crop_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(crop_file.name).suffix
        )

        crop_temp.write(
            crop_file.getbuffer()
        )

        crop_temp.close()


        # ==================================
        # SAVE TEMPORARY WEATHER IMAGE
        # ==================================

        weather_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(weather_file.name).suffix
        )

        weather_temp.write(
            weather_file.getbuffer()
        )

        weather_temp.close()


        # ==================================
        # CROP PREDICTION
        # ==================================

        with st.spinner(
            "Predicting crop..."
        ):

            crop, crop_confidence = predict_crop(
                crop_temp.name
            )


        # ==================================
        # WEATHER PREDICTION
        # ==================================

        with st.spinner(
            "Predicting weather..."
        ):

            weather, weather_confidence = predict_weather(
                weather_temp.name
            )


        # ==================================
        # DISPLAY PREDICTIONS
        # ==================================

        st.subheader("📊 Model Predictions")


        result_col1, result_col2 = st.columns(2)


        with result_col1:

            st.metric(
                "Predicted Crop",
                crop
            )

            st.metric(
                "Crop Confidence",
                f"{crop_confidence:.2f}%"
            )


        with result_col2:

            st.metric(
                "Predicted Weather",
                weather
            )

            st.metric(
                "Weather Confidence",
                f"{weather_confidence:.2f}%"
            )


        st.divider()


        # ==================================
        # AGENTIC AI
        # ==================================

        st.subheader(
            "🤖 Agentic AI Decision"
        )


        decision = make_decision(
            crop,
            weather
        )


        # ==================================
        # DISPLAY DECISION
        # ==================================

        if decision is not None:

            # ------------------------------
            # COMBINED RULE
            # ------------------------------

            if isinstance(decision, dict):

                st.success(
                    "Combined Rule Selected"
                )


                st.write(
                    "**IoT Devices:**"
                )

                for device in decision.get(
                    "devices",
                    []
                ):

                    st.write(
                        f"🔹 {device}"
                    )


                st.write(
                    "**Recommended Actions:**"
                )

                for action in decision.get(
                    "actions",
                    []
                ):

                    st.write(
                        f"🔹 {action}"
                    )


            # ------------------------------
            # SEPARATE RULES
            # ------------------------------

            elif isinstance(decision, list):

                st.info(
                    "Separate / Fallback Rules Selected"
                )


                for result in decision:

                    st.write(
                        f"**Rule Source:** "
                        f"{result.get('source', 'N/A')}"
                    )

                    st.write(
                        f"**Condition:** "
                        f"{result.get('condition', 'N/A')}"
                    )


                    st.write(
                        "**IoT Devices:**"
                    )

                    for device in result.get(
                        "devices",
                        []
                    ):

                        st.write(
                            f"🔹 {device}"
                        )


                    st.write(
                        "**Recommended Actions:**"
                    )

                    for action in result.get(
                        "actions",
                        []
                    ):

                        st.write(
                            f"🔹 {action}"
                        )

                    st.divider()


        else:

            st.warning(
                "No applicable agricultural rule found."
            )