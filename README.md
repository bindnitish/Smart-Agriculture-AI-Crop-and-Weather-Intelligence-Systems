# 🌾 Smart Agriculture AI – Crop and Weather Intelligence System

## 📌 Project Overview

Smart Agriculture AI is an image-based agricultural decision-support system that classifies crop and weather images using deep learning models. It combines both predictions through a rule-based decision engine to generate agricultural recommendations.

The system integrates image classification, decision support, software-level IoT action simulation, and a Streamlit dashboard.

## 🎯 Key Features

- Crop classification using EfficientNetV2-S.
- Weather classification using EfficientNetV2-S.
- Rule-based agricultural recommendations using crop and weather predictions.
- Software-level IoT action mapping and simulation.
- Interactive Streamlit dashboard.
- Modular architecture for future improvements.

## 🧠 Model Details

**Model Architecture:** EfficientNetV2-S

**Crop Classes (5):**
- Jute
- Maize
- Rice
- Sugarcane
- Wheat

**Weather Classes (11):**
- Dew
- Fog/Smog
- Frost
- Glaze
- Hail
- Lightning
- Rain
- Rainbow
- Rime
- Sandstorm
- Snow

## ⚙️ System Workflow

1. Upload a crop image and a weather image.
2. Preprocess both images by resizing them to 128 × 128 pixels and normalizing them.
3. Use separate EfficientNetV2-S models to classify crop and weather images.
4. Pass the predicted labels to the rule-based decision engine.
5. Generate an agricultural recommendation.
6. Map the recommendation to a simulated IoT action.
7. Display the results in the Streamlit dashboard.

## 🛠️ Technology Stack

- Python
- PyTorch
- Torchvision
- EfficientNetV2-S
- NumPy
- Pillow
- Streamlit
- Git and GitHub

## 🚀 Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/bindnitish/smart-agriculture-ai.git
cd smart-agriculture-ai
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure the trained models**

Place the trained crop and weather model checkpoints in the required local directory and ensure that the paths in the inference code are correct.

**5. Run the application**

```bash
streamlit run app.py
```

Ensure that `requirements.txt`, `app.py`, and the model paths match the actual project files.

## 📊 Results and Evaluation

During development, the project reported approximately:

- Crop classification validation accuracy: 99.67%
- Weather classification validation accuracy: 92.05%

These are development-stage validation results, not final independent test-set results.

## ⚠️ Current Limitations

- Performance on unseen real-world images requires further validation.
- The system supports only the specified crop and weather classes.
- Recommendations depend on predefined rules.
- IoT actions are simulated in software; physical sensors and actuators are not connected.

## 🔮 Future Scope

- Integrate soil-moisture and humidity sensors.
- Incorporate weather forecast data.
- Evaluate performance on an independent test dataset.
- Improve recommendations using expert-validated agricultural rules.
- Integrate physical IoT devices after appropriate testing.

## 👨‍💻 Author

**Nitish Kumar Bind**

M.Tech in Computer Science and Engineering  
Dr. SPM IIIT Naya Raipur

---

*This project is an implementation-oriented agricultural decision-support prototype inspired by an edge-enabled smart agriculture framework.*
