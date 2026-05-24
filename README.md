# 🔋 EV Battery Health Manager

A **Hybrid Deep Learning & Reinforcement Learning** system for Electric Vehicle battery health prediction and intelligent charging recommendations.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [How to Use](#how-to-use)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This application uses two AI models working together:

1. **LSTM Model**: Predicts battery **State of Health (SoH)** based on operational data
2. **Online DQN Agent**: Recommends optimal **charging actions** to extend battery life

---

## ✨ Features

- Real-time SoH prediction (State of Health)
- Intelligent charging recommendations
- Interactive dashboard with live metrics
- Low-latency inference (< 100ms)
- Visual Q-value analysis for agent decisions

---

## 🖥️ Prerequisites

### System Requirements

| Requirement | Minimum                     | Recommended    |
| ----------- | --------------------------- | -------------- |
| OS          | Windows 10 / macOS / Ubuntu | Latest version |
| RAM         | 4 GB                        | 8 GB+          |
| Storage     | 500 MB                      | 1 GB+          |
| Python      | 3.8                         | 3.10 or 3.11   |

### Required Software

- **Python 3.8 or higher** ([Download Python](https://www.python.org/downloads/))
- **pip** (comes with Python)

---

## 📦 Installation

### Step 1: Get the Project

**Option A - Download ZIP:**

1. Download the project as a ZIP file
2. Extract to your desired location

**Option B - Clone via Git:**

```bash
git clone https://github.com/yourusername/EV_Battery_App.git
cd EV_Battery_App
```

### Step 2: Install Dependencies

Open a terminal/command prompt and run:

```bash
# Navigate to project directory
cd EV_Battery_App

# Install all required packages
pip install -r requirements.txt
```

**This will install:**

- `streamlit` - Web application framework
- `tensorflow` - Deep learning framework
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning utilities
- `joblib` - Model serialization

---

## 🚀 Running the App

### Quick Start

1. **Open Terminal** in the project folder
2. **Run the application:**
   ```bash
   streamlit run app.py
   ```
3. **Access the app:**
   - Open your web browser
   - Go to: `http://localhost:8507`
   - Or click the URL shown in the terminal

---

## 📖 How to Use

### 1. Sidebar - Battery Telemetry

Adjust the following parameters:

| Parameter                    | Description                  | Range      |
| ---------------------------- | ---------------------------- | ---------- |
| **State of Charge (%)**      | Current battery charge level | 0 - 100%   |
| **Battery Voltage (V)**      | Current voltage              | 300 - 450V |
| **Charging Current (A)**     | Current flow during charging | 0 - 20A    |
| **Battery Temperature (°C)** | Current battery temp         | -10 - 60°C |
| **Cycle Count**              | Number of charge cycles      | 0 - 5000   |

### 2. Run Prediction

Click the **⚡ Run Prediction** button

### 3. View Results

The dashboard shows:

- **Predicted SoH**: Battery health percentage
  - 🟢 ≥ 90%: Healthy
  - 🟡 80-89%: Degrading
  - 🔴 < 80%: Critical

- **Recommended Action**:
  - ⬇️ Decrease: Reduce charging current
  - ⚖️ Maintain: Keep current optimal
  - ⬆️ Increase: Safe to charge faster

- **Inference Latency**: Response time in milliseconds

- **Q-Values Chart**: Agent decision confidence visualization

---

## 📂 Project Structure

```
EV_Battery_App/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── models/                   # Pre-trained models
│   ├── lstm_soh_model.keras  # LSTM SoH prediction model
│   ├── hybrid_online_dqn.keras # DQN charging agent
│   ├── scaler_X.pkl          # Feature scaler
│   └── scaler_y.pkl          # Target scaler
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── data_processing.py    # Data processing utilities
│   └── visualization.py      # Plotting functions
└── test_model_loading.py     # Model loading tests
```

---

## 🧠 Model Details

## 📦 Model Download

Due to GitHub file size limits, the trained model is not included.

👉 Download model here:
https://drive.google.com/models

After downloading, place it inside:
models/

### LSTM SoH Predictor

- **Input**: 20 timesteps × 6 features
- **Features**: IR, Tavg, Tmin, Tmax, ChargeTime, Cycle
- **Output**: State of Health (0.0 - 1.0)

### Online DQN Agent

- **Input State**: [SoH, Temperature, Cycle, Current]
- **Actions**: 3 (Decrease/Maintain/Increase current)
- **Training**: Online learning for adaptive control

---

## 🔧 Troubleshooting

### Issue: "command not found: streamlit"

**Solution:** Reinstall Streamlit

```bash
pip uninstall streamlit
pip install streamlit
```

### Issue: TensorFlow not found

**Solution:** Install TensorFlow

```bash
pip install tensorflow
```

### Issue: Port 8501 already in use

**Solution:** Streamlit will automatically use port 8507. Or specify a different port:

```bash
streamlit run app.py --server.port 8502
```

### Issue: Models fail to load

**Solution:**

1. Ensure all files are in the `models/` folder
2. Check file permissions
3. Verify Python version is 3.8+

### Issue: App runs slow

**Solution:**

1. Install watchdog for faster hot-reloading:
   ```bash
   pip install watchdog
   ```
2. Restart the app

---

## 📝 Sharing with Others

To share this project:

1. **Compress the entire folder** (EV_Battery_App/)
2. **Send to the other person**
3. **Provide these instructions**

They only need:

1. Python 3.8+ installed
2. Run `pip install -r requirements.txt`
3. Run `streamlit run app.py`

---

## 📄 License

This project is for educational and research purposes.

---

## 👨‍💻 Author

Created for Final Year Project / Research Demo

---

**Happy Battery Health Monitoring! 🔋⚡**
