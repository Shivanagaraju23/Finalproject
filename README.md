# 🌾 Crop Recommendation System
**University of Hyderabad · MBA Project 2024–2026**  
Student: Shivanagaraju Karnakanti (24MBMA70)  
Supervisor: Dr. C. Ganesh Kumar

---

## 📁 Project Structure

```
crop_recommendation_app/
├── app.py                      ← Main Streamlit application
├── requirements.txt            ← Python dependencies
├── Crop_recommendation.csv     ← Dataset (place here!)
├── README.md
└── models/                     ← Auto-created on first run
    ├── rf_model.pkl
    ├── scaler.pkl
    └── label_encoder.pkl
```

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.9 or higher
- VS Code (recommended) or any terminal

### 2. Setup

```bash
# Clone / navigate to project folder
cd crop_recommendation_app

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Add the Dataset
Place `Crop_recommendation.csv` inside the `crop_recommendation_app/` folder  
(same folder as `app.py`).

### 4. Run the App

```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**

---

## 📄 Pages

| Page | Description |
|---|---|
| 🏠 Home & Predict | Enter soil/climate parameters and get a crop recommendation |
| 📊 Data Analysis | Univariate, bivariate, and correlation EDA |
| 🤖 Model Comparison | Accuracy comparison and confusion matrices for all 5 models |
| 📈 Feature Importance | Random Forest feature importance rankings |
| ℹ️ About | Project details, architecture, and key findings |

---

## 🤖 Models Trained

| Model | Accuracy |
|---|---|
| Logistic Regression | ~92% |
| Decision Tree | ~98% |
| **Random Forest** | **~99%** ✅ Best |
| K-Nearest Neighbors | ~97% |
| XGBoost (Gradient Boosting) | ~98% |

---

## 🔧 Input Features

| Feature | Description | Unit |
|---|---|---|
| N | Nitrogen content | kg/ha |
| P | Phosphorus content | kg/ha |
| K | Potassium content | kg/ha |
| Temperature | Air temperature | °C |
| Humidity | Relative humidity | % |
| pH | Soil acidity | 0–14 |
| Rainfall | Annual rainfall | mm |

---

## 📦 Dependencies

```
streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn, joblib
```

---

## 📌 Notes
- Models are trained automatically on first run and cached.
- The `models/` folder is created automatically.
- No internet connection needed after setup (offline mode).
