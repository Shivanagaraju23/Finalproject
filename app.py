import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2d6a4f, #52b788);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 { font-size: 2.2rem; margin: 0; }
    .main-header p  { font-size: 1rem; opacity: 0.9; margin-top: 0.5rem; }

    .metric-card {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card h2 { color: #166534; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #15803d; margin: 0; font-size: 0.9rem; }

    .result-box {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border: 2px solid #4ade80;
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-box h2 { color: #14532d; font-size: 2.5rem; }
    .result-box p  { color: #166534; font-size: 1.1rem; }

    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }

    .stButton>button {
        background: linear-gradient(135deg, #2d6a4f, #52b788);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
DATA_PATH = "Crop_recommendation.csv"
MODEL_DIR = "models"

CROP_EMOJI = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘",
    "pigeonpeas": "🫘", "mothbeans": "🫘", "mungbean": "🫘", "blackgram": "🫘",
    "lentil": "🫘", "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍎",
    "orange": "🍊", "papaya": "🍈", "coconut": "🥥", "cotton": "🌿",
    "jute": "🌿", "coffee": "☕",
}

CROP_INFO = {
    "rice": {"desc": "Thrives in warm, humid climates with high rainfall and nitrogen-rich soil.",
             "irrigation": "Requires flooded or highly irrigated fields. Water depth of 5–10 cm."},
    "maize": {"desc": "Grows well in moderate temperature with balanced NPK and medium rainfall.",
              "irrigation": "Needs irrigation every 8–10 days; critical at tasseling and silking stages."},
    "chickpea": {"desc": "Prefers cool, dry conditions with well-drained, low-phosphorus soil.",
                 "irrigation": "Drought-tolerant; 1–2 irrigations at flowering and pod-fill stages."},
    "kidneybeans": {"desc": "Needs warm temperatures and well-drained, moderately fertile soil.",
                    "irrigation": "Moderate irrigation; avoid waterlogging."},
    "pigeonpeas": {"desc": "Drought-tolerant legume suited for semi-arid tropics.",
                   "irrigation": "Rain-fed mostly; supplemental irrigation at flowering if dry."},
    "mothbeans": {"desc": "Highly drought-resistant; grows in arid sandy soils.",
                  "irrigation": "Very low water requirement; 1 irrigation often sufficient."},
    "mungbean": {"desc": "Short-duration legume suited to warm, humid conditions.",
                 "irrigation": "Light irrigation every 7–10 days during dry spells."},
    "blackgram": {"desc": "Prefers warm and humid climate with well-drained loamy soil.",
                  "irrigation": "2–3 irrigations; critical at flowering and grain-fill."},
    "lentil": {"desc": "Cool-season legume with low nitrogen requirement.",
               "irrigation": "1–2 light irrigations; sensitive to excess water."},
    "pomegranate": {"desc": "Drought-tolerant fruit; thrives in hot, arid climates.",
                    "irrigation": "Drip irrigation preferred; 30–35 liters/plant/day."},
    "banana": {"desc": "Tropical fruit needing warm temperatures and high humidity.",
               "irrigation": "High water need; drip or flood irrigation every 3–5 days."},
    "mango": {"desc": "Tropical fruit; prefers dry weather during flowering.",
              "irrigation": "Irrigation needed during dry spells; withhold before flowering."},
    "grapes": {"desc": "Grown in warm, dry climates with well-drained soils.",
               "irrigation": "Drip irrigation; critical at berry set and veraison."},
    "watermelon": {"desc": "Needs sandy loam, warm temperature and ample sunshine.",
                   "irrigation": "Regular irrigation; drip preferred to avoid fruit rot."},
    "muskmelon": {"desc": "Warm-season crop suited to light, well-drained soil.",
                  "irrigation": "Moderate; reduce near maturity to improve sweetness."},
    "apple": {"desc": "Temperate fruit requiring cold winters and moderate summers.",
              "irrigation": "Sprinkler or drip; 1200–1500 mm per season."},
    "orange": {"desc": "Sub-tropical citrus; needs well-drained soil and moderate water.",
               "irrigation": "Drip irrigation; avoid water stress at fruit set."},
    "papaya": {"desc": "Tropical plant needing warm, humid conditions.",
               "irrigation": "Frequent light irrigation; very sensitive to waterlogging."},
    "coconut": {"desc": "Tropical palm needing high humidity and rainfall.",
                "irrigation": "Basin irrigation; 45–50 liters/palm/day in dry season."},
    "cotton": {"desc": "Warm-season fibre crop; needs deep, well-drained soil.",
               "irrigation": "Critical at squaring, flowering, and boll development."},
    "jute": {"desc": "Grows best in warm, humid climates with loamy alluvial soil.",
             "irrigation": "High rainfall crop; supplemental irrigation in dry spells."},
    "coffee": {"desc": "Shade-grown crop; prefers humid, mild-temperature highlands.",
               "irrigation": "Drip or sprinkler; 1200–2000 mm distributed evenly."},
}

# ─────────────────────────────────────────────
# Data & Model Loading
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource
def train_models(df):
    os.makedirs(MODEL_DIR, exist_ok=True)

    le = LabelEncoder()
    df["crop_num"] = le.fit_transform(df["label"])

    X = df.drop(["label", "crop_num"], axis=1)
    y = df["crop_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
        "XGBoost (GBM)":       GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    trained = {}
    for name, mdl in models.items():
        mdl.fit(X_train_sc, y_train)
        preds = mdl.predict(X_test_sc)
        acc   = accuracy_score(y_test, preds)
        cm    = confusion_matrix(y_test, preds)
        results[name]  = {"accuracy": acc, "cm": cm, "preds": preds, "y_test": y_test}
        trained[name]  = mdl

    # Save
    joblib.dump(trained["Random Forest"], f"{MODEL_DIR}/rf_model.pkl")
    joblib.dump(scaler,                   f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(le,                       f"{MODEL_DIR}/label_encoder.pkl")

    return trained, scaler, le, results, X_train_sc, X_test_sc, y_train, y_test, X.columns.tolist()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.markdown("## 🌿 Navigation")
page = st.sidebar.radio("", [
    "🏠 Home & Predict",
    "📊 Data Analysis",
    "🤖 Model Comparison",
    "📈 Feature Importance",
    "ℹ️ About",
])
st.sidebar.markdown("---")
st.sidebar.markdown("**University of Hyderabad**")
st.sidebar.markdown("MBA Project – 24MBMA70")
st.sidebar.markdown("Supervised by Dr. C. Ganesh Kumar")

# ─────────────────────────────────────────────
# Load data & train once
# ─────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    st.error(f"❌ Dataset not found. Place `Crop_recommendation.csv` in the same folder as `app.py`.")
    st.stop()

df = load_data()

with st.spinner("Training models… (first run only)"):
    trained_models, scaler, le, results, X_train_sc, X_test_sc, y_train, y_test, feature_cols = train_models(df)

crops_list = sorted(df["label"].unique().tolist())

# ═══════════════════════════════════════════════
# PAGE: Home & Predict
# ═══════════════════════════════════════════════
if page == "🏠 Home & Predict":
    st.markdown("""
    <div class="main-header">
        <h1>🌾 Crop Recommendation System</h1>
        <p>AI-powered precision agriculture using Machine Learning & Generative AI</p>
        <p>University of Hyderabad · MBA Project 2024–2026</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><h2>22</h2><p>Crop Types</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><h2>2200</h2><p>Data Records</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><h2>5</h2><p>ML Models</p></div>', unsafe_allow_html=True)
    with c4:
        best_acc = max(v["accuracy"] for v in results.values())
        st.markdown(f'<div class="metric-card"><h2>{best_acc:.1%}</h2><p>Best Accuracy</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔍 Enter Soil & Climate Parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🌱 Soil Nutrients**")
        N   = st.slider("Nitrogen (N)",        0,  140, 60, help="Nitrogen content in soil (kg/ha)")
        P   = st.slider("Phosphorus (P)",      5,  145, 40, help="Phosphorus content in soil (kg/ha)")
        K   = st.slider("Potassium (K)",       5,  205, 40, help="Potassium content in soil (kg/ha)")
        ph  = st.slider("Soil pH",             3.5, 10.0, 6.5, step=0.1, help="Soil acidity/alkalinity")

    with col2:
        st.markdown("**🌤️ Climate Conditions**")
        temp     = st.slider("Temperature (°C)",   8.0,  44.0, 25.0, step=0.5)
        humidity = st.slider("Humidity (%)",        14.0, 100.0, 70.0, step=0.5)
        rainfall = st.slider("Rainfall (mm)",       20.0, 300.0, 120.0, step=1.0)

    model_choice = st.selectbox("🤖 Select Model", list(trained_models.keys()), index=2)

    if st.button("🌱 Recommend Crop"):
        input_arr  = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        input_sc   = scaler.transform(input_arr)
        mdl        = trained_models[model_choice]
        pred_num   = mdl.predict(input_sc)[0]
        pred_crop  = le.inverse_transform([pred_num])[0]
        emoji      = CROP_EMOJI.get(pred_crop, "🌿")
        info       = CROP_INFO.get(pred_crop, {
            "desc": "A suitable crop for the given conditions.",
            "irrigation": "Follow standard irrigation practices for this crop."
        })
        acc        = results[model_choice]["accuracy"]

        st.markdown(f"""
        <div class="result-box">
            <h2>{emoji} {pred_crop.title()}</h2>
            <p>Recommended by <strong>{model_choice}</strong> · Model accuracy: <strong>{acc:.1%}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="info-box">
                <strong>🌾 Optimal Farming Practices</strong><br>{info['desc']}
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="info-box">
                <strong>💧 Irrigation Needs</strong><br>{info['irrigation']}
            </div>
            """, unsafe_allow_html=True)

        # Input summary
        st.markdown("**📋 Your Input Parameters**")
        input_df = pd.DataFrame({
            "Parameter": ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "pH", "Rainfall"],
            "Value":     [N, P, K, f"{temp} °C", f"{humidity} %", ph, f"{rainfall} mm"]
        })
        st.dataframe(input_df, use_container_width=True, hide_index=True)

        # Probability chart if model supports it
        if hasattr(mdl, "predict_proba"):
            proba  = mdl.predict_proba(input_sc)[0]
            top5_i = np.argsort(proba)[::-1][:5]
            top5_crops = le.inverse_transform(top5_i)
            top5_proba = proba[top5_i]

            fig, ax = plt.subplots(figsize=(7, 3))
            bars = ax.barh(top5_crops[::-1], top5_proba[::-1], color="#52b788")
            ax.set_xlabel("Probability")
            ax.set_title("Top 5 Crop Probabilities")
            for bar, val in zip(bars, top5_proba[::-1]):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                        f"{val:.1%}", va="center", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ═══════════════════════════════════════════════
# PAGE: Data Analysis
# ═══════════════════════════════════════════════
elif page == "📊 Data Analysis":
    st.header("📊 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["Univariate", "Bivariate", "Correlation"])

    with tab1:
        st.subheader("Distribution of Features")
        features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        axes = axes.flatten()
        colors = ["#52b788", "#40916c", "#2d6a4f", "#74c69d", "#95d5b2", "#b7e4c7", "#d8f3dc"]
        for i, feat in enumerate(features):
            axes[i].hist(df[feat], bins=30, color=colors[i], edgecolor="white", alpha=0.85)
            axes[i].set_title(feat.capitalize(), fontweight="bold")
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Count")
        axes[-1].axis("off")
        plt.suptitle("Univariate Distributions", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("Boxplots (Outlier Detection)")
        fig2, axes2 = plt.subplots(2, 4, figsize=(16, 7))
        axes2 = axes2.flatten()
        for i, feat in enumerate(features):
            axes2[i].boxplot(df[feat], patch_artist=True,
                             boxprops=dict(facecolor=colors[i], alpha=0.7))
            axes2[i].set_title(feat.capitalize(), fontweight="bold")
        axes2[-1].axis("off")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with tab2:
        st.subheader("Feature vs Crop Label")
        feat_sel = st.selectbox("Select feature", ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])

        fig, ax = plt.subplots(figsize=(14, 5))
        crop_mean = df.groupby("label")[feat_sel].mean().sort_values()
        bars = ax.barh(crop_mean.index, crop_mean.values, color="#52b788", edgecolor="white")
        ax.set_xlabel(f"Mean {feat_sel}")
        ax.set_title(f"Average {feat_sel} per Crop", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("Crop Distribution (Records per Crop)")
        counts = df["label"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.bar(counts.index, counts.values, color="#40916c", edgecolor="white")
        ax2.set_ylabel("Count")
        ax2.set_xticklabels(counts.index, rotation=45, ha="right")
        ax2.set_title("Balanced Dataset – 100 Samples per Crop", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with tab3:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = df.drop("label", axis=1).corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                    mask=mask, ax=ax, linewidths=0.5, square=True)
        ax.set_title("Feature Correlation Matrix", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        <div class="info-box">
        <strong>Key Insights:</strong> P and K show a strong positive correlation (0.74), 
        suggesting that soils rich in phosphorus tend to also be high in potassium. 
        Nitrogen is relatively independent of other nutrients, making it a powerful standalone predictor.
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# PAGE: Model Comparison
# ═══════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.header("🤖 Model Performance Comparison")

    # Accuracy table
    acc_data = {name: f"{v['accuracy']:.4f}" for name, v in results.items()}
    acc_df = pd.DataFrame(list(acc_data.items()), columns=["Model", "Accuracy"]).sort_values("Accuracy", ascending=False)
    acc_df["Accuracy %"] = acc_df["Accuracy"].astype(float).apply(lambda x: f"{x:.1%}")
    st.dataframe(acc_df[["Model", "Accuracy %"]], use_container_width=True, hide_index=True)

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    names = acc_df["Model"].tolist()
    accs  = acc_df["Accuracy"].astype(float).tolist()
    colors_bar = ["#2d6a4f" if a == max(accs) else "#95d5b2" for a in accs]
    bars  = ax.bar(names, [a * 100 for a in accs], color=colors_bar, edgecolor="white")
    ax.set_ylim(85, 101)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy Comparison", fontweight="bold")
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.1%}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticklabels(names, rotation=15, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm_model = st.selectbox("Select model", list(results.keys()))
    cm       = results[cm_model]["cm"]
    classes  = le.classes_

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=classes, yticklabels=classes, ax=ax,
                linewidths=0.3, cbar=True)
    ax.set_title(f"Confusion Matrix – {cm_model}", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Classification report
    with st.expander("📋 Classification Report"):
        report = classification_report(
            results[cm_model]["y_test"],
            results[cm_model]["preds"],
            target_names=classes
        )
        st.code(report)

# ═══════════════════════════════════════════════
# PAGE: Feature Importance
# ═══════════════════════════════════════════════
elif page == "📈 Feature Importance":
    st.header("📈 Feature Importance Analysis")

    rf = trained_models["Random Forest"]
    importances = rf.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors_fi = ["#2d6a4f" if i == 0 else "#52b788" if i <= 2 else "#95d5b2"
                 for i in range(len(feat_imp_df))]
    bars = ax.barh(feat_imp_df["Feature"][::-1], feat_imp_df["Importance"][::-1],
                   color=colors_fi[::-1], edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title("Random Forest – Feature Importance", fontweight="bold")
    for bar, val in zip(bars, feat_imp_df["Importance"][::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("📌 Key Insights")
    for _, row in feat_imp_df.iterrows():
        bar_width = int(row["Importance"] * 400)
        st.markdown(f"""
        **{row['Feature'].capitalize()}** — Importance: `{row['Importance']:.4f}`  
        <div style="background:#d1fae5;height:10px;width:{bar_width}px;border-radius:5px;margin-bottom:10px;"></div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>Summary:</strong> Rainfall and humidity are the top environmental predictors, 
    while nitrogen is the most important soil nutrient. Temperature plays a significant role 
    in differentiating tropical vs. temperate crops. pH has the lowest individual importance 
    but still contributes to model accuracy when combined with other features.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# PAGE: About
# ═══════════════════════════════════════════════
elif page == "ℹ️ About":
    st.header("ℹ️ About This Project")

    st.markdown("""
    <div class="main-header">
        <h1>Application of Machine Learning Models for Crop Recommendation System</h1>
        <p>Final Project Report · Master of Business Administration · University of Hyderabad</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **👤 Student**  
        Shivanagaraju Karnakanti  
        Roll No: 24MBMA70  
        MBA 2024–2026

        **👨‍🏫 Supervisor**  
        Dr. C. Ganesh Kumar  
        Associate Professor  
        School of Management Studies  
        University of Hyderabad
        """)
    with col2:
        st.markdown("""
        **📦 Dataset**  
        Kaggle Crop Recommendation Dataset  
        2200 records · 22 crop types · 7 features

        **🔧 Tech Stack**  
        Python · Scikit-learn · Streamlit  
        Pandas · Matplotlib · Seaborn

        **🤖 Models**  
        Logistic Regression · Decision Tree  
        Random Forest · KNN · XGBoost (GBM)
        """)

    st.markdown("---")
    st.subheader("📐 System Architecture")
    st.markdown("""
    ```
    Input Data (N, P, K, Temperature, Humidity, pH, Rainfall)
             ↓
       Data Preprocessing (Label Encoding + MinMaxScaler)
             ↓
    Machine Learning Models (RF / XGBoost / DT / KNN / LR)
             ↓
            Crop Prediction
             ↓
    Generative AI Layer (Farming Practices + Irrigation Advice)
             ↓
       Final Recommendation → Streamlit Dashboard
    ```
    """)

    st.subheader("📊 Model Accuracies (from report)")
    report_acc = {
        "Logistic Regression": 0.920,
        "Decision Tree":       0.984,
        "Random Forest":       0.990,
        "KNN":                 0.970,
        "XGBoost (GBM)":       0.982,
    }
    for m, a in report_acc.items():
        st.markdown(f"- **{m}**: {a:.1%}")

    st.subheader("🔑 Key Findings")
    st.markdown("""
    - Soil nutrients (N, P, K) and climatic conditions (rainfall, temperature) are the most influential factors in crop selection.
    - Ensemble methods (Random Forest, XGBoost) significantly outperform simpler models (Logistic Regression).
    - Random Forest achieved the highest accuracy (~99%) and was selected for deployment.
    - A Generative AI layer (LLM-based explanations) greatly improves system interpretability and farmer trust.
    - pH values between 5.5–7.5 (slightly acidic to neutral) support the majority of crops in the dataset.
    """)

    st.subheader("🚀 Future Scope")
    st.markdown("""
    - Integration of real-time IoT sensor data
    - SHAP / LIME for advanced model interpretability
    - Mobile-first PWA deployment
    - Multilingual support for regional Indian farmers
    - Satellite imagery and remote sensing integration
    """)
