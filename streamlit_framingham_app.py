import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, io, zipfile
import xgboost as xgb
import matplotlib.pyplot as plt

st.set_page_config(page_title="Framingham Stacked CHD Risk", layout="wide")
st.title("💓 10-Year CHD Risk Prediction — Stacked Ensemble Model")

ARTIFACT_DIR = "model_artifacts"
REQUIRED = ["imputer.pkl", "scaler.pkl", "meta_clf.pkl", "xgb_booster_full.json", "threshold.txt"]

# -----------------------------
# Load artifacts
# -----------------------------
def load_artifacts():
    artifacts = {}
    missing = []
    for fn in REQUIRED:
        path = os.path.join(ARTIFACT_DIR, fn)
        if os.path.exists(path):
            artifacts[fn] = path
        else:
            missing.append(fn)
    return artifacts, missing

artifacts, missing = load_artifacts()

st.sidebar.header("⚙️ Model Artifacts")
if missing:
    st.sidebar.warning("Missing artifacts: " + ", ".join(missing))
    uploaded = st.sidebar.file_uploader("Upload model_artifacts.zip or files", accept_multiple_files=True)

    if uploaded:
        for f in uploaded:
            name = f.name
            if name.endswith(".zip"):
                z = zipfile.ZipFile(io.BytesIO(f.read()))
                z.extractall(ARTIFACT_DIR)
            else:
                with open(os.path.join(ARTIFACT_DIR, name), "wb") as out:
                    out.write(f.read())
        artifacts, missing = load_artifacts()

if missing:
    st.error("❌ Required model artifacts missing. Upload them to proceed.")
    st.stop()
    
# -----------------------------
# SHAP VECTOR EXTRACTOR FUNCTION
# -----------------------------


def extract_single_shap_vector(shap_output):
    shap_output = np.array(shap_output)

    # Case: List (multi-class)
    if isinstance(shap_output, list):
        # Assume binary → use class 1
        shap_output = np.array(shap_output[-1])

    # Case: 3D array (1, n_features, n_classes)
    if shap_output.ndim == 3:
        # Use class 1 (positive class)
        return shap_output[0, :, -1]

    # Case: 2D array (n_samples, n_features)
    if shap_output.ndim == 2:
        return shap_output[0]

    # Case: 1D vector
    if shap_output.ndim == 1:
        return shap_output

    raise ValueError("Unsupported SHAP output shape:", shap_output.shape)



# -----------------------------
# Load models
# -----------------------------
imputer = joblib.load(os.path.join(ARTIFACT_DIR, "imputer.pkl"))
scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
meta_clf = joblib.load(os.path.join(ARTIFACT_DIR, "meta_clf.pkl"))

lr_base = joblib.load(os.path.join(ARTIFACT_DIR, "lr_base.pkl")) if os.path.exists(os.path.join(ARTIFACT_DIR,"lr_base.pkl")) else None
rf_base = joblib.load(os.path.join(ARTIFACT_DIR, "rf_base.pkl")) if os.path.exists(os.path.join(ARTIFACT_DIR,"rf_base.pkl")) else None

booster = xgb.Booster()
booster.load_model(os.path.join(ARTIFACT_DIR, "xgb_booster_full.json"))

with open(os.path.join(ARTIFACT_DIR, "threshold.txt"), "r") as f:
    threshold = float(f.read().strip())

st.sidebar.markdown(f"**Model Threshold (F1):** `{threshold:.4f}`")

# -----------------------------
# User input section
# -----------------------------
st.sidebar.header("🧍 Patient Information (Matching Training Columns)")

male = 1 if st.sidebar.selectbox("Sex", ["Male", "Female"]) == "Male" else 0
age = st.sidebar.number_input("Age", 20, 120, 60, 1)
education = st.sidebar.selectbox("Education Level", [1,2,3,4], index=0)

currentSmoker = st.sidebar.selectbox("Current Smoker?", ["No", "Yes"]) == "Yes"
currentSmoker_num = 1 if currentSmoker else 0
cigsPerDay = st.sidebar.number_input("Cigarettes/day", 0, 100, 0, 1)

BPMeds = st.sidebar.selectbox("On BP Medication?", ["No", "Yes"]) == "Yes"
BPMeds_num = 1 if BPMeds else 0

prevalentStroke = st.sidebar.selectbox("Past Stroke?", ["No", "Yes"]) == "Yes"
prevalentStroke_num = 1 if prevalentStroke else 0

prevalentHyp = st.sidebar.selectbox("Hypertension History?", ["No", "Yes"]) == "Yes"
prevalentHyp_num = 1 if prevalentHyp else 0

diabetes = st.sidebar.selectbox("Diabetes?", ["No", "Yes"]) == "Yes"
diabetes_num = 1 if diabetes else 0

totChol = st.sidebar.number_input("Total Cholesterol", 50, 500, 200, 1)
sysBP = st.sidebar.number_input("Systolic BP", 70, 250, 130, 1)
diaBP = st.sidebar.number_input("Diastolic BP", 40, 160, 85, 1)

BMI = st.sidebar.number_input("BMI", 10.0, 60.0, 25.0, 0.1)
heartRate = st.sidebar.number_input("Heart Rate", 30, 150, 72, 1)
glucose = st.sidebar.number_input("Glucose", 40, 400, 100, 1)

# -----------------------------
# Build input feature set (MATCHES TRAINING)
# -----------------------------
input_dict = {
    "male": male,
    "age": age,
    "education": education,
    "currentSmoker": currentSmoker_num,
    "cigsPerDay": cigsPerDay,
    "BPMeds": BPMeds_num,
    "prevalentStroke": prevalentStroke_num,
    "prevalentHyp": prevalentHyp_num,
    "diabetes": diabetes_num,
    "totChol": totChol,
    "sysBP": sysBP,
    "diaBP": diaBP,
    "BMI": BMI,
    "heartRate": heartRate,
    "glucose": glucose,

    # engineered
    "pulse_pressure": sysBP - diaBP,
    "chol_per_bmi": totChol / (BMI + 1e-6),
    "age_sysbp": age * sysBP,
}

input_df = pd.DataFrame([input_dict])
st.subheader("📋 Input Summary")
st.table(input_df.T)

# -----------------------------
# Preprocess
# -----------------------------
X_imp = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)
X_scaled = pd.DataFrame(scaler.transform(X_imp), columns=input_df.columns)

# -----------------------------
# Base model probabilities
# -----------------------------
lr_p = lr_base.predict_proba(X_scaled)[0][1] if lr_base else 0.0
rf_p = rf_base.predict_proba(X_scaled)[0][1] if rf_base else 0.0
xgb_p = booster.predict(xgb.DMatrix(X_imp.values))[0]

meta_features = pd.DataFrame([{
    "lr": lr_p,
    "rf": rf_p,
    "xgb": xgb_p
}])

meta_proba = meta_clf.predict_proba(meta_features)[0][1]

# -----------------------------
# Risk classification
# -----------------------------
risk_label = "High Risk" if meta_proba >= threshold else "Low Risk"

# Risk Category
if meta_proba < 0.10:
    risk_cat = "🟢 Low Risk (<10%)"
elif meta_proba < 0.20:
    risk_cat = "🟡 Medium Risk (10–20%)"
else:
    risk_cat = "🔴 High Risk (>20%)"

st.markdown("## 🧠 Prediction Result")
st.metric("10-year CHD Risk", f"{meta_proba*100:.2f}%")
st.write(f"**Risk Category:** {risk_cat}")

# -----------------------------
# Base model contributions
# -----------------------------
st.subheader("📊 Base Model Contributions")
st.table(pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Probability": [lr_p, rf_p, xgb_p]
}))

# -----------------------------
# SHAP Explainability
# -----------------------------
st.sidebar.header("Explainability")
do_shap = st.sidebar.checkbox("Show SHAP Explanations", value=False)

if do_shap:
    import shap
    st.subheader("🔍 SHAP Analysis")

    # ---- XGBoost SHAP (single instance) ----
    explainer = shap.TreeExplainer(booster)
    shap_vals = explainer.shap_values(xgb.DMatrix(X_imp.values))[0]

    st.markdown("### XGBoost Feature Impact")

    # Convert to DataFrame
    shap_df = pd.DataFrame({
        "feature": X_imp.columns,
        "shap_value": shap_vals
    }).sort_values("shap_value", key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(shap_df["feature"], shap_df["shap_value"])
    ax.invert_yaxis()
    ax.set_title("XGBoost SHAP Feature Impact — Single Prediction")
    st.pyplot(fig)

    # ---- RandomForest SHAP ----
    if rf_base:
        st.markdown("### Random Forest Feature Impact")
        import shap
        explainer_rf = shap.TreeExplainer(rf_base)
        shap_output = explainer_rf.shap_values(X_imp)

        # SAFE extraction (handles 1D, 2D, list, 3D including (1,18,2))
        shap_vals_rf = extract_single_shap_vector(shap_output)

        shap_df2 = pd.DataFrame({
            "feature": X_imp.columns,
            "shap_value": shap_vals_rf
        }).sort_values("shap_value", key=abs, ascending=False)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.barh(shap_df2["feature"], shap_df2["shap_value"])
        ax2.invert_yaxis()
        ax2.set_title("Random Forest SHAP Feature Impact — Single Prediction")
        st.pyplot(fig2)



    # ---- Meta Model (Logistic Regression) ----
    st.markdown("### Meta Learner Coefficient Weights")
    coef_df = pd.DataFrame({
        "Feature": ["lr", "rf", "xgb"],
        "Weight": meta_clf.coef_[0]
    }).sort_values("Weight")

    fig3, ax3 = plt.subplots(figsize=(6,4))
    ax3.barh(coef_df["Feature"], coef_df["Weight"])
    ax3.set_title("Meta Model (Logistic Regression) Weights")
    st.pyplot(fig3)
