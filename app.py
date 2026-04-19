from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from lime.lime_tabular import LimeTabularExplainer

# Initialize Flask app
app = Flask(__name__)

print("Loading models...")

# Load ML model
ml_model = joblib.load("best_ml_model.pkl")

# Load DL model
dl_model = load_model("best_dl_model.h5")

# Load scaler
scaler = joblib.load("scaler.pkl")

print("Models loaded successfully!")

# ---------------- FEATURE NAMES ---------------- #

feature_names = [
    "Time", "Amount",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26",
    "V27", "V28"
]

# IMPORTANT: Ideally use real training data
dummy_data = np.random.normal(size=(1000, 30))

explainer = LimeTabularExplainer(
    dummy_data,
    feature_names=feature_names,
    class_names=["Normal", "Fraud"],
    mode="classification"
)

# ---------------- HYBRID PREDICT FUNCTION ---------------- #

def hybrid_predict(input_data):
    input_scaled = scaler.transform(input_data)

    ml_prob = ml_model.predict_proba(input_scaled)[:, 1]
    dl_prob = dl_model.predict(input_scaled).flatten()

    final_prob = (ml_prob + dl_prob) / 2

    return np.vstack([1-final_prob, final_prob]).T


# ---------------- HOME PAGE ---------------- #

@app.route("/")
def home():
    return render_template("manual_input.html")


# ---------------- MANUAL PREDICTION ---------------- #

@app.route("/predict_manual", methods=["POST"])
def predict_manual():
    try:
        # Get values in correct order
        Time = float(request.form["Time"])
        Amount = float(request.form["Amount"])

        V_features = []
        for i in range(1, 29):
            V_features.append(float(request.form[f"V{i}"]))

        # Combine in correct order
        values = [Time, Amount] + V_features

        data = np.array(values).reshape(1, -1)

        # Scale
        data_scaled = scaler.transform(data)

        # ML prediction
        ml_prob = ml_model.predict_proba(data_scaled)[0][1]

        # DL prediction
        dl_prob = dl_model.predict(data_scaled)[0][0]

        # Hybrid
        final_prob = (ml_prob + dl_prob) / 2

        if final_prob > 0.5:
            result = "Fraud Transaction"
        else:
            result = "Normal Transaction"

        # -------- LIME EXPLANATION -------- #

        exp = explainer.explain_instance(
            data[0],
            hybrid_predict,
            num_features=10
        )

        explanation = exp.as_list()

        return render_template(
            "result.html",
            prediction=result,
            probability=round(final_prob * 100, 2),
            explanation=explanation
        )

    except Exception as e:
        return f"Error occurred: {str(e)}"


# ---------------- RUN APP ---------------- #

if __name__ == "__main__":
    app.run(debug=True)