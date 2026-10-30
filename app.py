"""Flask website for the heart disease prediction model."""
import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

model = joblib.load(os.path.join(BASE, "model.pkl"))
with open(os.path.join(BASE, "metrics.json")) as f:
    metrics = json.load(f)

# Form definition: one entry per model input (same column names as the training data)
FIELDS = [
    {"name": "Age", "label": "Age (years)", "kind": "number", "min": 18, "max": 100, "default": 54, "step": 1},
    {"name": "Sex", "label": "Sex", "kind": "select", "default": "M",
     "options": [("M", "Male"), ("F", "Female")]},
    {"name": "ChestPainType", "label": "Chest pain type", "kind": "select", "default": "ASY",
     "options": [("ASY", "Asymptomatic"), ("ATA", "Atypical angina"),
                 ("NAP", "Non-anginal pain"), ("TA", "Typical angina")]},
    {"name": "RestingBP", "label": "Resting blood pressure (mm Hg)", "kind": "number",
     "min": 50, "max": 250, "default": 130, "step": 1},
    {"name": "Cholesterol", "label": "Cholesterol (mg/dl, 0 if unknown)", "kind": "number",
     "min": 0, "max": 700, "default": 220, "step": 1},
    {"name": "FastingBS", "label": "Fasting blood sugar above 120 mg/dl", "kind": "select", "default": "0",
     "options": [("0", "No"), ("1", "Yes")]},
    {"name": "RestingECG", "label": "Resting ECG", "kind": "select", "default": "Normal",
     "options": [("Normal", "Normal"), ("ST", "ST-T wave abnormality"), ("LVH", "Left ventricular hypertrophy")]},
    {"name": "MaxHR", "label": "Maximum heart rate (bpm)", "kind": "number",
     "min": 50, "max": 250, "default": 138, "step": 1},
    {"name": "ExerciseAngina", "label": "Chest pain during exercise", "kind": "select", "default": "N",
     "options": [("N", "No"), ("Y", "Yes")]},
    {"name": "Oldpeak", "label": "ST depression (Oldpeak)", "kind": "number",
     "min": -3, "max": 8, "default": 0.9, "step": 0.1},
    {"name": "ST_Slope", "label": "Slope of ST segment", "kind": "select", "default": "Flat",
     "options": [("Up", "Upsloping"), ("Flat", "Flat"), ("Down", "Downsloping")]},
]

# Example patients so people can try the site in one click (invented, not rows from the dataset)
SAMPLES = {
    "high": {"Age": 63, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 150, "Cholesterol": 250,
             "FastingBS": "1", "RestingECG": "ST", "MaxHR": 110, "ExerciseAngina": "Y",
             "Oldpeak": 2.5, "ST_Slope": "Flat"},
    "low": {"Age": 35, "Sex": "F", "ChestPainType": "ATA", "RestingBP": 118, "Cholesterol": 190,
            "FastingBS": "0", "RestingECG": "Normal", "MaxHR": 172, "ExerciseAngina": "N",
            "Oldpeak": 0.0, "ST_Slope": "Up"},
}


@app.route("/")
def index():
    ranked = sorted(metrics["models"].items(), key=lambda kv: kv[1]["cv_accuracy"], reverse=True)
    return render_template("index.html", fields=FIELDS, samples=SAMPLES, metrics=metrics, ranked=ranked)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    row = {}
    for f in FIELDS:
        raw = payload.get(f["name"])
        if raw in (None, ""):
            return jsonify(error=f"Please fill in: {f['label']}"), 400
        if f["kind"] == "number":
            try:
                val = float(raw)
            except ValueError:
                return jsonify(error=f"{f['label']} must be a number."), 400
            if not (f["min"] <= val <= f["max"]):
                return jsonify(error=f"{f['label']} should be between {f['min']} and {f['max']}."), 400
            row[f["name"]] = val
        else:
            if str(raw) not in [o[0] for o in f["options"]]:
                return jsonify(error=f"Invalid choice for {f['label']}."), 400
            row[f["name"]] = int(raw) if f["name"] == "FastingBS" else str(raw)

    # The training data used 0 to mean "not recorded" for cholesterol; the model expects a missing value
    if row["Cholesterol"] == 0:
        row["Cholesterol"] = np.nan

    p = float(model.predict_proba(pd.DataFrame([row]))[0, 1])
    return jsonify(p_disease=round(p, 4), label="High risk" if p >= 0.5 else "Low risk")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
