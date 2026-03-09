from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model & scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return "Financial Fraud Detection API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    amount = data["amount"]

    features = np.zeros((1, 30))
    features[0, -1] = amount

    features[:, -1] = scaler.transform(features[:, -1].reshape(-1,1)).flatten()

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    result = "Fraud" if prediction == 1 else "Genuine"

    return jsonify({
        "prediction": result,
        "fraud_probability": float(probability)
    })
    amount = data["amount"]

    # Create feature array (dummy zeros for other features)
    features = np.zeros((1, 30))
    features[0, -1] = amount

    # Scale amount
    features[:, -1] = scaler.transform(features[:, -1].reshape(-1,1)).flatten()

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    result = "Fraud" if prediction == 1 else "Genuine"

    return jsonify({
        "prediction": result,
        "fraud_probability": float(probability)
    })

if __name__ == "__main__":
   import os

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)