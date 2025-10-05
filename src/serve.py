# src/serve.py
from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd
import traceback

MODEL_DIR = "model_store"
LATEST_FILE = os.path.join(MODEL_DIR, "latest.txt")

app = Flask(__name__)

def load_latest_model():
    """Load the model referenced in model_store/latest.txt"""
    try:
        with open(LATEST_FILE, "r") as f:
            model_name = f.read().strip()
    except Exception:
        raise RuntimeError("latest.txt not found in model_store. Train a model first.")
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    return model

# 🌐 Homepage
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "🚀 Auto-Retrain ML Model API is running successfully!",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST, body: {\"instances\": [[...],[...]]})"
        }
    }), 200

# ✅ Health check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# 🔮 Prediction
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON:
    {
      "instances": [[feature1, feature2, feature3, feature4], ...]
    }
    """
    try:
        payload = request.get_json(force=True)
        instances = payload.get("instances")

        if not instances:
            return jsonify({"error": "Missing 'instances' key in JSON body."}), 400

        # Convert to DataFrame
        df = pd.DataFrame(instances)

        # Load latest model and predict
        model = load_latest_model()
        preds = model.predict(df).tolist()
        return jsonify({"predictions": preds, "error": None})
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"predictions": None, "error": str(e), "trace": tb}), 500

# 🧹 Ignore favicon requests
@app.route("/favicon.ico")
def favicon():
    return ("", 204)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
