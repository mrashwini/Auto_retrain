# src/evaluate.py
import json
import os
import glob

MODEL_DIR = "model_store"
MIN_ACCEPTABLE_ACCURACY = 0.70  # change to your required threshold

def latest_metrics_file():
    files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.json")))
    if not files:
        return None
    return files[-1]

def load_accuracy(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Support both {"accuracy": 0.95} and {"accuracy": 0.95, ...}
    return float(data.get("accuracy", data.get("acc", 0)))

if __name__ == "__main__":
    m = latest_metrics_file()
    if not m:
        print("No metrics file found. Failing evaluation.")
        raise SystemExit(1)
    acc = load_accuracy(m)
    print(f"Latest model accuracy = {acc:.4f}")
    if acc < MIN_ACCEPTABLE_ACCURACY:
        print(f"Accuracy {acc:.4f} is below threshold {MIN_ACCEPTABLE_ACCURACY}. Failing.")
        raise SystemExit(2)
    print("Model passed accuracy gate.")
