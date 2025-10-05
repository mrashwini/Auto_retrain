import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import json
from datetime import datetime

# Make sure model_store exists
os.makedirs("model_store", exist_ok=True)

# Load data
df = pd.read_csv("data/train.csv")

# Features (X) and Labels (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate accuracy
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"✅ Model trained successfully with accuracy: {acc:.2f}")

# Save model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"model_{timestamp}.pkl"
joblib.dump(model, f"model_store/{model_name}")

# Save accuracy info
with open(f"model_store/{model_name}.json", "w") as f:
    json.dump({"accuracy": acc, "timestamp": timestamp}, f, indent=2)

# Update 'latest.txt' to point to this model
with open("model_store/latest.txt", "w") as f:
    f.write(model_name)

print(f"📦 Saved model as {model_name}")
