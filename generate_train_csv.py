import pandas as pd
from sklearn.datasets import make_classification
import os

os.makedirs("data", exist_ok=True)

X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=3,
    n_redundant=1,
    random_state=42
)
df = pd.DataFrame(X, columns=["feature1", "feature2", "feature3", "feature4"])
df["label"] = y
df.to_csv("data/train.csv", index=False)
print("✅ Created data/train.csv")
