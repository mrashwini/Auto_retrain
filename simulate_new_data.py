# simulate_new_data.py
import pandas as pd
import numpy as np

df = pd.read_csv("data/train.csv")
# create 5 new rows by sampling and slightly perturbing
new_rows = df.sample(5, replace=True).reset_index(drop=True)
noise = np.random.normal(0, 0.01, new_rows.iloc[:, :-1].shape)
new_rows.iloc[:, :-1] = new_rows.iloc[:, :-1] + noise
df = pd.concat([df, new_rows], ignore_index=True)
df.to_csv("data/train.csv", index=False)
print("Added 5 new rows to data/train.csv")
