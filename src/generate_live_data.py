import pandas as pd
import numpy as np
import json
import os

# Load historical dataset
df = pd.read_csv("data/raw/historical_customer_behavior.csv")

# Number of synthetic records
n_new = 200

# Helper to sample realistic values
def sample_like(col):
    if df[col].dtype == "O":  # categorical
        return np.random.choice(df[col].dropna().unique())
    else:
        return df[col].dropna().sample(1).iloc[0]

# Build synthetic data
new_data = []
for i in range(n_new):
    record = {}
    for col in df.columns:
        value = sample_like(col)
        # Convert numpy types to native Python
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        record[col] = value
    record["Customer ID"] = int(10000 + i)
    new_data.append(record)

# Make sure output folder exists
os.makedirs("data/raw", exist_ok=True)

# Custom converter for leftover numpy/object types
def convert(o):
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)

# Save JSON safely
out_path = "data/raw/live_customer_data.json"
with open(out_path, "w") as f:
    json.dump(new_data, f, indent=4, default=convert)

print(f"✅ Generated {n_new} new records → {out_path}")

