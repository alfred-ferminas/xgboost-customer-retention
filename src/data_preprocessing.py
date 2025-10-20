import pandas as pd
import numpy as np
import os

RAW_DATA = "data/processed/combined_data.csv"
PROCESSED_DIR = "data/processed"


def load_data(path=RAW_DATA):
    print(f"📂 Loading data from {path}")
    df = pd.read_csv(path)
    print(f"✅ Shape before cleaning: {df.shape}")
    return df


def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates(subset=["Customer ID"])

    # Handle missing values safely
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["City"] = df["City"].fillna(df["City"].mode()[0])
    df["Membership Type"] = df["Membership Type"].fillna("Unknown")
    df["Satisfaction Level"] = df["Satisfaction Level"].fillna("Neutral")

    # Normalize text columns
    df["Gender"] = df["Gender"].astype(str).str.strip().str.capitalize()
    df["City"] = df["City"].astype(str).str.strip().str.title()
    df["Membership Type"] = df["Membership Type"].astype(str).str.strip().str.title()
    df["Satisfaction Level"] = df["Satisfaction Level"].astype(str).str.strip().str.capitalize()

    print(f"✅ Shape after cleaning: {df.shape}")
    return df


def feature_engineering(df):
    # Satisfaction encoding
    satisfaction_map = {"Unsatisfied": 1, "Neutral": 2, "Satisfied": 3}
    df["Satisfaction Encoded"] = df["Satisfaction Level"].map(satisfaction_map).fillna(2)

    # Membership encoding
    membership_map = {"Silver": 1, "Gold": 2, "Platinum": 3, "Unknown": 0}
    df["Membership Encoded"] = df["Membership Type"].map(membership_map).fillna(0)

    # Behavioral ratios
    df["Avg Spend Per Item"] = df["Total Spend"] / df["Items Purchased"].replace(0, np.nan)
    df["Spend Per Age"] = df["Total Spend"] / df["Age"].replace(0, np.nan)

    # Binary spend features
    df["Recent Buyer"] = (df["Days Since Last Purchase"] < 30).astype(int)
    df["High Spender"] = (df["Total Spend"] > df["Total Spend"].median()).astype(int)
    df["Discount Applied"] = df["Discount Applied"].astype(int)

    # RFM composite (recency, frequency, monetary)
    df["RFM_Score"] = (
        (5 - pd.qcut(df["Days Since Last Purchase"], 5, labels=False, duplicates='drop'))
        + pd.qcut(df["Items Purchased"], 5, labels=False, duplicates='drop')
        + pd.qcut(df["Total Spend"], 5, labels=False, duplicates='drop')
    )

    # Interaction term
    df["Loyalty Score"] = (
        0.4 * (1 - df["Days Since Last Purchase"] / df["Days Since Last Purchase"].max())
        + 0.3 * (df["Satisfaction Encoded"] / 3)
        + 0.3 * (df["Membership Encoded"] / 3)
    )

    print("✨ Added derived features: satisfaction, membership, ratios, RFM, loyalty score")
    return df


def save_processed(df):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = f"{PROCESSED_DIR}/derived_features.csv"
    df.to_csv(out_path, index=False)
    print(f"💾 Saved derived features → {out_path}")


if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)
    save_processed(df)
