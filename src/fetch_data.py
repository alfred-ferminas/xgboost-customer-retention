"""
fetch_data.py
----------------
Handles data ingestion for the XGBoost Customer Retention project.

Functions:
    - load_historical_data(): Loads historical data from CSV.
    - fetch_live_data(): Pulls JSON data from a hosted API (Gist or similar).
    - combine_datasets(): Merges historical and live data into one file.
"""

import os
import pandas as pd
import requests

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"


def load_historical_data(file_path=f"{RAW_DATA_DIR}/historical_customer_behavior.csv") -> pd.DataFrame:
    """Loads the historical dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")
    print(f"📂 Loading historical data from {file_path}")
    df = pd.read_csv(file_path)
    print(f"✅ Historical data shape: {df.shape}")
    return df


def fetch_live_data(api_url: str) -> pd.DataFrame:
    """Fetches live customer data from hosted JSON (API endpoint)."""
    print(f"🌐 Fetching live data from {api_url}")
    response = requests.get(api_url)
    response.raise_for_status()
    data = response.json()
    df_live = pd.DataFrame(data)
    out_path = f"{RAW_DATA_DIR}/live_customer_data.csv"
    df_live.to_csv(out_path, index=False)
    print(f"✅ Live data saved to {out_path} | Shape: {df_live.shape}")
    return df_live


def combine_datasets(df_hist: pd.DataFrame, df_live: pd.DataFrame) -> pd.DataFrame:
    """Combines historical and live data, removing duplicates."""
    combined = pd.concat([df_hist, df_live], ignore_index=True)
    combined.drop_duplicates(subset=["Customer ID"], inplace=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    out_path = f"{PROCESSED_DATA_DIR}/combined_data.csv"
    combined.to_csv(out_path, index=False)
    print(f"📦 Combined data saved → {out_path} | Shape: {combined.shape}")
    return combined


if __name__ == "__main__":
    # Step 1: Load historical data
    df_hist = load_historical_data()

    # Step 2: Fetch live data (replace with your Gist URL)
    api_url = "https://gist.githubusercontent.com/alfred-ferminas/2349e9b6d8543ee7b0c77e0fb7b24dd2/raw/52265e7b9510b5face733fa7b2c10e554a3afe1f/live_customer_data.json"
    df_live = fetch_live_data(api_url)

    # Step 3: Combine and save
    combine_datasets(df_hist, df_live)
