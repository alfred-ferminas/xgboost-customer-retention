import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

DATA_PATH = "data/processed/derived_features.csv"
OUTPUT_PATH = "data/processed/segmented_customers.csv"
PLOTS_DIR = "reports/plots"


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"📂 Loaded data → {df.shape}")
    return df


def perform_clustering(df):
    # Select behavioral features only
    features = [
        "Total Spend",
        "Items Purchased",
        "Age",
        "Avg Spend Per Item",
        "Loyalty Score",
        "RFM_Score"
    ]
    X = df[features].fillna(0)

    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method to find optimal K
    inertias = []
    K_range = range(2, 10)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.plot(K_range, inertias, marker='o')
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.savefig(f"{PLOTS_DIR}/elbow_method.png", dpi=120)
    plt.close()

    # Choose K manually after checking the elbow plot
    best_k = 4
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["CustomerSegment"] = km.fit_predict(X_scaled)

    print(f"✅ Clustering complete with K={best_k}")
    return df


def save_results(df):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"💾 Segmented data saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    df = load_data()
    df = perform_clustering(df)
    save_results(df)
