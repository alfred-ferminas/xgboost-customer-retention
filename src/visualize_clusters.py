import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = "data/processed/segmented_customers.csv"
PLOTS_DIR = "reports/plots"


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"📂 Loaded segmented data → {df.shape}")
    return df


def visualize_clusters(df):
    # Select only numeric behavioral features for PCA
    features = [
        "Total Spend",
        "Items Purchased",
        "Age",
        "Avg Spend Per Item",
        "Loyalty Score",
        "RFM_Score"
    ]
    X = df[features].fillna(0)

    # Standardize features before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reduce to 2 components for visualization
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(X_scaled)

    df["PCA1"] = pca_result[:, 0]
    df["PCA2"] = pca_result[:, 1]

    os.makedirs(PLOTS_DIR, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="PCA1",
        y="PCA2",
        hue="CustomerSegment",
        palette="tab10",
        alpha=0.8,
        s=60,
        edgecolor="white"
    )
    plt.title("Customer Segments (PCA Projection)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Segment", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/customer_segments_pca.png", dpi=120)
    plt.close()

    print(f"✅ PCA scatterplot saved → {PLOTS_DIR}/customer_segments_pca.png")
    print("Explained variance ratios:", pca.explained_variance_ratio_)


if __name__ == "__main__":
    df = load_data()
    visualize_clusters(df)
