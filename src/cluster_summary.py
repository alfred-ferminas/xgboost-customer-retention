import pandas as pd
import numpy as np

DATA_PATH = "data/processed/segmented_customers.csv"
OUTPUT_PATH = "reports/cluster_summary.csv"


def summarize_clusters():
    df = pd.read_csv(DATA_PATH)
    print(f"📂 Loaded segmented data → {df.shape}")

    # Select numeric features for aggregation
    num_cols = [
        "Age",
        "Total Spend",
        "Items Purchased",
        "Avg Spend Per Item",
        "Loyalty Score",
        "RFM_Score",
        "Satisfaction Encoded",
        "Membership Encoded"
    ]

    # Group by cluster and compute stats
    cluster_summary = df.groupby("CustomerSegment")[num_cols].agg(["mean", "median", "min", "max"])
    cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]
    cluster_summary.reset_index(inplace=True)

    # Add cluster sizes and proportions
    cluster_sizes = df["CustomerSegment"].value_counts().sort_index()
    cluster_summary["Cluster_Size"] = cluster_sizes.values
    cluster_summary["Cluster_Percentage"] = (cluster_sizes.values / len(df) * 100).round(2)

    # Save report
    cluster_summary.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Cluster summary saved → {OUTPUT_PATH}")

    # Display short preview
    print("\n📊 Cluster Summary Preview:")
    print(cluster_summary.head())


if __name__ == "__main__":
    summarize_clusters()
