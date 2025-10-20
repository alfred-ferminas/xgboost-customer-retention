import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib
import os

MODEL_PATH = "models/xgb_retention_model.pkl"
DATA_PATH = "data/processed/cleaned_features.csv"
PLOTS_DIR = "reports/plots"


def evaluate_model():
    # Load model and data
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    print(f"📂 Loaded data: {df.shape}")

    # Prepare features and target
    features = [
        "Age",
        "Total Spend",
        "Items Purchased",
        "Satisfaction Encoded",
        "Membership Encoded",
        "Avg Spend Per Item",
        "High Spender",
        "Loyalty Score",
    ]
    X = df[features]
    y = (df["Recent Buyer"] == 0).astype(int)

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    # Confusion matrix
    cm = confusion_matrix(y, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(f"{PLOTS_DIR}/confusion_matrix.png", dpi=120)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"{PLOTS_DIR}/roc_curve.png", dpi=120)
    plt.close()

    # Feature importance
    importance = model.feature_importances_
    imp_df = pd.DataFrame({"Feature": features, "Importance": importance}).sort_values(
        by="Importance", ascending=False
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=imp_df, palette="viridis")
    plt.title("Feature Importance (XGBoost)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/feature_importance.png", dpi=120)
    plt.close()

    print(f"✅ Evaluation complete — plots saved in {PLOTS_DIR}")
    print(f"ROC AUC: {roc_auc:.3f}")


if __name__ == "__main__":
    evaluate_model()
