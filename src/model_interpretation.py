import joblib
from xgboost import plot_importance
import matplotlib.pyplot as plt
import os

MODEL_PATH = "models/xgb_retention_model.pkl"
PLOTS_DIR = "reports/plots"

def feature_importance():
    model = joblib.load(MODEL_PATH)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plot_importance(model, max_num_features=10)
    plt.title("Top 10 Feature Importances (XGBoost)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/xgb_feature_importance.png", dpi=120)
    plt.close()

    print(f"✅ Feature importance plot saved → {PLOTS_DIR}/xgb_feature_importance.png")

if __name__ == "__main__":
    feature_importance()
