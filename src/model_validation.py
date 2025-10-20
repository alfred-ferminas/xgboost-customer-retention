import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

DATA_PATH = "data/processed/segmented_customers.csv"

def validate_model():
    df = pd.read_csv(DATA_PATH)
    df["Churn"] = (df["Recent Buyer"] == 0).astype(int)

    features = [
        "Age", "Total Spend", "Items Purchased", "Satisfaction Encoded",
        "Membership Encoded", "Avg Spend Per Item", "High Spender",
        "Loyalty Score", "RFM_Score", "CustomerSegment"
    ]
    X = df[features]
    y = df["Churn"]

    model = XGBClassifier(
        n_estimators=250,
        learning_rate=0.08,
        max_depth=6,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold, scoring="roc_auc")
    print(f"📊 5-Fold Cross-Validation AUC Scores: {np.round(scores, 3)}")
    print(f"Mean AUC: {scores.mean():.3f} | Std: {scores.std():.3f}")

if __name__ == "__main__":
    validate_model()
