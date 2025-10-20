import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import os

DATA_PATH = "data/processed/segmented_customers.csv"
MODEL_PATH = "models/xgb_retention_model.pkl"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"📂 Loaded data → {df.shape}")

    # Define target: churn = not a recent buyer
    df["Churn"] = (df["Recent Buyer"] == 0).astype(int)

    features = [
        "Age",
        "Total Spend",
        "Items Purchased",
        "Satisfaction Encoded",
        "Membership Encoded",
        "Avg Spend Per Item",
        "High Spender",
        "Loyalty Score",
        "RFM_Score",
        "CustomerSegment"   # new feature
    ]

    X = df[features]
    y = df["Churn"]
    return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=250,
        learning_rate=0.08,
        max_depth=6,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    print("🚀 Training enhanced model with CustomerSegment...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, preds)
    print(f"✅ Accuracy: {acc:.3f} | ROC-AUC: {auc:.3f}")
    print(classification_report(y_test, preds))

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    X, y = load_data()
    train_model(X, y)
