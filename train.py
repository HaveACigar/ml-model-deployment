import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "monthly_spend",
    "tenure_months",
    "support_tickets",
    "logins_last_30d",
    "discount_ratio",
    "payment_failures",
]


def generate_reference_data():
    X, y = make_classification(
        n_samples=20000,
        n_features=len(FEATURES),
        n_informative=5,
        n_redundant=0,
        weights=[0.75, 0.25],
        class_sep=1.15,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=FEATURES)
    df["monthly_spend"] = np.round(np.interp(df["monthly_spend"], (df["monthly_spend"].min(), df["monthly_spend"].max()), (20, 350)), 2)
    df["tenure_months"] = np.round(np.interp(df["tenure_months"], (df["tenure_months"].min(), df["tenure_months"].max()), (1, 72))).astype(int)
    df["support_tickets"] = np.clip(np.round(np.interp(df["support_tickets"], (df["support_tickets"].min(), df["support_tickets"].max()), (0, 12))), 0, None).astype(int)
    df["logins_last_30d"] = np.clip(np.round(np.interp(df["logins_last_30d"], (df["logins_last_30d"].min(), df["logins_last_30d"].max()), (0, 55))), 0, None).astype(int)
    df["discount_ratio"] = np.round(np.clip(np.interp(df["discount_ratio"], (df["discount_ratio"].min(), df["discount_ratio"].max()), (0, 0.5)), 0, 0.5), 3)
    df["payment_failures"] = np.clip(np.round(np.interp(df["payment_failures"], (df["payment_failures"].min(), df["payment_failures"].max()), (0, 5))), 0, None).astype(int)
    return df, pd.Series(y, name="target")


def main():
    X, y = generate_reference_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "reference_data": X_train.reset_index(drop=True),
            "feature_names": FEATURES,
            "roc_auc": float(auc),
        },
        "models/artifacts.pkl",
        compress=3,
    )


if __name__ == "__main__":
    main()
