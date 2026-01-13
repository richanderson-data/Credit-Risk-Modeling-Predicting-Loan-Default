# src/train.py

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

# Prefer config.py if you have it; otherwise fall back to sensible defaults.
try:
    from .config import DATA_PATH, ARTIFACT_DIR, REQUIRED_FEATURES, TARGET_COL
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_data_2016Q1.parquet"
    ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
    REQUIRED_FEATURES = [
        "orig_interest_rate",
        "orig_upb",
        "orig_loan_term",
        "property_type",
        "loan_purpose",
        "property_state",
        "loan_type",
    ]
    TARGET_COL = "default_24m"


def build_pipeline():
    numeric_features = ["orig_interest_rate", "orig_upb", "orig_loan_term"]
    categorical_features = ["property_type", "loan_purpose", "property_state", "loan_type"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        class_weight="balanced",
    )

    return Pipeline(steps=[("preprocess", preprocess), ("clf", model)])


def main():
    # Resolve project-root relative paths safely
    artifact_dir = Path(ARTIFACT_DIR)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(DATA_PATH)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found at: {data_path}\n"
            f"Expected a parquet like data/processed/model_data_2016Q1.parquet.\n"
            f"Export it from Notebook 02, then re-run."
        )

    df = pd.read_parquet(data_path)

    missing_cols = [c for c in REQUIRED_FEATURES + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in training data: {missing_cols}\n"
            f"Found columns: {list(df.columns)[:30]}{' ...' if len(df.columns) > 30 else ''}"
        )

    X = df[REQUIRED_FEATURES].copy()
    y = df[TARGET_COL].astype(int)

    # Basic sanity checks
    if y.nunique() < 2:
        raise ValueError("Target has only one class. Check label construction / censoring logic.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)

    # --- Save artifacts (this is the part your current script likely lacks) ---
    model_path = artifact_dir / "model.joblib"
    joblib.dump(pipeline, model_path)

    schema_path = artifact_dir / "feature_schema.json"
    schema_path.write_text(json.dumps({"required_features": REQUIRED_FEATURES}, indent=2))

    metadata_path = artifact_dir / "metadata.json"
    metadata = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "data_path": str(data_path),
        "n_rows": int(len(df)),
        "default_rate": float(y.mean()),
        "model": {
            "type": "LogisticRegression",
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 5000,
        },
        "metrics": {
            "roc_auc": float(roc),
            "pr_auc": float(pr),
        },
        "notes": (
            "Baseline PD model. Thresholding is policy-dependent. "
            "For production use: probability calibration, loss-based threshold selection, "
            "and model governance checks (fairness, drift, monitoring)."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # Print confirmation (so you *see* it ran)
    print(f"Saved model: {model_path}")
    print(f"Saved schema: {schema_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"ROC AUC: {roc:.6f}")
    print(f"PR AUC:  {pr:.6f}")
    print(f"Default rate: {y.mean():.6f}  (positives={int(y.sum())}, total={len(y)})")


if __name__ == "__main__":
    main()
