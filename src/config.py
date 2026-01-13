from pathlib import Path

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
