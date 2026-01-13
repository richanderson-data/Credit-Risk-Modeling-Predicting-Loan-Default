import streamlit as st
import pandas as pd
from pathlib import Path
import joblib
import json

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "model.joblib"
SCHEMA_PATH = ARTIFACT_DIR / "feature_schema.json"

st.set_page_config(page_title="Loan Default PD Scorer", layout="wide")
st.title("Loan Default Probability Scorer (24-Month Horizon)")

# Load artifacts once
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    schema = json.loads(SCHEMA_PATH.read_text())
    return model, schema

model, schema = load_artifacts()
REQUIRED_FEATURES = schema["required_features"]

st.markdown(
    "Upload a CSV with the required feature columns to receive default probabilities."
)

with st.expander("Required input columns"):
    st.code(", ".join(REQUIRED_FEATURES))

uploaded = st.file_uploader("Upload CSV", type=["csv"])

threshold = st.slider(
    "Decision threshold (flag loans with PD ≥ threshold)",
    min_value=0.0001,
    max_value=0.50,
    value=0.02,
    step=0.0001,
)

def validate_input(df: pd.DataFrame):
    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    for c in ["orig_interest_rate", "orig_upb", "orig_loan_term"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df[["orig_interest_rate", "orig_upb", "orig_loan_term"]].isna().any().any():
        raise ValueError(
            "Numeric conversion failed for one or more rows. "
            "Check orig_interest_rate, orig_upb, orig_loan_term."
        )

    return df

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        df = validate_input(df)

        X = df[REQUIRED_FEATURES].copy()
        proba = model.predict_proba(X)[:, 1]

        out = df.copy()
        out["pd_default_24m"] = proba
        out["flag"] = (out["pd_default_24m"] >= threshold).astype(int)

        out["risk_bucket"] = pd.cut(
            out["pd_default_24m"],
            bins=[-1, 0.01, 0.03, 1],
            labels=["Low", "Medium", "High"],
        )

        st.subheader("Scored Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Loans scored", f"{len(out):,}")
        col2.metric("Flag rate", f"{out['flag'].mean()*100:.2f}%")
        col3.metric("Average PD", f"{out['pd_default_24m'].mean():.4f}")

        st.dataframe(out.head(200), use_container_width=True)

        st.download_button(
            "Download scored CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="scored_loans.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Scoring failed: {e}")
