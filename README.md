# Credit Risk Modeling – Predicting Loan Default

End-to-end credit risk modeling project using Fannie Mae mortgage data.  
This repository demonstrates a full pipeline from data preparation and model training
to a deployable inference application for scoring new loans.

The project is intentionally structured to reflect production-oriented workflows:
training and inference are separated, model artifacts are persisted, and scoring is
exposed through a lightweight application interface.

---

## Project Overview

**Objective:**  
Estimate the probability that a mortgage loan defaults within 24 months of origination,
using only origination-time features (no post-origination data leakage).

**Key components:**
- Data ingestion, labeling, and feature engineering
- Baseline probability-of-default (PD) model
- Reproducible training pipeline with persisted artifacts
- Browser-based scoring app for inference

---

## Repository Structure

```
.
├── src/
│   ├── train.py              # Training pipeline (produces model artifacts)
│   └── config.py             # Paths and configuration
├── app/
│   └── app.py                # Streamlit inference/scoring app
├── artifacts/
│   ├── model.joblib          # Trained model + preprocessing pipeline
│   ├── feature_schema.json   # Required input feature contract
│   └── metadata.json         # Training metadata and metrics
├── Notebooks/
│   ├── 01_data_ingestion_and_labeling.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling_and_evaluation.ipynb
├── data/
│   └── processed/            # Processed feature data (raw files excluded)
├── requirements.txt
└── README.md
```

---

## Data

The model is trained using **Fannie Mae Single-Family Loan Performance data**.

- Raw quarterly loan files are **not included** in this repository due to size.
- Labeling logic and feature construction are demonstrated in the notebooks.
- The final model uses **origination-level features only**, avoiding look-ahead bias
  and data leakage.

---

## Model

- **Algorithm:** Logistic Regression (baseline)
- **Target:** 24-month default indicator
- **Class imbalance:** handled via class weighting
- **Evaluation metrics:**
  - ROC AUC
  - Precision–Recall AUC
- **Thresholding:** separated from training and configurable at inference time

The baseline model is intentionally simple and interpretable. The emphasis of this
project is on **end-to-end system design and reproducibility**, not model complexity.

---

## Training

From the project root:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python3 -m src.train
```

This will:
- Load processed feature data
- Train the model
- Persist artifacts to the `artifacts/` directory

---

## Scoring App (Streamlit)

A lightweight inference application is included to score new loans using the trained
model artifact. The app loads the persisted preprocessing + model pipeline and applies
it deterministically at inference time.

### Run the app

```bash
streamlit run app/app.py
```

The application opens in a browser and allows users to upload a CSV for scoring.

---

## App Input Contract

The app expects a **clean origination-level feature CSV** with the following columns:

- `orig_interest_rate`
- `orig_upb`
- `orig_loan_term`
- `property_type`
- `loan_purpose`
- `property_state`
- `loan_type`

Raw Fannie Mae quarterly performance files are **intentionally not accepted** by the app.  
In a production system, ETL and labeling occur upstream; the scoring service operates on
validated feature tables only.

---

## App Output

For each loan, the app produces:

- `pd_default_24m` — predicted probability of default within 24 months
- `flag` — binary decision based on a configurable threshold
- `risk_bucket` — coarse risk category (Low / Medium / High) for reporting

The decision threshold can be adjusted at runtime to support different risk policies
(e.g., screening vs. underwriting).

Scored results can be downloaded as a CSV.

---

## Notes on Modeling Choices

- Probabilities are **uncalibrated** and optimized for ranking and screening.
- Class imbalance is handled via class weighting.
- Threshold selection is policy-dependent and intentionally separated from training.
- The baseline model is intentionally simple; the emphasis of this project is on
  end-to-end system design and reproducibility rather than model complexity.

---

## License

MIT License
