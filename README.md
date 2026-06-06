# Fraud Triage

Fraud Triage is a Python research project for building and evaluating a human-in-the-loop fraud routing system. It trains a LightGBM fraud classifier, calibrates the resulting scores, wraps them in conformal prediction sets, and compares routing strategies under analyst-capacity constraints.

The workflow is documented in [notebook.ipynb](notebook.ipynb), with reusable helpers in [src/](src/) and generated figures in [outputs/plots/](outputs/plots/).

## Project Overview

The project models fraud detection as a triage problem instead of a simple binary classification task. Transactions can be:

- automatically approved,
- automatically blocked,
- sent to verification, or
- escalated to an analyst.

The core question is how to balance fraud risk, false positives, analyst workload, and fairness across proxy groups when analyst capacity is limited.

## What This Includes

- Data loading and cleaning utilities for transaction and identity CSVs.
- LightGBM model training with class imbalance handling and validation AUC tracking.
- Probability calibration with temperature scaling and Expected Calibration Error.
- Adaptive prediction set conformal scoring for uncertainty-aware routing.
- Three-zone routing: auto-approve, auto-block, auto-decide, and escalate.
- Capacity-aware routing with a shadow-price policy for analyst slots.
- Analyst and verification simulation for downstream decision quality.
- System-level metrics for risk, cost, false positives, false negatives, and coverage.
- Fairness analysis across proxy groups such as card type, transaction amount bracket, and product type.

## Repository Structure

```text
fraud_triage/
|-- data/                  # Expected local CSV inputs, ignored by git
|-- outputs/plots/         # Generated analysis figures
|-- src/
|   |-- conformal.py       # APS conformal prediction and routing signals
|   |-- data_loader.py     # Loading, cleaning, splitting, and proxy groups
|   |-- evaluation.py      # Risk-coverage and fairness helpers
|   |-- model.py           # LightGBM training and calibration helpers
|   |-- plotting.py        # Plot generation utilities
|   `-- routing.py         # Routing strategies and analyst simulation
|-- notebook.ipynb         # End-to-end experiment notebook
|-- report.pdf             # Project report
`-- requirements.txt       # Python dependencies
```

## Data

The notebook expects IEEE-CIS-style fraud detection CSV files in `data/`:

- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`
- `sample_submission.csv`

These files can be downloaded from [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data).

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Start Jupyter:

```powershell
jupyter notebook notebook.ipynb
```

## Current Results

- LightGBM validation AUC: `0.9522`
- Empirical conformal coverage at alpha `0.10`: `0.9988`
- Empirical conformal coverage at alpha `0.30`: `0.9958`
- Capacity-aware routing reduced the simulated risk from `0.0655` for the model-alone strategy to `0.0371`.
- Capacity-aware routing used `880` of `1,074` analyst slots in the sampled day.
- In the sampled day, capacity-aware routing escalated `8.80%` of transactions and sent `18.05%` to verification.

Selected strategy comparison:

| Metric | Model Alone | Confidence Threshold | Capacity-Aware |
| --- | ---: | ---: | ---: |
| Risk | 0.0655 | 0.0641 | 0.0371 |
| False negative rate | 0.1994 | 0.0954 | 0.1156 |
| False positive rate | 0.0607 | 0.0630 | 0.0343 |
| Escalation rate | 0.0000 | 0.2685 | 0.0880 |
| Verify rate | 0.0000 | 0.0000 | 0.1805 |
| Total cost | 1000.0000 | 27581.5000 | 13141.5000 |
| Cost per correct | 0.1070 | 2.9471 | 1.3648 |


