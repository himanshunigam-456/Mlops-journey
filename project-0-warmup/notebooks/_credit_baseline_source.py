# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python (mlops-journey)
#     language: python
#     name: mlops-journey
# ---

# %% [markdown]
# # Credit Baseline — Week 1 End-to-End
#
# Loads the German Credit dataset from UCI (small enough for a warmup), trains
# the baseline RandomForest defined in `credit_baseline.pipeline`, and logs the
# run to MLflow with the model stored in MinIO.
#
# This is the **vertical slice** for Project 1: data → model → tracked run.

# %%
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from credit_baseline.pipeline import train_baseline

# %% [markdown]
# ## Load German Credit dataset
# UCI provides it at a stable URL. ~1000 rows, 20 features, binary target
# (1 = good credit, 2 = bad credit → recoded to 1 = default).

# %%
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/" "statlog/german/german.data"
COLS = [
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_commitment",
    "personal_status",
    "other_parties",
    "residence_since",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
    "default",
]

df = pd.read_csv(UCI_URL, sep=" ", header=None, names=COLS)
# Recode target: 1=good → 0 (no default), 2=bad → 1 (default)
df["default"] = (df["default"] == 2).astype(int)

# One-hot encode the categorical columns (RandomForest needs numeric input)
df = pd.get_dummies(df, drop_first=True)
print(f"Shape: {df.shape}, default rate: {df['default'].mean():.2%}")
df.head()

# %% [markdown]
# ## Train and log to MLflow
#
# `MLFLOW_TRACKING_URI` is set by direnv to `http://localhost:5000`.
# The model artifact lands in MinIO bucket `mlflow/` via the S3 API.

# %%
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("credit-baseline")

with mlflow.start_run(run_name="warmup-randomforest"):
    result = train_baseline(df, target_col="default", random_state=42)

    mlflow.log_params(
        {
            "model": "RandomForestClassifier",
            "n_estimators": 200,
            "max_depth": 8,
            "random_state": 42,
            "dataset": "UCI German Credit",
            "n_features": len(df.columns) - 1,
        }
    )
    mlflow.log_metrics(
        {
            "accuracy": result.accuracy,
            "roc_auc": result.roc_auc,
            "n_train": result.n_train,
            "n_test": result.n_test,
        }
    )
    # MLflow 2.x API: use artifact_path (3.x switched to `name=`)
    mlflow.sklearn.log_model(result.model, artifact_path="model")

    print(f"accuracy: {result.accuracy:.3f}  roc_auc: {result.roc_auc:.3f}")

# %% [markdown]
# ## Verify the run lands in MLflow
#
# Open http://localhost:5000 — you should see one experiment "credit-baseline"
# with one run named "warmup-randomforest", including the model artifact in
# MinIO under s3://mlflow/.
