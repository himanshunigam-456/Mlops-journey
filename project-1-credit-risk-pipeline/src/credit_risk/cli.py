"""Typer CLI for the credit-risk pipeline.

Commands:
    credit-risk train       — 3-trial hyperparameter sweep, log each to MLflow
    credit-risk register    — promote best run to the Model Registry @ Staging
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import mlflow
import mlflow.sklearn
import typer
from mlflow.models import infer_signature

from credit_risk.data_loader import load_german_credit
from credit_risk.features import train_test_split_stratified
from credit_risk.registry import REGISTERED_MODEL_NAME, register_model_from_run
from credit_risk.train import train_xgboost

app = typer.Typer(no_args_is_help=True, help="Credit-risk pipeline CLI.")

EXPERIMENT_NAME = "credit-risk-classifier"

# cli.py is at project-1-credit-risk-pipeline/src/credit_risk/cli.py
# parents[2] resolves to project-1-credit-risk-pipeline/
# Hoisted to module-level (not a function-default) to satisfy ruff B008.
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "german_credit.csv"

# Three trial configs — kept small for laptop runs. Phase 2 expands to a real sweep.
SWEEP_TRIALS = [
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.10},
    {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05},
    {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.03},
]


@app.command()
def train(
    data_path: Annotated[
        Path,
        typer.Option(help="Path to the German Credit CSV."),
    ] = DEFAULT_DATA_PATH,
) -> None:
    """Run the 3-trial sweep, log each trial to MLflow."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_german_credit(data_path)
    split = train_test_split_stratified(df, target_col="default", random_state=42)

    for i, hp in enumerate(SWEEP_TRIALS, start=1):
        with mlflow.start_run(run_name=f"trial-{i}-xgb"):
            result = train_xgboost(split, random_state=42, **hp)
            mlflow.log_params({**result.params, "dataset": "UCI German Credit"})
            mlflow.log_metrics(
                {
                    "accuracy": result.accuracy,
                    "precision": result.precision,
                    "recall": result.recall,
                    "f1": result.f1,
                    "roc_auc": result.roc_auc,
                    "n_train": result.n_train,
                    "n_test": result.n_test,
                }
            )
            signature = infer_signature(split.X_train, result.model.predict(split.X_train))
            mlflow.sklearn.log_model(
                result.model,
                artifact_path="model",
                signature=signature,
                input_example=split.X_train.head(2),
            )
            typer.echo(
                f"  trial-{i}: roc_auc={result.roc_auc:.3f}  "
                f"accuracy={result.accuracy:.3f}  f1={result.f1:.3f}"
            )


@app.command()
def register(
    metric: Annotated[str, typer.Option(help="Metric used to pick the best run.")] = "roc_auc",
    stage: Annotated[str, typer.Option(help="Target stage (None/Staging/Production).")] = "Staging",
) -> None:
    """Find the best run in the current experiment and promote to Staging."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if runs.empty:
        typer.echo(f"No finished runs in experiment '{EXPERIMENT_NAME}'.", err=True)
        raise typer.Exit(code=1)

    best = runs.iloc[0]
    run_id = best["run_id"]
    typer.echo(f"Best run: {run_id[:12]}  {metric}={best[f'metrics.{metric}']:.4f}")

    target_stage = None if stage.lower() in {"none", ""} else stage
    mv = register_model_from_run(run_id, stage=target_stage)
    typer.echo(f"Registered '{REGISTERED_MODEL_NAME}' v{mv.version} at stage '{mv.current_stage}'.")


if __name__ == "__main__":
    app()
