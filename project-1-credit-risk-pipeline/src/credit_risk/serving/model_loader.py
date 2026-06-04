"""Load the Staging-stage model from MLflow Registry once at app startup."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from credit_risk.registry import REGISTERED_MODEL_NAME

STAGE = "Staging"


@dataclass(frozen=True)
class ModelBundle:
    """Everything the predict endpoint needs in memory."""

    model: Any
    version: int
    feature_names: list[str]


def load_staging_model() -> ModelBundle:
    """Load the latest Staging model + its signature's input columns.

    Reads MLFLOW_TRACKING_URI from env (default http://localhost:5000).
    Falls back to ``model.feature_names_in_`` if the signature is missing.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(uri)

    client = MlflowClient()
    versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=[STAGE])
    if not versions:
        raise RuntimeError(f"No '{STAGE}'-stage version found for model '{REGISTERED_MODEL_NAME}'.")
    mv = versions[0]

    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{STAGE}")

    feature_names: list[str] = []
    try:
        info = client.get_model_version(name=mv.name, version=mv.version)
        run = client.get_run(info.run_id)
        mlflow_model = mlflow.models.Model.load(f"{run.info.artifact_uri}/model")
        sig = mlflow_model.signature
        if sig and sig.inputs:
            feature_names = [c.name for c in sig.inputs.inputs if c.name]
    except Exception:
        feature_names = list(getattr(model, "feature_names_in_", []) or [])

    return ModelBundle(model=model, version=int(mv.version), feature_names=feature_names)
