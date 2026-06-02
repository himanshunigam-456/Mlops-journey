"""MLflow Model Registry helpers for the credit-risk pipeline."""

from __future__ import annotations

from typing import Final, Literal

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

REGISTERED_MODEL_NAME: Final[str] = "credit-risk-classifier"

Stage = Literal["Staging", "Production", "Archived"]


def register_model_from_run(
    run_id: str,
    *,
    stage: Stage | None = None,
    artifact_path: str = "model",
) -> ModelVersion:
    """Register the model logged in `run_id` under REGISTERED_MODEL_NAME.

    Args:
        run_id: the MLflow run that contains the model artifact.
        stage: if given, immediately transition the new version to this stage.
        artifact_path: the sub-path within the run's artifacts (default 'model').
    """
    client = MlflowClient()

    # Ensure the registered model exists (idempotent — no-op if already there).
    try:
        client.create_registered_model(REGISTERED_MODEL_NAME)
    except mlflow.exceptions.RestException:
        pass  # already exists

    model_version = client.create_model_version(
        name=REGISTERED_MODEL_NAME,
        source=f"runs:/{run_id}/{artifact_path}",
        run_id=run_id,
    )

    if stage is not None:
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=model_version.version,
            stage=stage,
            archive_existing_versions=False,
        )
        # Re-fetch so caller sees the updated stage in the returned object.
        model_version = client.get_model_version(
            name=REGISTERED_MODEL_NAME, version=model_version.version
        )

    return model_version
