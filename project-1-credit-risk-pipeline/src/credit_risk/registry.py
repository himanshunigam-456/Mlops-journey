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
    name: str = REGISTERED_MODEL_NAME,
) -> ModelVersion:
    """Register the model logged in ``run_id`` under ``name``.

    Args:
        run_id: the MLflow run that contains the model artifact.
        stage: if given, immediately transition the new version to this stage.
        artifact_path: the sub-path within the run's artifacts (default 'model').
        name: registered model name. Defaults to the production constant;
            tests pass a sandboxed name to avoid polluting the prod model.

    Promoting to a stage archives any previous versions already at that stage —
    enforces the "one Staging, one Production at a time" invariant.
    """
    client = MlflowClient()

    try:
        client.create_registered_model(name)
    except mlflow.exceptions.RestException:
        pass  # already exists — idempotent

    model_version = client.create_model_version(
        name=name,
        source=f"runs:/{run_id}/{artifact_path}",
        run_id=run_id,
    )

    if stage is not None:
        client.transition_model_version_stage(
            name=name,
            version=model_version.version,
            stage=stage,
            archive_existing_versions=True,
        )
        model_version = client.get_model_version(name=name, version=model_version.version)

    return model_version
