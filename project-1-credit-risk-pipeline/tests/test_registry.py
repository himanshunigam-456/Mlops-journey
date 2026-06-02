"""Integration tests for the MLflow Model Registry helpers.

These require the docker-compose stack to be UP. Skipped otherwise so the
test suite still runs cleanly offline (CI sets up MLflow separately).
"""

import os
import urllib.request

import mlflow
import pytest
from credit_risk.registry import REGISTERED_MODEL_NAME, register_model_from_run
from mlflow.tracking import MlflowClient


@pytest.fixture(scope="module")
def mlflow_up() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    try:
        urllib.request.urlopen(uri + "/health", timeout=2)
    except Exception:
        pytest.skip(f"MLflow server not reachable at {uri} — run `make up` first")
    mlflow.set_tracking_uri(uri)
    return uri


@pytest.fixture
def fake_logged_run(mlflow_up):
    """Create a throwaway run with a tiny sklearn model logged as the artifact."""
    from sklearn.linear_model import LogisticRegression

    mlflow.set_experiment("test-registry")
    with mlflow.start_run() as run:
        clf = LogisticRegression().fit([[0], [1], [2], [3]], [0, 0, 1, 1])
        mlflow.sklearn.log_model(clf, artifact_path="model")
    return run.info.run_id


def test_register_returns_version_object(fake_logged_run):
    mv = register_model_from_run(fake_logged_run, stage=None)
    assert mv.name == REGISTERED_MODEL_NAME
    assert int(mv.version) >= 1


def test_register_with_staging_promotes_to_staging(fake_logged_run):
    mv = register_model_from_run(fake_logged_run, stage="Staging")
    client = MlflowClient()
    fresh = client.get_model_version(name=mv.name, version=mv.version)
    assert fresh.current_stage == "Staging"
