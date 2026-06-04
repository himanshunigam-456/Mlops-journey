# Project 1 · Phase 2 (Week 3) — FastAPI Serving + Dockerization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the Phase 1 model (`credit-risk-classifier` v5 @ Staging) in a production-grade FastAPI service. Validated request/response schemas, model loaded at startup, `/predict`, `/healthz`, `/readyz` endpoints, dockerized with a multi-stage build, published to GitHub Container Registry on every push to `main`, load-tested with Locust.

**Architecture:** A new sub-module `src/credit_risk/serving/` adds: `schemas.py` (Pydantic v2 request/response), `model_loader.py` (loads `models:/credit-risk-classifier/Staging` once at startup via FastAPI lifespan), `app.py` (FastAPI router with `/predict` `/healthz` `/readyz`). A separate `serving/Dockerfile` (multi-stage) produces a slim runtime image. Container connects to the existing docker-compose stack via the `infra_default` network (resolves MLflow as `http://mlflow:5000` from inside the container, `http://localhost:5000` from outside).

**Why "encoded features in, prediction out":** The `/predict` endpoint accepts the **already-one-hot-encoded** feature dict, not raw German Credit columns. In real production, feature engineering lives in a separate upstream layer (feature store, batch transform, or client-side encoder). Coupling serving to encoding makes serving brittle to dataset schema changes. We expose the encoded contract; clients/notebooks handle the encoding. A `examples/encode_and_call.py` helper script demonstrates the full flow.

**Tech Stack:** FastAPI · uvicorn · Pydantic v2 · httpx (test client) · Locust (load test) · Docker multi-stage · GitHub Container Registry · GitHub Actions

**Time budget:** 10 hrs across 5 weekday sessions (2 hrs/day).

**Prerequisites (verified before Day 1 starts):**
- Phase 1 complete: `credit-risk-classifier` v5 @ Staging in MLflow Registry
- 17 / 17 Project-1 tests green
- `make verify` shows 18 / 18 ✅
- `gh` CLI installed + authenticated as `himanshunigam-456`
- Stack is up (`make up`)

---

## File Structure Created This Phase

```
project-1-credit-risk-pipeline/
├── pyproject.toml                                ← MODIFY: +fastapi, uvicorn, pydantic, httpx, locust
├── src/credit_risk/
│   ├── train.py                                  ← MODIFY: add signature + input_example to log_model
│   ├── cli.py                                    ← MODIFY: pass signature through train command
│   └── serving/                                  ← NEW directory
│       ├── __init__.py                           ← NEW
│       ├── schemas.py                            ← NEW: Pydantic request/response
│       ├── model_loader.py                       ← NEW: lifespan-managed model + columns
│       └── app.py                                ← NEW: FastAPI app, 3 endpoints
├── tests/
│   ├── test_serving_schemas.py                   ← NEW: Pydantic validation
│   ├── test_serving_app.py                       ← NEW: TestClient unit tests (mocked model)
│   └── test_serving_smoke.py                     ← NEW: live model smoke (skips if MLflow down)
├── serving/
│   ├── Dockerfile                                ← NEW: multi-stage
│   ├── .dockerignore                             ← NEW
│   └── locustfile.py                             ← NEW: load test scenario
├── examples/
│   └── encode_and_call.py                        ← NEW: helper that encodes raw → calls /predict
└── README.md                                     ← MODIFY: add Phase 2 section

Repo-root changes:
├── Makefile                                      ← MODIFY: +p2-serve, +p2-docker-build, +p2-docker-run, +p2-load-test, +p2-test
├── README.md                                     ← MODIFY: Project 1 status → ✅ Phase 2
├── STATUS.md                                     ← MODIFY: Phase 2 checkbox
└── .github/workflows/
    ├── ci.yml                                    ← MODIFY: include project-1 serving tests
    └── docker-publish.yml                        ← NEW: build + push to GHCR on main
```

---

## Daily Plan At-A-Glance

| Day | 2-hr session focus | Tasks |
|-----|--------------------|-------|
| Day 1 | Deps + Pydantic schemas + model signature retrain | T1-T4 |
| Day 2 | Model loader + FastAPI app + tests | T5-T8 |
| Day 3 | Dockerfile + local container smoke | T9-T11 |
| Day 4 | Locust load test + GHCR publish workflow | T12-T14 |
| Day 5 | End-to-end run + docs + push + CI green | T15-T17 |

---

# DAY 1 — Dependencies + Pydantic Schemas + Model Signature

## Task 1: Add Phase 2 dependencies

**Files:**
- Modify: `project-1-credit-risk-pipeline/pyproject.toml`

- [ ] **Step 1: Add the new deps**

Open `project-1-credit-risk-pipeline/pyproject.toml`. Find the `dependencies` block and add the four new lines (keep existing entries):

```toml
dependencies = [
  "xgboost>=2.1",
  "scikit-learn>=1.5",
  "pandas>=2.2",
  "numpy>=2.0",
  "mlflow>=2.16,<3.0",
  "boto3>=1.34",
  "dvc[s3]>=3.55",
  "pyarrow>=17",
  "typer>=0.12",
  "fastapi>=0.115",
  "uvicorn[standard]>=0.32",
  "pydantic>=2.9",
  "httpx>=0.27",
]
```

And add `locust>=2.31` to the `[project.optional-dependencies] dev` block:

```toml
[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-cov>=5.0",
  "ruff>=0.6",
  "locust>=2.31",
]
```

- [ ] **Step 2: Re-install the workspace package**

```bash
cd /home/himanshu/learning/mlops-journey
uv pip install -e "./project-1-credit-risk-pipeline[dev]"
```

Expected last lines mention installing `fastapi`, `uvicorn`, `pydantic`, `httpx`, `locust`.

- [ ] **Step 3: Verify imports**

```bash
.venv/bin/python -c "import fastapi, uvicorn, pydantic, httpx, locust; print('OK')"
```

Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add project-1-credit-risk-pipeline/pyproject.toml
git commit -m "chore(project-1): add Phase 2 deps (fastapi, uvicorn, pydantic, httpx, locust)"
```

---

## Task 2: Modify `train.py` to log a model signature

**Files:**
- Modify: `project-1-credit-risk-pipeline/src/credit_risk/train.py`
- Modify: `project-1-credit-risk-pipeline/src/credit_risk/cli.py`
- Modify: `project-1-credit-risk-pipeline/tests/test_train.py`

**Why:** Phase 1 logged the model without a signature — MLflow warned us, we ignored it. Signature = declared input column schema + output dtype. Phase 2 serving must enforce this (Pydantic schema is derived from the signature's input feature space).

- [ ] **Step 1: Add a failing test for signature presence**

Open `project-1-credit-risk-pipeline/tests/test_train.py` and append:

```python
def test_train_result_exposes_signature_inputs(synthetic_split):
    result = train_xgboost(synthetic_split, n_estimators=20, max_depth=3, random_state=0)
    assert result.feature_names is not None
    assert len(result.feature_names) == synthetic_split.X_train.shape[1]
    assert all(isinstance(c, str) for c in result.feature_names)
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/himanshu/learning/mlops-journey
.venv/bin/pytest project-1-credit-risk-pipeline/tests/test_train.py -v -k signature
```

Expected: FAIL — `AttributeError: 'TrainingResult' object has no attribute 'feature_names'`.

- [ ] **Step 3: Update `train.py` — add `feature_names` to `TrainingResult`**

Open `project-1-credit-risk-pipeline/src/credit_risk/train.py`. In the `TrainingResult` dataclass, add a `feature_names` field after `params`:

```python
@dataclass(frozen=True)
class TrainingResult:
    """Outcome of one XGBoost training run."""

    model: XGBClassifier
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    n_train: int
    n_test: int
    params: dict[str, Any] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
```

In `train_xgboost(...)`, after `model.fit(...)` and before the `return`, capture column names:

```python
    feature_names = list(split.X_train.columns)
```

And add the field to the `return TrainingResult(...)`:

```python
    return TrainingResult(
        model=model,
        accuracy=accuracy_score(split.y_test, y_pred),
        precision=precision_score(split.y_test, y_pred, zero_division=0),
        recall=recall_score(split.y_test, y_pred, zero_division=0),
        f1=f1_score(split.y_test, y_pred, zero_division=0),
        roc_auc=roc_auc_score(split.y_test, y_proba),
        n_train=len(split.X_train),
        n_test=len(split.X_test),
        params=params,
        feature_names=feature_names,
    )
```

- [ ] **Step 4: Confirm signature test passes**

```bash
.venv/bin/pytest project-1-credit-risk-pipeline/tests/test_train.py -v
```

Expected: `6 passed` (5 original + 1 new).

- [ ] **Step 5: Update `cli.py` train command to log signature + input example**

Open `project-1-credit-risk-pipeline/src/credit_risk/cli.py`. Find the existing import of `mlflow.sklearn` and add an import of `infer_signature`:

```python
from mlflow.models import infer_signature
```

In the `train(...)` function, inside the `for i, hp in ...` loop, replace this line:

```python
mlflow.sklearn.log_model(result.model, artifact_path="model")
```

with:

```python
signature = infer_signature(split.X_train, result.model.predict(split.X_train))
mlflow.sklearn.log_model(
    result.model,
    artifact_path="model",
    signature=signature,
    input_example=split.X_train.head(2),
)
```

- [ ] **Step 6: Smoke-run the CLI to confirm no syntax errors**

```bash
.venv/bin/python -m credit_risk.cli train --help
```

Expected: shows the train command help text (no traceback).

- [ ] **Step 7: Commit**

```bash
git add project-1-credit-risk-pipeline/src/credit_risk/train.py \
        project-1-credit-risk-pipeline/src/credit_risk/cli.py \
        project-1-credit-risk-pipeline/tests/test_train.py
git commit -m "feat(project-1): log model signature + input example + expose feature_names"
```

---

## Task 3: Re-run sweep + register v6 with signature

**Files:** none new.

**Why:** v5 has no signature. Phase 2 serving derives the request schema from the signature, so we need a fresh registered version that includes one.

- [ ] **Step 1: Make sure stack is up**

```bash
cd /home/himanshu/learning/mlops-journey
make verify
```

Expected: 18 / 18.

- [ ] **Step 2: Re-train (3 new trials, all with signatures this time)**

```bash
make p1-train
```

Expected: 3 `trial-N: roc_auc=...` lines, **no** "Model logged without a signature" warning.

- [ ] **Step 3: Register best run as v6**

```bash
make p1-register
```

Expected: `Registered 'credit-risk-classifier' v6 at stage 'Staging'.`

- [ ] **Step 4: Confirm via API**

```bash
curl -s "http://localhost:5000/api/2.0/mlflow/registered-models/get?name=credit-risk-classifier" \
  | python3 -m json.tool | head -40
```

Expected: `latest_versions` array shows a version with `current_stage="Staging"` and `version="6"` (or higher if you've re-run).

- [ ] **Step 5: No commit (only artifacts in MLflow changed, no code).**

---

## Task 4: TDD `serving/schemas.py` — Pydantic request/response

**Files:**
- Create: `project-1-credit-risk-pipeline/src/credit_risk/serving/__init__.py`
- Create: `project-1-credit-risk-pipeline/src/credit_risk/serving/schemas.py`
- Create: `project-1-credit-risk-pipeline/tests/test_serving_schemas.py`

- [ ] **Step 1: Create the serving subpackage init**

```bash
mkdir -p project-1-credit-risk-pipeline/src/credit_risk/serving
echo '"""Serving layer: FastAPI app, schemas, model loader."""' \
  > project-1-credit-risk-pipeline/src/credit_risk/serving/__init__.py
```

- [ ] **Step 2: Write the failing schema tests**

Create `project-1-credit-risk-pipeline/tests/test_serving_schemas.py`:

```python
"""Tests for Pydantic request/response schemas at the serving boundary."""

import pytest
from pydantic import ValidationError

from credit_risk.serving.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
)


def test_request_accepts_feature_dict():
    req = PredictionRequest(features={"duration": 12.0, "credit_amount": 1000.0})
    assert req.features["duration"] == 12.0


def test_request_rejects_empty_features():
    with pytest.raises(ValidationError):
        PredictionRequest(features={})


def test_request_accepts_optional_request_id():
    req = PredictionRequest(features={"x": 1.0}, request_id="abc-123")
    assert req.request_id == "abc-123"


def test_request_request_id_defaults_to_none():
    req = PredictionRequest(features={"x": 1.0})
    assert req.request_id is None


def test_response_default_class_is_int():
    resp = PredictionResponse(
        prediction=1, probability_default=0.83, model_version="6", request_id="abc"
    )
    assert resp.prediction == 1
    assert isinstance(resp.prediction, int)


def test_response_probability_must_be_unit_interval():
    with pytest.raises(ValidationError):
        PredictionResponse(
            prediction=1, probability_default=1.5, model_version="6", request_id="abc"
        )


def test_response_prediction_must_be_zero_or_one():
    with pytest.raises(ValidationError):
        PredictionResponse(
            prediction=2, probability_default=0.5, model_version="6", request_id="abc"
        )


def test_health_response_status_must_be_ok_or_degraded():
    HealthResponse(status="ok", model_loaded=True)
    HealthResponse(status="degraded", model_loaded=False)
    with pytest.raises(ValidationError):
        HealthResponse(status="weird", model_loaded=True)
```

- [ ] **Step 3: Run, confirm failure**

```bash
.venv/bin/pytest project-1-credit-risk-pipeline/tests/test_serving_schemas.py -v
```

Expected: `ImportError: cannot import name 'PredictionRequest' from 'credit_risk.serving.schemas'`.

- [ ] **Step 4: Write minimal `schemas.py`**

Create `project-1-credit-risk-pipeline/src/credit_risk/serving/schemas.py`:

```python
"""Pydantic v2 schemas for the credit-risk prediction API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """One prediction request — already-encoded features only.

    `features` is a dict of one-hot-encoded feature_name → float, matching
    the model's training-time column order. Use `examples/encode_and_call.py`
    to convert raw German Credit rows into this shape.
    """

    features: dict[str, float] = Field(min_length=1)
    request_id: str | None = None


class PredictionResponse(BaseModel):
    """One prediction result, with model lineage for observability."""

    prediction: Literal[0, 1] = Field(description="0=non-default, 1=default")
    probability_default: float = Field(ge=0.0, le=1.0)
    model_version: str
    request_id: str | None = None


class HealthResponse(BaseModel):
    """Health endpoint response."""

    status: Literal["ok", "degraded"]
    model_loaded: bool

    @field_validator("status", mode="before")
    @classmethod
    def _normalize(cls, v: str) -> str:
        return v.lower() if isinstance(v, str) else v
```

- [ ] **Step 5: Run, confirm 8 pass**

```bash
.venv/bin/pytest project-1-credit-risk-pipeline/tests/test_serving_schemas.py -v
```

Expected: `8 passed`.

- [ ] **Step 6: Commit**

```bash
git add project-1-credit-risk-pipeline/src/credit_risk/serving/__init__.py \
        project-1-credit-risk-pipeline/src/credit_risk/serving/schemas.py \
        project-1-credit-risk-pipeline/tests/test_serving_schemas.py
git commit -m "feat(project-1): serving — Pydantic v2 request/response schemas"
```

- [ ] **Step 7: End-of-Day-1 — run full test suite + push**

```bash
.venv/bin/pytest project-1-credit-risk-pipeline -v
git push
```

Expected: `25 passed` (4 data_loader + 6 features + 6 train + 2 registry + 8 schemas + 1 from earlier — total may vary by 1; key signal is no failures).

---

# DAY 2 — Model Loader + FastAPI App + Tests

## Task 5: TDD `serving/model_loader.py`

**Files:**
- Create: `project-1-credit-risk-pipeline/src/credit_risk/serving/model_loader.py`

- [ ] **Step 1: Write the (offline-friendly) tests**

Create `project-1-credit-risk-pipeline/tests/test_serving_smoke.py`:

```python
"""Live smoke tests for the model loader. Skipped if MLflow is down."""

import os
import urllib.request

import pytest

from credit_risk.serving.model_loader import load_staging_model


@pytest.fixture(scope="module")
def mlflow_up() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    try:
        urllib.request.urlopen(uri + "/health", timeout=2)
    except Exception:
        pytest.skip(f"MLflow not reachable at {uri} — run `make up` first")
    return uri


def test_load_staging_returns_bundle(mlflow_up):
    bundle = load_staging_model()
    assert bundle.model is not None
    assert bundle.version >= 1
    assert len(bundle.feature_names) > 0


def test_loaded_model_can_predict(mlflow_up):
    import numpy as np
    bundle = load_staging_model()
    X = np.zeros((1, len(bundle.feature_names)))
    pred = bundle.model.predict(X)
    assert pred.shape == (1,)
    assert pred[0] in (0, 1)
```

- [ ] **Step 2: Run to confirm failure (import error)**

```bash
.venv/bin/pytest project-1-credit-risk-pipeline/tests/test_serving_smoke.py -v
```

Expected: `ImportError: cannot import name 'load_staging_model'`.

- [ ] **Step 3: Write `model_loader.py`**

Create `project-1-credit-risk-pipeline/src/credit_risk/serving/model_loader.py`:

```python
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
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(uri)

    client = MlflowClient()
    versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=[STAGE])
    if not versions:
        raise RuntimeError(
            f"No '{STAGE}'-stage version found for model '{REGISTERED_MODEL_NAME}'."
        )
    mv = versions[0]

    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{STAGE}")

    # Pull feature names from the signature stored on the model artifact.
    feature_names: list[str] = []
    try:
        info = client.get_model_version(name=mv.name, version=mv.version)
        run_id = info.run_id
        if run_id:
            run = client.get_run(run_id)
            artifact_uri = f"{run.info.artifact_uri}/model"
            mlflow_model = mlflow.models.Model.load(artifact_uri)
            sig = mlflow_model.signature
            if sig and sig.inputs:
                feature_names = [c.name for c in sig.inputs.inputs if c.name]
    except Exception:
        # Fall back to model.feature_names_in_ if MLflow signature is missing
        feature_names = list(getattr(model, "feature_names_in_", []) or [])

    return ModelBundle(model=model, version=int(mv.version), feature_names=feature_names)
```

- [ ] **Step 4: Run smoke tests — needs stack up**

```bash
make verify
.venv/bin/pytest project-1-credit-risk-pipeline/tests/test_serving_smoke.py -v
```

Expected: `2 passed`. If you re-trained in Task 3, `feature_names` will have ~60 columns (the encoded German Credit feature space).

- [ ] **Step 5: Commit**

```bash
git add project-1-credit-risk-pipeline/src/credit_risk/serving/model_loader.py \
        project-1-credit-risk-pipeline/tests/test_serving_smoke.py
git commit -m "feat(project-1): serving — lifespan-managed model loader from Registry"
```

---

## Task 6: TDD `serving/app.py` — FastAPI app with 3 endpoints

**Files:**
- Create: `project-1-credit-risk-pipeline/src/credit_risk/serving/app.py`
- Create: `project-1-credit-risk-pipeline/tests/test_serving_app.py`

- [ ] **Step 1: Write the failing TestClient tests (offline, mocked bundle)**

Create `project-1-credit-risk-pipeline/tests/test_serving_app.py`:

```python
"""Unit tests for the FastAPI app — mocks the model bundle to stay offline."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from credit_risk.serving.app import app, get_bundle
from credit_risk.serving.model_loader import ModelBundle


class _StubModel:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        n = len(X)
        out = np.zeros((n, 2))
        out[:, 0] = 0.7
        out[:, 1] = 0.3
        return out


@pytest.fixture
def client():
    """Inject a stub model bundle so tests don't need MLflow."""
    stub = ModelBundle(
        model=_StubModel(),
        version=99,
        feature_names=["duration", "credit_amount"],
    )
    app.dependency_overrides[get_bundle] = lambda: stub
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_healthz_returns_ok_when_model_loaded(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_readyz_returns_200_when_model_loaded(client):
    r = client.get("/readyz")
    assert r.status_code == 200


def test_predict_returns_valid_response(client):
    r = client.post(
        "/predict",
        json={"features": {"duration": 12.0, "credit_amount": 1500.0}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability_default"] <= 1.0
    assert body["model_version"] == "99"


def test_predict_echoes_request_id(client):
    r = client.post(
        "/predict",
        json={"features": {"x": 1.0}, "request_id": "trace-abc"},
    )
    assert r.json()["request_id"] == "trace-abc"


def test_predict_rejects_empty_features(client):
    r = client.post("/predict", json={"features": {}})
    assert r.status_code == 422
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/pytest project-1-credit-risk-pipeline/tests/test_serving_app.py -v
```

Expected: `ImportError: cannot import name 'app' from 'credit_risk.serving.app'`.

- [ ] **Step 3: Write `app.py`**

Create `project-1-credit-risk-pipeline/src/credit_risk/serving/app.py`:

```python
"""FastAPI app — /predict, /healthz, /readyz endpoints."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Response

from credit_risk.serving.model_loader import ModelBundle, load_staging_model
from credit_risk.serving.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger("credit_risk.serving")

# Module-level slot the lifespan handler fills in. Tests can override via
# dependency_overrides on `get_bundle`.
_bundle: ModelBundle | None = None


def get_bundle() -> ModelBundle:
    if _bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bundle
    try:
        _bundle = load_staging_model()
        logger.info("Loaded model v%s with %d features", _bundle.version, len(_bundle.feature_names))
    except Exception as exc:
        logger.exception("Failed to load model on startup: %s", exc)
        _bundle = None
    yield
    _bundle = None


app = FastAPI(
    title="Credit Risk Prediction API",
    version="0.2.0",
    description="Predicts default probability from already-encoded German Credit features.",
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    loaded = _bundle is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
    )


@app.get("/readyz")
def readyz(response: Response) -> dict:
    if _bundle is None:
        response.status_code = 503
        return {"ready": False}
    return {"ready": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(
    req: PredictionRequest,
    bundle: Annotated[ModelBundle, Depends(get_bundle)],
) -> PredictionResponse:
    """Run inference. The request's `features` dict is reindexed to the
    model's training-time column order; missing columns default to 0."""
    if bundle.feature_names:
        row = pd.DataFrame([req.features]).reindex(
            columns=bundle.feature_names, fill_value=0.0
        )
    else:
        # Fall back to whatever order the request supplied (best-effort).
        row = pd.DataFrame([req.features])

    X = row.to_numpy(dtype=float)
    pred = int(bundle.model.predict(X)[0])
    proba_default = float(bundle.model.predict_proba(X)[0, 1])

    return PredictionResponse(
        prediction=pred,
        probability_default=proba_default,
        model_version=str(bundle.version),
        request_id=req.request_id,
    )
```

- [ ] **Step 4: Run app tests — all 5 should pass**

```bash
.venv/bin/pytest project-1-credit-risk-pipeline/tests/test_serving_app.py -v
```

Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add project-1-credit-risk-pipeline/src/credit_risk/serving/app.py \
        project-1-credit-risk-pipeline/tests/test_serving_app.py
git commit -m "feat(project-1): serving — FastAPI app with /predict /healthz /readyz"
```

---

## Task 7: Smoke-run the FastAPI app against the live stack

**Files:** none new.

- [ ] **Step 1: Start the server in foreground**

In one terminal:

```bash
cd /home/himanshu/learning/mlops-journey/project-1-credit-risk-pipeline
../.venv/bin/uvicorn credit_risk.serving.app:app --host 0.0.0.0 --port 8000
```

Expected: `INFO  Loaded model v6 with 61 features` (or similar — exact count depends on the dataset's category counts).

- [ ] **Step 2: From another terminal, curl `/healthz`**

```bash
curl -s http://localhost:8000/healthz | python3 -m json.tool
```

Expected:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

- [ ] **Step 3: curl `/readyz`**

```bash
curl -si http://localhost:8000/readyz | head -3
```

Expected: HTTP 200, body `{"ready":true}`.

- [ ] **Step 4: curl `/predict` with a zero-vector request**

```bash
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features":{"duration":12.0,"credit_amount":1500.0},"request_id":"smoke-1"}' \
  | python3 -m json.tool
```

Expected: response with `prediction`, `probability_default`, `model_version`, `request_id=smoke-1`.

- [ ] **Step 5: Stop the server** (Ctrl-C in the first terminal).

- [ ] **Step 6: No new commit — this was a smoke test.**

---

## Task 8: Helper script — `examples/encode_and_call.py`

**Files:**
- Create: `project-1-credit-risk-pipeline/examples/encode_and_call.py`

- [ ] **Step 1: Write the helper**

```bash
mkdir -p project-1-credit-risk-pipeline/examples
```

Create `project-1-credit-risk-pipeline/examples/encode_and_call.py`:

```python
"""Encode a raw German Credit row and POST it to the running API.

Usage:
    python examples/encode_and_call.py           # uses a built-in sample row
    python examples/encode_and_call.py http://localhost:8000

Run the API first (in another terminal):
    uvicorn credit_risk.serving.app:app --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pandas as pd

from credit_risk.data_loader import load_german_credit
from credit_risk.features import encode_features

DEFAULT_URL = "http://localhost:8000"
REPO_DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "german_credit.csv"


def main(url: str = DEFAULT_URL) -> None:
    df = load_german_credit(REPO_DATA)
    raw_row = df.drop(columns=["default"]).iloc[[0]]
    encoded = encode_features(raw_row)
    features = {k: float(v) for k, v in encoded.iloc[0].to_dict().items()}

    payload = {"features": features, "request_id": "example-1"}
    print(f"POST {url}/predict ({len(features)} features)")
    r = httpx.post(f"{url}/predict", json=payload, timeout=10.0)
    r.raise_for_status()
    print(r.json())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL)
```

- [ ] **Step 2: With the server running (from Task 7), test the helper**

```bash
cd /home/himanshu/learning/mlops-journey/project-1-credit-risk-pipeline
../.venv/bin/uvicorn credit_risk.serving.app:app --port 8000 &
sleep 3
../.venv/bin/python examples/encode_and_call.py
kill %1
```

Expected: prints a JSON response with `prediction`, `probability_default`, `model_version`.

- [ ] **Step 3: Commit**

```bash
git add project-1-credit-risk-pipeline/examples/encode_and_call.py
git commit -m "feat(project-1): example helper — raw row → encoded → /predict"
```

- [ ] **Step 4: End-of-Day-2 — push**

```bash
git push
```

---

# DAY 3 — Dockerization

## Task 9: Multi-stage Dockerfile

**Files:**
- Create: `project-1-credit-risk-pipeline/serving/Dockerfile`
- Create: `project-1-credit-risk-pipeline/serving/.dockerignore`

- [ ] **Step 1: Create the serving directory**

```bash
mkdir -p /home/himanshu/learning/mlops-journey/project-1-credit-risk-pipeline/serving
```

- [ ] **Step 2: Write the multi-stage Dockerfile**

Create `project-1-credit-risk-pipeline/serving/Dockerfile`:

```dockerfile
# syntax=docker/dockerfile:1.7

# ──────────── builder stage ────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Copy only what's needed for install, in layer-cache-friendly order
COPY pyproject.toml /build/pyproject.toml
COPY src /build/src

RUN pip install --upgrade pip && \
    pip install --prefix=/install \
      "fastapi>=0.115" "uvicorn[standard]>=0.32" "pydantic>=2.9" \
      "mlflow>=2.16,<3.0" "scikit-learn>=1.5" "xgboost>=2.1" \
      "pandas>=2.2" "numpy>=2.0" "boto3>=1.34" && \
    pip install --prefix=/install /build

# ──────────── runtime stage ────────────
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    PORT=8000

# Non-root user for security
RUN useradd -m -u 1001 -s /bin/bash app
WORKDIR /home/app

# Pull in installed deps + our package from the builder stage
COPY --from=builder /install /usr/local
USER app

EXPOSE 8000

# Lightweight healthcheck — hits our /healthz endpoint
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)" \
  || exit 1

CMD ["uvicorn", "credit_risk.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Write `.dockerignore`**

Create `project-1-credit-risk-pipeline/serving/.dockerignore`:

```
**/.git
**/.venv
**/__pycache__
**/*.pyc
**/.pytest_cache
**/.dvc
**/data/
**/tests/
**/.env
**/.env.*
**/Dockerfile
**/docker-compose*.yml
notebooks/
examples/
```

- [ ] **Step 4: Build the image**

```bash
cd /home/himanshu/learning/mlops-journey/project-1-credit-risk-pipeline
docker build -f serving/Dockerfile -t credit-risk-api:dev .
```

Expected: 2-3 minute build; final line `naming to docker.io/library/credit-risk-api:dev done`.

- [ ] **Step 5: Inspect image size**

```bash
docker images credit-risk-api
```

Expected: `SIZE` under 1.5 GB (XGBoost + scikit-learn are heavy; if you see >2 GB, the .dockerignore is missing files).

- [ ] **Step 6: Commit**

```bash
cd /home/himanshu/learning/mlops-journey
git add project-1-credit-risk-pipeline/serving/Dockerfile \
        project-1-credit-risk-pipeline/serving/.dockerignore
git commit -m "feat(project-1): multi-stage Dockerfile for serving image"
```

---

## Task 10: Run the container against the live MLflow stack

**Files:** none new.

- [ ] **Step 1: Make sure the docker-compose stack is up**

```bash
make up
sleep 5
make verify
```

Expected: 18 / 18.

- [ ] **Step 2: Run the container on the `infra_default` network**

```bash
docker run --rm -d --name credit-risk-api \
  --network infra_default \
  -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e AWS_ACCESS_KEY_ID=$(grep '^MINIO_ROOT_USER=' infra/.env | cut -d= -f2) \
  -e AWS_SECRET_ACCESS_KEY=$(grep '^MINIO_ROOT_PASSWORD=' infra/.env | cut -d= -f2) \
  -e MLFLOW_S3_ENDPOINT_URL=http://minio:9000 \
  credit-risk-api:dev
```

Expected: container id printed.

- [ ] **Step 3: Tail startup logs to confirm model loaded**

```bash
docker logs -f credit-risk-api &
sleep 8
kill %1 2>/dev/null
```

Expected: an `INFO Loaded model v6 with 61 features` line.

- [ ] **Step 4: Hit `/healthz` from the host**

```bash
curl -s http://localhost:8000/healthz | python3 -m json.tool
```

Expected: `{"status":"ok","model_loaded":true}`.

- [ ] **Step 5: Hit `/predict`**

```bash
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features":{"duration":12.0,"credit_amount":1500.0},"request_id":"docker-smoke"}' \
  | python3 -m json.tool
```

Expected: full prediction response.

- [ ] **Step 6: Inspect Docker's own healthcheck status**

```bash
docker inspect --format '{{.State.Health.Status}}' credit-risk-api
```

Expected: `healthy` (give it 30s if it shows `starting`).

- [ ] **Step 7: Stop the container**

```bash
docker stop credit-risk-api
```

- [ ] **Step 8: No new commit — smoke test only.**

---

## Task 11: End-of-Day-3 — run all tests + push

- [ ] **Step 1: Run full Project-1 test suite**

```bash
cd /home/himanshu/learning/mlops-journey
.venv/bin/pytest project-1-credit-risk-pipeline -v
```

Expected: all tests pass (data_loader + features + train + registry + schemas + app + smoke — roughly 26+).

- [ ] **Step 2: Push**

```bash
git push
```

---

# DAY 4 — Locust Load Test + GHCR Publish Workflow

## Task 12: Locust load test

**Files:**
- Create: `project-1-credit-risk-pipeline/serving/locustfile.py`

- [ ] **Step 1: Write the locustfile**

Create `project-1-credit-risk-pipeline/serving/locustfile.py`:

```python
"""Locust load test for the credit-risk API.

Run (with the API on :8000):
    locust -f serving/locustfile.py --headless \\
        --users 20 --spawn-rate 5 --run-time 60s \\
        --host http://localhost:8000
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

# Tiny synthetic feature dict — encoded space.
# In a real scenario you'd record from real traffic and replay.
SAMPLE_FEATURES = {
    "duration": 12.0,
    "credit_amount": 1500.0,
    "installment_commitment": 4.0,
    "residence_since": 4.0,
    "age": 35.0,
    "existing_credits": 1.0,
    "num_dependents": 1.0,
}


class CreditRiskUser(HttpUser):
    """Simulates a client hitting /predict with jittered features."""

    wait_time = between(0.1, 0.5)

    @task(10)
    def predict(self):
        features = {k: v * random.uniform(0.8, 1.2) for k, v in SAMPLE_FEATURES.items()}
        self.client.post(
            "/predict",
            json={"features": features, "request_id": f"locust-{self._user_id}"},
            timeout=5.0,
        )

    @task(1)
    def healthz(self):
        self.client.get("/healthz")
```

- [ ] **Step 2: Run the API in one terminal**

```bash
cd /home/himanshu/learning/mlops-journey/project-1-credit-risk-pipeline
../.venv/bin/uvicorn credit_risk.serving.app:app --port 8000 &
sleep 3
```

- [ ] **Step 3: Run locust for 30 seconds, headless**

```bash
../.venv/bin/locust -f serving/locustfile.py --headless \
  --users 10 --spawn-rate 2 --run-time 30s \
  --host http://localhost:8000
```

Expected: a stats table at the end. **p95 latency should be < 100ms** on this laptop. RPS will be modest (single-process uvicorn). 0 failures.

- [ ] **Step 4: Stop the API**

```bash
kill %1
```

- [ ] **Step 5: Commit**

```bash
cd /home/himanshu/learning/mlops-journey
git add project-1-credit-risk-pipeline/serving/locustfile.py
git commit -m "feat(project-1): Locust load test for /predict (10 users / 30s baseline)"
```

---

## Task 13: GHCR publish workflow

**Files:**
- Create: `.github/workflows/docker-publish.yml`

**Why:** Every push to `main` builds the serving image and pushes it to GitHub Container Registry under `ghcr.io/himanshunigam-456/credit-risk-api:<sha>` and `:latest`. The image is then runnable from anywhere.

- [ ] **Step 1: Write the workflow**

Create `.github/workflows/docker-publish.yml`:

```yaml
name: Build & Publish Docker Image (Project 1)

on:
  push:
    branches: [main]
    paths:
      - 'project-1-credit-risk-pipeline/**'
      - '.github/workflows/docker-publish.yml'
  workflow_dispatch:

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository_owner }}/credit-risk-api

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha,prefix=
            type=ref,event=branch
            type=raw,value=latest,enable={{is_default_branch}}

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: project-1-credit-risk-pipeline
          file: project-1-credit-risk-pipeline/serving/Dockerfile
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

- [ ] **Step 2: Commit (do NOT push yet — Day 5 push triggers this)**

```bash
git add .github/workflows/docker-publish.yml
git commit -m "ci(project-1): GHCR publish workflow for the serving image"
```

---

## Task 14: Phase 2 Makefile shortcuts

**Files:**
- Modify: `Makefile` (repo root)

- [ ] **Step 1: Append the project-2 targets**

Add to the END of `/home/himanshu/learning/mlops-journey/Makefile`:

```makefile
# ── Project 1 Phase 2 — Serving ──

p2-serve:  ## Run the FastAPI app locally on :8000 against live MLflow
	cd project-1-credit-risk-pipeline && \
	  ../.venv/bin/uvicorn credit_risk.serving.app:app --host 0.0.0.0 --port 8000 --reload

p2-docker-build:  ## Build the credit-risk-api Docker image
	cd project-1-credit-risk-pipeline && \
	  docker build -f serving/Dockerfile -t credit-risk-api:dev .

p2-docker-run:  ## Run the container against the compose stack
	docker run --rm -d --name credit-risk-api \
	  --network infra_default \
	  -p 8000:8000 \
	  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
	  -e AWS_ACCESS_KEY_ID=$$(grep '^MINIO_ROOT_USER=' infra/.env | cut -d= -f2) \
	  -e AWS_SECRET_ACCESS_KEY=$$(grep '^MINIO_ROOT_PASSWORD=' infra/.env | cut -d= -f2) \
	  -e MLFLOW_S3_ENDPOINT_URL=http://minio:9000 \
	  credit-risk-api:dev
	@echo "Container started — http://localhost:8000/docs"

p2-docker-stop:  ## Stop the credit-risk-api container
	-docker stop credit-risk-api

p2-load-test:  ## Run a 30s Locust load test against http://localhost:8000
	cd project-1-credit-risk-pipeline && \
	  ../.venv/bin/locust -f serving/locustfile.py --headless \
	    --users 10 --spawn-rate 2 --run-time 30s \
	    --host http://localhost:8000

p2-test:  ## Run only the Phase 2 (serving) tests
	.venv/bin/pytest project-1-credit-risk-pipeline/tests/test_serving_schemas.py \
	                 project-1-credit-risk-pipeline/tests/test_serving_app.py \
	                 project-1-credit-risk-pipeline/tests/test_serving_smoke.py -v
```

- [ ] **Step 2: Verify the targets show up**

```bash
make help | grep p2-
```

Expected: 6 `p2-*` lines listed.

- [ ] **Step 3: Smoke-run `make p2-test`**

```bash
make p2-test
```

Expected: ~15 tests pass (8 schemas + 5 app + 2 smoke).

- [ ] **Step 4: Commit + push**

```bash
git add Makefile
git commit -m "chore: add Makefile p2- shortcuts (serve/docker-build/run/load-test/test)"
git push
```

---

# DAY 5 — End-to-End Run + Docs + Phase 2 Close

## Task 15: Full end-to-end smoke test

- [ ] **Step 1: Cold-restart the stack**

```bash
make down && make up
sleep 30
make verify
```

Expected: 18 / 18.

- [ ] **Step 2: Re-build the Docker image (fresh layer cache after restart)**

```bash
make p2-docker-build
```

Expected: build completes; no errors.

- [ ] **Step 3: Run the container**

```bash
make p2-docker-run
sleep 10
docker inspect --format '{{.State.Health.Status}}' credit-risk-api
```

Expected: `healthy`.

- [ ] **Step 4: Hit `/healthz` + `/predict`**

```bash
curl -s http://localhost:8000/healthz | python3 -m json.tool
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features":{"duration":12.0,"credit_amount":1500.0,"age":35.0},"request_id":"e2e"}' \
  | python3 -m json.tool
```

Expected: both succeed; predict returns a full response with `model_version=6`.

- [ ] **Step 5: Run the example helper to confirm encoding round-trip works**

```bash
cd project-1-credit-risk-pipeline
../.venv/bin/python examples/encode_and_call.py
cd ..
```

Expected: successful JSON response printed.

- [ ] **Step 6: Run the load test**

```bash
make p2-load-test
```

Expected: stats table at end. 0 failures. Capture the **p95 latency** number — you'll quote it in the README.

- [ ] **Step 7: Stop the container**

```bash
make p2-docker-stop
```

---

## Task 16: Update docs (README + STATUS)

**Files:**
- Modify: `README.md` (repo root)
- Modify: `STATUS.md`
- Modify: `project-1-credit-risk-pipeline/README.md`

- [ ] **Step 1: Update root `README.md` projects table**

Find and replace:

```markdown
| 1 | Self-Healing Credit-Risk Pipeline | Fintech | ✅ Phase 1 |
```

with:

```markdown
| 1 | Self-Healing Credit-Risk Pipeline | Fintech | ✅ Phase 2 |
```

And the bottom line:

```markdown
*Currently executing Week 2 of 26 — Project 1 Phase 1 complete. See [`STATUS.md`](STATUS.md) for live progress.*
```

with:

```markdown
*Currently executing Week 3 of 26 — Project 1 Phase 2 complete. See [`STATUS.md`](STATUS.md) for live progress.*
```

- [ ] **Step 2: Update `STATUS.md`**

Replace the `## Current` block:

```markdown
## Current

- **Week:** 3 of 26
- **Project:** project-1-credit-risk-pipeline — Phase 2 ✅ shipped
- **Infrastructure:** 18 / 18 verify checks passing
- **Model Registry:** `credit-risk-classifier` v6 @ `Staging` (with model signature)
- **Serving:** FastAPI on :8000, dockerized to `ghcr.io/himanshunigam-456/credit-risk-api:latest`
- **Load test p95:** <FILL IN from Task 15 Step 6>
- **Tests:** 30+ passing (schemas, app, smoke, full Phase 1 suite)
- **Blockers:** none
```

In the Phases checklist, tick Phase 2:

```markdown
- [x] Phase 1 (Week 2) — Data + XGBoost + MLflow Model Registry
- [x] Phase 2 (Week 3) — FastAPI serving + Dockerization
- [ ] Phase 3 (Week 4) — Evidently AI drift detection + auto-retrain
- [ ] Phase 4 (Week 5) — Canary deploy + Prometheus monitoring
- [ ] Phase 5 (Week 6) — AWS EC2 production demo + portfolio polish
```

- [ ] **Step 3: Update `project-1-credit-risk-pipeline/README.md`**

Add a new section under `## Phase 1`:

```markdown
## Phase 2 (Week 3): FastAPI Serving + Dockerization ✅

- Pydantic-validated `/predict` endpoint
- Model loaded once at startup via FastAPI `lifespan` from `models:/credit-risk-classifier/Staging`
- Multi-stage Dockerfile → image at `ghcr.io/himanshunigam-456/credit-risk-api`
- Locust load test (p95 < 100ms locally on 10 users)
- GitHub Actions auto-publishes to GHCR on push to main

### Run

```bash
make p2-serve            # local uvicorn
make p2-docker-build     # build image
make p2-docker-run       # run container against compose stack
make p2-load-test        # 30s Locust scenario
```
```

- [ ] **Step 4: Commit**

```bash
git add README.md STATUS.md project-1-credit-risk-pipeline/README.md
git commit -m "docs: Project 1 Phase 2 complete — FastAPI service shipped to GHCR"
```

---

## Task 17: Push + verify GHCR image published

- [ ] **Step 1: Push**

```bash
git push
```

This triggers BOTH workflows: `ci.yml` and `docker-publish.yml`.

- [ ] **Step 2: Watch the Actions runs**

```bash
gh run list --limit 3
```

Expected: 2-3 workflow runs in flight. Wait ~5 minutes.

- [ ] **Step 3: Watch the docker-publish workflow live**

```bash
gh run watch
```

Pick the `Build & Publish Docker Image` run. Expected: ends in green.

- [ ] **Step 4: Confirm image landed in GHCR**

```bash
gh api /users/himanshunigam-456/packages/container/credit-risk-api/versions --jq '.[].metadata.container.tags' | head -20
```

Expected: a list of tags including `latest` and the latest commit SHA.

- [ ] **Step 5: Verify the public image can be pulled (no auth)**

```bash
docker pull ghcr.io/himanshunigam-456/credit-risk-api:latest
docker images ghcr.io/himanshunigam-456/credit-risk-api
```

Expected: image pulls successfully.

If pull asks for auth, the package is private by default — make it public:

```bash
gh api -X PATCH /user/packages/container/credit-risk-api --field visibility=public
```

Then retry the pull.

---

# Phase 2 Complete

**You shipped:**
- FastAPI service with Pydantic-validated `/predict`, `/healthz`, `/readyz`
- Model signature now baked into v6 (closing the Phase 1 warning loop)
- Multi-stage Docker image, slim (<1.5 GB), non-root user, healthcheck
- GHCR auto-publish on every push to `main`
- Locust load test scenario + Makefile shortcut
- 15+ new tests (Pydantic, TestClient, live smoke)
- 5 new `make p2-*` targets

**Next phase preview (Phase 3 / Week 4 — Drift Detection):**
- Evidently AI detects feature & target drift between training data and production traffic
- Drift report served at `/drift` endpoint and exported as HTML
- GitHub Actions cron retrains the model nightly if drift is detected
- Model registry gets new versions automatically; canary promotion logic
- Project status auto-updates: `Staging → Production` when retrained model beats current

Tell Claude: *"plan project 1 phase 3"* to begin.
