# Project 1 · Phase 1 (Week 2) — Data + XGBoost + Model Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the Week 1 warmup into a real, version-controlled training pipeline: DVC-tracked dataset, XGBoost trainer (replaces RandomForest), MLflow Model Registry with explicit stage transitions, and a hyperparameter sweep that picks the best model. End artifact: a `credit-risk-classifier` model in the Registry at the `Staging` stage with full lineage.

**Architecture:** New monorepo package `project-1-credit-risk-pipeline/` that depends on (and improves) `project-0-warmup/`. Data flows: raw CSV in `data/raw/` (DVC-tracked, pushed to MinIO via S3 remote) → feature engineering → XGBoost training with 3-trial hyperparameter sweep → best model registered to MLflow with metadata. Re-uses the running Docker-Compose stack from Week 1 (MLflow + MinIO + Postgres).

**Tech Stack:** XGBoost · DVC[s3] · MLflow Model Registry · pytest · uv workspace · MinIO (S3 remote for DVC) · scikit-learn (preprocessing only)

**Time budget:** 10 hrs across 5 weekday sessions (2 hrs/day).

**Prerequisites (verified before Day 1 starts):**
- Week 1 verify.sh shows 18/18 ✅
- `infra/.env` exists with MinIO credentials
- Stack is up (`make up`)
- Existing run from Week 1 visible in MLflow at experiment `credit-baseline`

---

## File Structure Created This Phase

```
project-1-credit-risk-pipeline/
├── pyproject.toml                                ← project deps + tool config
├── README.md                                     ← project-local landing page
├── dvc.yaml                                      ← DVC pipeline definition
├── .dvc/                                         ← DVC repo metadata (auto-created)
├── data/
│   ├── raw/                                      ← DVC-tracked (gitignored)
│   │   └── german_credit.csv
│   └── README.md                                 ← documents the dataset
├── src/credit_risk/
│   ├── __init__.py
│   ├── data_loader.py                            ← loads CSV → DataFrame
│   ├── features.py                               ← preprocessing + one-hot
│   ├── train.py                                  ← XGBoost training function
│   ├── registry.py                               ← MLflow Model Registry helpers
│   └── cli.py                                    ← `python -m credit_risk.cli`
└── tests/
    ├── __init__.py
    ├── test_data_loader.py
    ├── test_features.py
    ├── test_train.py
    └── test_registry.py

Repo-root changes:
├── pyproject.toml                                ← add new workspace member
├── Makefile                                      ← add `train` + `register` targets
└── .gitignore                                    ← add /data/raw/ and DVC patterns
```

---

## Daily Plan At-A-Glance

| Day | 2-hr session focus | Tasks |
|-----|--------------------|-------|
| Day 1 | Project skeleton + DVC setup + push data to MinIO | T1-T5 |
| Day 2 | data_loader.py + features.py (TDD) | T6-T9 |
| Day 3 | train.py with XGBoost + tests | T10-T13 |
| Day 4 | MLflow integration + registry.py + cli.py | T14-T18 |
| Day 5 | Run end-to-end, polish, commit, push | T19-T22 |

---

# DAY 1 — Project Skeleton + DVC Setup

## Task 1: Scaffold the `project-1-credit-risk-pipeline` package

**Files:**
- Create: `project-1-credit-risk-pipeline/pyproject.toml`
- Create: `project-1-credit-risk-pipeline/README.md`
- Create: `project-1-credit-risk-pipeline/src/credit_risk/__init__.py`
- Create: `project-1-credit-risk-pipeline/tests/__init__.py`

- [ ] **Step 1: Create directory tree**

```bash
cd /home/himanshu/learning/mlops-journey
mkdir -p project-1-credit-risk-pipeline/src/credit_risk
mkdir -p project-1-credit-risk-pipeline/tests
mkdir -p project-1-credit-risk-pipeline/data/raw
```

- [ ] **Step 2: Write `project-1-credit-risk-pipeline/pyproject.toml`**

```toml
[project]
name = "credit-risk-pipeline"
version = "0.1.0"
description = "Project 1: Self-healing credit-risk ML pipeline (fintech domain)."
requires-python = ">=3.11"
dependencies = [
  "xgboost>=2.1",
  "scikit-learn>=1.5",
  "pandas>=2.2",
  "numpy>=2.0",
  "mlflow>=2.16,<3.0",
  "boto3>=1.34",
  "dvc[s3]>=3.55",
  "pyarrow>=17",                # fast parquet I/O for DVC
  "typer>=0.12",                # CLI framework
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-cov>=5.0",
  "ruff>=0.6",
]

[project.scripts]
credit-risk = "credit_risk.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/credit_risk"]
```

- [ ] **Step 3: Write `project-1-credit-risk-pipeline/README.md`**

```markdown
# project-1-credit-risk-pipeline

**Domain:** Fintech · **Phase:** Week 2-6 of the MLOps journey.

A self-healing credit-risk classification pipeline. Trains an XGBoost model
on the UCI German Credit dataset, registers the best model in MLflow,
detects drift in production traffic, and auto-retrains via GitHub Actions.

## Phase 1 (Week 2): Data + XGBoost + Registry

- DVC-tracked raw data in MinIO
- XGBoost trainer with hyperparameter sweep
- Best model promoted to MLflow Model Registry @ `Staging`

## Run

```bash
make train        # train + log to MLflow
make register     # promote best run to Model Registry
```

## Phases (planned)

- Phase 2 (Week 3): FastAPI serving + Dockerization
- Phase 3 (Week 4): Evidently AI drift detection + auto-retrain
- Phase 4 (Week 5): Canary deploy + Prometheus monitoring
- Phase 5 (Week 6): AWS EC2 production demo + portfolio polish
```

- [ ] **Step 4: Create empty `__init__.py` files**

```bash
echo '"""Credit-risk pipeline package."""' > project-1-credit-risk-pipeline/src/credit_risk/__init__.py
touch project-1-credit-risk-pipeline/tests/__init__.py
```

- [ ] **Step 5: Register new workspace member in root `pyproject.toml`**

The root `pyproject.toml` already lists `project-1-credit-risk-pipeline` in `[tool.uv.workspace]` members. No edit needed. Verify:

```bash
grep "project-1-credit-risk-pipeline" /home/himanshu/learning/mlops-journey/pyproject.toml
```

Expected: shows the line listing it as a member.

- [ ] **Step 6: Install the new package into the venv**

```bash
cd /home/himanshu/learning/mlops-journey
uv pip install -e "./project-1-credit-risk-pipeline[dev]"
```

Expected last lines: a list of installed packages including `xgboost`, `dvc`, `typer`.

- [ ] **Step 7: Verify imports work**

```bash
.venv/bin/python -c "import credit_risk; import xgboost; import dvc; print('OK')"
```

Expected: `OK`.

- [ ] **Step 8: Commit**

```bash
git add project-1-credit-risk-pipeline/
git commit -m "feat(project-1): scaffold credit-risk-pipeline package"
```

---

## Task 2: Initialize DVC + configure MinIO as the S3 remote

**Files:**
- Create: `project-1-credit-risk-pipeline/.dvc/config` (via `dvc init`)
- Modify: `project-1-credit-risk-pipeline/.gitignore`

- [ ] **Step 1: Initialize DVC inside the project package (subdir DVC)**

```bash
cd /home/himanshu/learning/mlops-journey/project-1-credit-risk-pipeline
.venv/bin/dvc init --subdir
```

Expected: creates `.dvc/config` and `.dvc/.gitignore`. Reports "Initialized DVC repository."

- [ ] **Step 2: Configure MinIO as the DVC remote named `minio`**

```bash
cd /home/himanshu/learning/mlops-journey/project-1-credit-risk-pipeline
.venv/bin/dvc remote add -d minio s3://datasets/project-1
.venv/bin/dvc remote modify minio endpointurl http://localhost:9000
.venv/bin/dvc remote modify --local minio access_key_id $(grep '^MINIO_ROOT_USER=' ../infra/.env | cut -d= -f2)
.venv/bin/dvc remote modify --local minio secret_access_key $(grep '^MINIO_ROOT_PASSWORD=' ../infra/.env | cut -d= -f2)
```

Note: `--local` modifies `.dvc/config.local` (gitignored). Public `.dvc/config` only stores the bucket URL + endpoint, never credentials.

- [ ] **Step 3: Verify the remote config**

```bash
.venv/bin/dvc remote list
.venv/bin/dvc config remote.minio.endpointurl
```

Expected:
- First command: `minio    s3://datasets/project-1`
- Second command: `http://localhost:9000`

- [ ] **Step 4: Extend project `.gitignore`**

Append to `/home/himanshu/learning/mlops-journey/project-1-credit-risk-pipeline/.gitignore`:

```
# DVC-tracked data (synced from MinIO via `dvc pull`)
/data/raw/
```

(If the file doesn't exist, create it with this content.)

- [ ] **Step 5: Commit DVC config (the public, non-credential parts)**

```bash
cd /home/himanshu/learning/mlops-journey
git add project-1-credit-risk-pipeline/.dvc project-1-credit-risk-pipeline/.gitignore
git commit -m "feat(project-1): initialize DVC with MinIO S3 remote"
```

Note: `.dvc/config.local` is auto-gitignored by DVC's own `.gitignore` — credentials never reach git.

---

## Task 3: Download German Credit dataset + track with DVC

**Files:**
- Create: `project-1-credit-risk-pipeline/data/raw/german_credit.csv`
- Create: `project-1-credit-risk-pipeline/data/raw/.gitignore` (auto by DVC)
- Create: `project-1-credit-risk-pipeline/data/raw/german_credit.csv.dvc`
- Create: `project-1-credit-risk-pipeline/data/README.md`

- [ ] **Step 1: Download the raw dataset**

```bash
cd /home/himanshu/learning/mlops-journey/project-1-credit-risk-pipeline
curl -fsSL https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data \
  -o data/raw/german_credit.csv
wc -l data/raw/german_credit.csv
```

Expected: `1000 data/raw/german_credit.csv`.

- [ ] **Step 2: Track the file with DVC**

```bash
.venv/bin/dvc add data/raw/german_credit.csv
```

Expected output mentions creating `data/raw/german_credit.csv.dvc` and updating `data/raw/.gitignore`.

- [ ] **Step 3: Push the dataset to MinIO**

```bash
.venv/bin/dvc push
```

Expected: `1 file pushed`.

- [ ] **Step 4: Verify file landed in MinIO `datasets` bucket**

```bash
docker run --rm --network infra_default \
  -e MC_HOST_local="http://$(grep MINIO_ROOT_USER ../infra/.env | cut -d= -f2):$(grep MINIO_ROOT_PASSWORD ../infra/.env | cut -d= -f2)@minio:9000" \
  minio/mc:RELEASE.2024-08-17T11-33-50Z ls --recursive local/datasets/project-1
```

Expected: at least one `.dvc/files/md5/...` object representing the file.

- [ ] **Step 5: Document the dataset**

Write `project-1-credit-risk-pipeline/data/README.md`:

```markdown
# Data

## Raw

| File | Source | Rows | Target | DVC-tracked |
|------|--------|------|--------|-------------|
| `raw/german_credit.csv` | [UCI Statlog](https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/) | 1000 | `default` (1=bad credit recoded to 1=default) | ✅ |

Columns: 20 mixed-type features. See `src/credit_risk/data_loader.py` for the
schema and `src/credit_risk/features.py` for preprocessing.

## Reproduce

```bash
dvc pull              # fetches data/raw/german_credit.csv from MinIO
```
```

- [ ] **Step 6: Commit the DVC pointer + docs**

```bash
cd /home/himanshu/learning/mlops-journey
git add project-1-credit-risk-pipeline/data/raw/german_credit.csv.dvc \
        project-1-credit-risk-pipeline/data/raw/.gitignore \
        project-1-credit-risk-pipeline/data/README.md
git commit -m "feat(project-1): DVC-track German Credit dataset (pushed to MinIO)"
```

---

## Task 4: Verify `dvc pull` round-trip works

**Files:** none new.

- [ ] **Step 1: Simulate a fresh checkout by deleting the local copy**

```bash
cd /home/himanshu/learning/mlops-journey/project-1-credit-risk-pipeline
rm data/raw/german_credit.csv
ls data/raw/
```

Expected: only `.gitignore` and `german_credit.csv.dvc` listed.

- [ ] **Step 2: Pull from MinIO**

```bash
.venv/bin/dvc pull
```

Expected: `1 file fetched`.

- [ ] **Step 3: Confirm the data file is back**

```bash
wc -l data/raw/german_credit.csv
```

Expected: `1000 data/raw/german_credit.csv`.

---

## Task 5: End-of-Day-1 verification

- [ ] **Step 1: Run repo verify.sh (should still be 18/18)**

```bash
cd /home/himanshu/learning/mlops-journey
make verify
```

Expected: `Result: 18 passed, 0 failed`.

- [ ] **Step 2: Push Day 1 progress to GitHub**

```bash
git push
```

Expected: 3 new commits land on `origin/main`.

---

# DAY 2 — Data Loader + Feature Engineering (TDD)

## Task 6: TDD `data_loader.py`

**Files:**
- Create: `project-1-credit-risk-pipeline/src/credit_risk/data_loader.py`
- Create: `project-1-credit-risk-pipeline/tests/test_data_loader.py`

- [ ] **Step 1: Write the failing test**

Write `project-1-credit-risk-pipeline/tests/test_data_loader.py`:

```python
"""Tests for the German Credit data loader."""

from pathlib import Path

import pandas as pd
import pytest

from credit_risk.data_loader import GERMAN_CREDIT_COLUMNS, load_german_credit

REPO_DATA = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "german_credit.csv"
)


@pytest.fixture
def raw_path() -> Path:
    if not REPO_DATA.exists():
        pytest.skip(f"raw data missing — run `dvc pull` first ({REPO_DATA})")
    return REPO_DATA


def test_load_returns_dataframe(raw_path):
    df = load_german_credit(raw_path)
    assert isinstance(df, pd.DataFrame)


def test_load_has_expected_shape(raw_path):
    df = load_german_credit(raw_path)
    assert df.shape == (1000, len(GERMAN_CREDIT_COLUMNS))


def test_load_target_is_binary_0_1(raw_path):
    df = load_german_credit(raw_path)
    assert set(df["default"].unique()) == {0, 1}


def test_load_target_default_rate_is_30_percent(raw_path):
    """UCI documents this dataset's default rate as 30%."""
    df = load_german_credit(raw_path)
    rate = df["default"].mean()
    assert 0.28 <= rate <= 0.32, f"unexpected default rate: {rate:.2%}"
```

- [ ] **Step 2: Run tests; verify they FAIL with import error**

```bash
cd /home/himanshu/learning/mlops-journey
.venv/bin/pytest project-1-credit-risk-pipeline -v
```

Expected: `ImportError: cannot import name 'load_german_credit' from 'credit_risk.data_loader'`.

- [ ] **Step 3: Write minimal `data_loader.py`**

Write `project-1-credit-risk-pipeline/src/credit_risk/data_loader.py`:

```python
"""Load the UCI German Credit dataset from a local CSV path."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

GERMAN_CREDIT_COLUMNS: Final[list[str]] = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job",
    "num_dependents", "own_telephone", "foreign_worker", "default",
]


def load_german_credit(path: Path | str) -> pd.DataFrame:
    """Load the German Credit CSV and recode the target to 0/1.

    UCI's raw labels are 1=good credit, 2=bad credit. We map bad → 1 ("default").
    """
    df = pd.read_csv(path, sep=" ", header=None, names=GERMAN_CREDIT_COLUMNS)
    df["default"] = (df["default"] == 2).astype(int)
    return df
```

- [ ] **Step 4: Run tests; verify all 4 PASS**

```bash
.venv/bin/pytest project-1-credit-risk-pipeline -v
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add project-1-credit-risk-pipeline/src/credit_risk/data_loader.py \
        project-1-credit-risk-pipeline/tests/test_data_loader.py
git commit -m "feat(project-1): add data_loader with German Credit schema"
```

---

## Task 7: TDD `features.py` — one-hot encoding + train/test split

**Files:**
- Create: `project-1-credit-risk-pipeline/src/credit_risk/features.py`
- Create: `project-1-credit-risk-pipeline/tests/test_features.py`

- [ ] **Step 1: Write the failing test**

Write `project-1-credit-risk-pipeline/tests/test_features.py`:

```python
"""Tests for feature engineering + train/test split."""

import numpy as np
import pandas as pd
import pytest

from credit_risk.features import FeatureSplit, encode_features, train_test_split_stratified


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """Mimics a few rows from the German Credit schema."""
    return pd.DataFrame(
        {
            "duration": [12, 24, 36, 18],
            "credit_amount": [1000, 5000, 2500, 800],
            "checking_status": ["A11", "A12", "A11", "A13"],
            "purpose": ["A40", "A41", "A40", "A42"],
            "default": [0, 1, 0, 1],
        }
    )


def test_encode_features_returns_numeric_only(tiny_df):
    X = encode_features(tiny_df.drop(columns=["default"]))
    assert X.select_dtypes(include="object").shape[1] == 0


def test_encode_features_keeps_row_count(tiny_df):
    X = encode_features(tiny_df.drop(columns=["default"]))
    assert len(X) == len(tiny_df)


def test_encode_features_drops_one_dummy_per_category(tiny_df):
    """drop_first=True avoids the dummy variable trap."""
    X = encode_features(tiny_df.drop(columns=["default"]))
    # 2 numeric + (3-1) checking_status + (3-1) purpose = 6 columns
    assert X.shape[1] == 6


def test_split_returns_featuresplit(tiny_df):
    out = train_test_split_stratified(tiny_df, target_col="default", random_state=0)
    assert isinstance(out, FeatureSplit)


def test_split_train_plus_test_sums_to_input(tiny_df):
    out = train_test_split_stratified(tiny_df, target_col="default", random_state=0)
    assert len(out.X_train) + len(out.X_test) == len(tiny_df)


def test_split_y_is_int(tiny_df):
    out = train_test_split_stratified(tiny_df, target_col="default", random_state=0)
    assert np.issubdtype(out.y_train.dtype, np.integer)
    assert np.issubdtype(out.y_test.dtype, np.integer)
```

- [ ] **Step 2: Run tests; verify they FAIL**

```bash
.venv/bin/pytest project-1-credit-risk-pipeline/tests/test_features.py -v
```

Expected: `ImportError: cannot import name 'encode_features' from 'credit_risk.features'`.

- [ ] **Step 3: Write minimal `features.py`**

Write `project-1-credit-risk-pipeline/src/credit_risk/features.py`:

```python
"""Feature engineering and train/test splitting for German Credit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class FeatureSplit:
    """Outcome of train/test splitting + encoding."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all categorical columns, drop_first=True.

    German Credit has mixed numeric + 'A11'-style categorical columns. XGBoost
    handles numeric input fastest, so we encode upfront rather than relying on
    enable_categorical.
    """
    return pd.get_dummies(df, drop_first=True)


def train_test_split_stratified(
    df: pd.DataFrame,
    target_col: str,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
) -> FeatureSplit:
    """Stratified split that one-hot encodes the features in one call."""
    y = df[target_col].astype(int).to_numpy()
    X = encode_features(df.drop(columns=[target_col]))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return FeatureSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
```

- [ ] **Step 4: Run tests; verify all 6 PASS**

```bash
.venv/bin/pytest project-1-credit-risk-pipeline/tests/test_features.py -v
```

Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add project-1-credit-risk-pipeline/src/credit_risk/features.py \
        project-1-credit-risk-pipeline/tests/test_features.py
git commit -m "feat(project-1): add encode_features + stratified split"
```

---

## Task 8: End-of-Day-2 — run full test suite

- [ ] **Step 1: Run all project-1 tests**

```bash
.venv/bin/pytest project-1-credit-risk-pipeline -v
```

Expected: `10 passed` (4 data_loader + 6 features).

- [ ] **Step 2: Push Day 2 progress**

```bash
git push
```

---

# DAY 3 — XGBoost Trainer (TDD)

## Task 9: TDD `train.py` — single XGBoost training run

**Files:**
- Create: `project-1-credit-risk-pipeline/src/credit_risk/train.py`
- Create: `project-1-credit-risk-pipeline/tests/test_train.py`

- [ ] **Step 1: Write the failing test**

Write `project-1-credit-risk-pipeline/tests/test_train.py`:

```python
"""Tests for the XGBoost training pipeline."""

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from credit_risk.features import train_test_split_stratified
from credit_risk.train import TrainingResult, train_xgboost


@pytest.fixture
def synthetic_split():
    """A 200-row synthetic dataset with two classes, enough to train."""
    rng = np.random.default_rng(seed=7)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "x3": rng.integers(0, 5, n),
            # signal-bearing target: positive class when x1+x2 > 0
            "default": ((rng.normal(0, 1, n) + 0.5) > 0).astype(int),
        }
    )
    return train_test_split_stratified(df, target_col="default", random_state=0)


def test_train_returns_trainingresult(synthetic_split):
    result = train_xgboost(synthetic_split, n_estimators=20, max_depth=3, random_state=0)
    assert isinstance(result, TrainingResult)


def test_train_model_is_xgbclassifier(synthetic_split):
    result = train_xgboost(synthetic_split, n_estimators=20, max_depth=3, random_state=0)
    assert isinstance(result.model, XGBClassifier)


def test_train_metrics_are_in_unit_interval(synthetic_split):
    result = train_xgboost(synthetic_split, n_estimators=20, max_depth=3, random_state=0)
    for metric_name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        value = getattr(result, metric_name)
        assert 0.0 <= value <= 1.0, f"{metric_name}={value} out of range"


def test_train_records_n_rows(synthetic_split):
    result = train_xgboost(synthetic_split, n_estimators=20, max_depth=3, random_state=0)
    assert result.n_train + result.n_test == 200


def test_train_records_params(synthetic_split):
    result = train_xgboost(synthetic_split, n_estimators=20, max_depth=3, random_state=0)
    assert result.params["n_estimators"] == 20
    assert result.params["max_depth"] == 3
```

- [ ] **Step 2: Run tests; verify they FAIL**

```bash
.venv/bin/pytest project-1-credit-risk-pipeline/tests/test_train.py -v
```

Expected: `ImportError: cannot import name 'train_xgboost'`.

- [ ] **Step 3: Write minimal `train.py`**

Write `project-1-credit-risk-pipeline/src/credit_risk/train.py`:

```python
"""XGBoost training for credit-risk classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from credit_risk.features import FeatureSplit


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


def train_xgboost(
    split: FeatureSplit,
    *,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> TrainingResult:
    """Train an XGBClassifier and return metrics + model + params."""
    params: dict[str, Any] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "random_state": random_state,
        "eval_metric": "logloss",
        "n_jobs": -1,
    }
    model = XGBClassifier(**params)
    model.fit(split.X_train, split.y_train)

    y_pred = model.predict(split.X_test)
    y_proba = model.predict_proba(split.X_test)[:, 1]

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
    )
```

- [ ] **Step 4: Run tests; verify all 5 PASS**

```bash
.venv/bin/pytest project-1-credit-risk-pipeline/tests/test_train.py -v
```

Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add project-1-credit-risk-pipeline/src/credit_risk/train.py \
        project-1-credit-risk-pipeline/tests/test_train.py
git commit -m "feat(project-1): add XGBoost trainer with 5 metrics + params"
```

---

## Task 10: End-of-Day-3 verification

- [ ] **Step 1: Run all project-1 tests**

```bash
.venv/bin/pytest project-1-credit-risk-pipeline -v
```

Expected: `15 passed` (4 data_loader + 6 features + 5 train).

- [ ] **Step 2: Push**

```bash
git push
```

---

# DAY 4 — MLflow Integration + Model Registry

## Task 11: TDD `registry.py` — register a model to MLflow

**Files:**
- Create: `project-1-credit-risk-pipeline/src/credit_risk/registry.py`
- Create: `project-1-credit-risk-pipeline/tests/test_registry.py`

- [ ] **Step 1: Write the failing test (integration-style — hits real MLflow)**

Write `project-1-credit-risk-pipeline/tests/test_registry.py`:

```python
"""Integration tests for the MLflow Model Registry helpers.

These require the docker-compose stack to be UP. Skipped otherwise.
"""

import os

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from credit_risk.registry import REGISTERED_MODEL_NAME, register_model_from_run


@pytest.fixture(scope="module")
def mlflow_up() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    try:
        import urllib.request

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
```

- [ ] **Step 2: Run tests; verify they FAIL with import error**

```bash
.venv/bin/pytest project-1-credit-risk-pipeline/tests/test_registry.py -v
```

Expected: `ImportError: cannot import name 'register_model_from_run'`.

- [ ] **Step 3: Write minimal `registry.py`**

Write `project-1-credit-risk-pipeline/src/credit_risk/registry.py`:

```python
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
    # Ensure the registered model exists (idempotent)
    try:
        client.create_registered_model(REGISTERED_MODEL_NAME)
    except mlflow.exceptions.RestException:
        # already exists
        pass

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
        # Re-fetch so caller sees the updated stage
        model_version = client.get_model_version(
            name=REGISTERED_MODEL_NAME, version=model_version.version
        )

    return model_version
```

- [ ] **Step 4: Confirm the stack is UP, then run tests**

```bash
cd /home/himanshu/learning/mlops-journey
make up
.venv/bin/pytest project-1-credit-risk-pipeline/tests/test_registry.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add project-1-credit-risk-pipeline/src/credit_risk/registry.py \
        project-1-credit-risk-pipeline/tests/test_registry.py
git commit -m "feat(project-1): MLflow Model Registry helpers + stage transitions"
```

---

## Task 12: Build `cli.py` — orchestrates the full sweep + registration

**Files:**
- Create: `project-1-credit-risk-pipeline/src/credit_risk/cli.py`

- [ ] **Step 1: Write the CLI module**

Write `project-1-credit-risk-pipeline/src/credit_risk/cli.py`:

```python
"""Typer CLI for the credit-risk pipeline.

Commands:
    credit-risk train       — run a 3-trial hyperparameter sweep + log to MLflow
    credit-risk register    — promote the best run to the Model Registry @ Staging
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import mlflow
import mlflow.sklearn
import typer

from credit_risk.data_loader import load_german_credit
from credit_risk.features import train_test_split_stratified
from credit_risk.registry import REGISTERED_MODEL_NAME, register_model_from_run
from credit_risk.train import train_xgboost

app = typer.Typer(no_args_is_help=True, help="Credit-risk pipeline CLI.")

EXPERIMENT_NAME = "credit-risk-classifier"

# Three trial configs — kept small for laptop runs; Project 1 Phase 2 expands this.
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
    # cli.py is at project-1-credit-risk-pipeline/src/credit_risk/cli.py
    # parents[2] resolves to project-1-credit-risk-pipeline/
    ] = Path(__file__).resolve().parents[2] / "data" / "raw" / "german_credit.csv",
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
            mlflow.sklearn.log_model(result.model, artifact_path="model")
            typer.echo(
                f"  trial-{i}: roc_auc={result.roc_auc:.3f}  "
                f"accuracy={result.accuracy:.3f}  f1={result.f1:.3f}"
            )


@app.command()
def register(
    metric: Annotated[
        str, typer.Option(help="Metric used to pick the best run.")
    ] = "roc_auc",
    stage: Annotated[
        str, typer.Option(help="Target stage (None/Staging/Production).")
    ] = "Staging",
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
    typer.echo(
        f"Registered '{REGISTERED_MODEL_NAME}' v{mv.version} at stage '{mv.current_stage}'."
    )


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Smoke-test the CLI's `--help`**

```bash
cd /home/himanshu/learning/mlops-journey
.venv/bin/python -m credit_risk.cli --help
```

Expected: shows the two subcommands `train` and `register`.

- [ ] **Step 3: Commit**

```bash
git add project-1-credit-risk-pipeline/src/credit_risk/cli.py
git commit -m "feat(project-1): typer CLI with train + register subcommands"
```

---

## Task 13: Add Makefile shortcuts

**Files:**
- Modify: `Makefile` (repo root)

- [ ] **Step 1: Append the project-1 targets**

Add these targets to the END of `/home/himanshu/learning/mlops-journey/Makefile`:

```makefile
# ── Project 1 — Credit Risk Pipeline ──

p1-data:  ## Pull project-1 raw data from MinIO (DVC)
	cd project-1-credit-risk-pipeline && ../.venv/bin/dvc pull

p1-train:  ## Run the 3-trial hyperparameter sweep, log all to MLflow
	cd project-1-credit-risk-pipeline && ../.venv/bin/python -m credit_risk.cli train

p1-register:  ## Promote the best run to Model Registry @ Staging
	cd project-1-credit-risk-pipeline && ../.venv/bin/python -m credit_risk.cli register

p1-test:  ## Run all project-1 tests
	.venv/bin/pytest project-1-credit-risk-pipeline -v
```

- [ ] **Step 2: Test each target**

```bash
make help | grep p1-
make p1-test
```

Expected: `make help` lists 4 p1- targets; `make p1-test` shows `15 passed` (or 17 if registry tests can connect to MLflow).

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "chore: add Makefile p1- shortcuts (data/train/register/test)"
```

---

# DAY 5 — Run end-to-end, polish, push

## Task 14: Execute the full pipeline end-to-end

- [ ] **Step 1: Ensure the stack is up**

```bash
make verify
```

Expected: `18 passed, 0 failed`.

- [ ] **Step 2: Run the training sweep**

```bash
make p1-train
```

Expected: 3 lines like `trial-1: roc_auc=0.7xx  accuracy=0.7xx  f1=0.xxx`. Takes ~30-60 seconds.

- [ ] **Step 3: Confirm three runs appear in MLflow**

```bash
curl -s "http://localhost:5000/api/2.0/mlflow/experiments/search" -X POST -H "Content-Type: application/json" -d '{"max_results":10}' | python3 -c "import json,sys; data=json.load(sys.stdin); [print(f'  {e[\"name\"]:30s} id={e[\"experiment_id\"]}') for e in data['experiments']]"
```

Expected: includes `credit-risk-classifier` experiment.

- [ ] **Step 4: Register the best run to the Model Registry**

```bash
make p1-register
```

Expected: prints `Registered 'credit-risk-classifier' v1 at stage 'Staging'.` (or higher version if you've run it before).

- [ ] **Step 5: Confirm the registered model exists**

```bash
curl -s "http://localhost:5000/api/2.0/mlflow/registered-models/get?name=credit-risk-classifier" | python3 -m json.tool | head -20
```

Expected: JSON object showing `name=credit-risk-classifier` with `latest_versions` array containing at least one entry with `current_stage=Staging`.

---

## Task 15: Update repo-level docs (README + STATUS)

**Files:**
- Modify: `README.md` (repo root)
- Modify: `STATUS.md`

- [ ] **Step 1: Update the projects table in `README.md`**

Find the row in `README.md` for Project 1 and change its Status column from `📋 Planned` to `🟡 Phase 1 in progress`. Specifically, replace:

```markdown
| 1 | Self-Healing Credit-Risk Pipeline | Fintech | 📋 Planned |
```

with:

```markdown
| 1 | Self-Healing Credit-Risk Pipeline | Fintech | 🟡 Phase 1 |
```

- [ ] **Step 2: Update `STATUS.md` "Current" section**

Replace the `## Current` block in `STATUS.md` with:

```markdown
## Current

- **Week:** 2 of 26
- **Project:** project-1-credit-risk-pipeline (Phase 1: data + XGBoost + registry)
- **Infrastructure:** 18 / 18 verify checks passing
- **Model Registry:** `credit-risk-classifier` v1 @ Staging (Phase 1 deliverable)
- **Blockers:** none
```

And update the milestones section to show Project 1 Phase 1 progress:

```markdown
## Project 1 — Phases (5 weeks)

- [x] Phase 1 (Week 2) — Data + XGBoost + MLflow Model Registry
- [ ] Phase 2 (Week 3) — FastAPI serving + Dockerization
- [ ] Phase 3 (Week 4) — Evidently AI drift detection + auto-retrain
- [ ] Phase 4 (Week 5) — Canary deploy + Prometheus monitoring
- [ ] Phase 5 (Week 6) — AWS EC2 production demo + portfolio polish
```

- [ ] **Step 3: Commit**

```bash
git add README.md STATUS.md
git commit -m "docs: Project 1 Phase 1 complete — model in Registry @ Staging"
```

---

## Task 16: Final smoke test + push

- [ ] **Step 1: Restart the stack from cold + verify everything is reproducible**

```bash
make down && make up
sleep 30
make verify
```

Expected: `18 passed, 0 failed`.

- [ ] **Step 2: Re-run the pipeline from scratch**

```bash
make p1-test       # 15 tests pass
make p1-train      # 3 trials log
make p1-register   # registers a new version
```

Expected: all three commands succeed; new model version appears in Registry.

- [ ] **Step 3: Push everything to GitHub**

```bash
git push
```

Expected: ~16-18 new commits visible on `origin/main`. CI workflow re-runs and stays green.

- [ ] **Step 4: Verify CI passed**

Open `https://github.com/himanshunigam-456/Mlops-journey/actions` in a browser. Latest run should show 🟢.

---

# Phase 1 Complete

**You shipped:**
- DVC-tracked dataset stored in MinIO, reproducible via `dvc pull`
- XGBoost trainer with full metric coverage (accuracy, precision, recall, f1, roc_auc)
- 15 tests across 4 modules (data_loader, features, train, registry)
- Typer CLI with `train` + `register` subcommands
- MLflow Model Registry usage with `Staging` stage transitions
- `make p1-*` Makefile targets

**Next phase preview (Phase 2 / Week 3):**
- FastAPI app loading the `Staging` model on startup
- `/predict` endpoint with Pydantic request/response schemas
- Dockerized image pushed to GHCR
- Smoke + load tests in CI

Tell Claude: *"plan project 1 phase 2"* to begin.
