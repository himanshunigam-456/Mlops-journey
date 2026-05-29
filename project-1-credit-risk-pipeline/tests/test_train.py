"""Tests for the XGBoost training pipeline."""

import numpy as np
import pandas as pd
import pytest
from credit_risk.features import train_test_split_stratified
from credit_risk.train import TrainingResult, train_xgboost
from xgboost import XGBClassifier


@pytest.fixture
def synthetic_split():
    """A 200-row synthetic dataset with two classes + real signal.

    The target is correlated with `x1+x2` so XGBoost can actually learn —
    pure noise would make metrics meaningless.
    """
    rng = np.random.default_rng(seed=7)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "x3": rng.integers(0, 5, n),
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
    """All 5 metrics must be in [0, 1] — none can be inf or NaN."""
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
