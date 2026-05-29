"""Smoke tests for the warmup baseline. Red-Green-Refactor TDD style."""

import numpy as np
import pandas as pd
import pytest
from credit_baseline.pipeline import BaselineResult, train_baseline


@pytest.fixture
def tiny_synthetic_df():
    """A 100-row synthetic credit-like dataset. Just enough to train end-to-end."""
    rng = np.random.default_rng(seed=42)
    n = 100
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 75, size=n),
            "income": rng.normal(50_000, 20_000, size=n).clip(min=5_000),
            "loan_amount": rng.normal(15_000, 8_000, size=n).clip(min=500),
            "credit_history_years": rng.integers(0, 30, size=n),
            # binary target: ~50/50 split so stratification works
            "default": (rng.normal(0, 1, size=n) > 0).astype(int),
        }
    )
    return df


def test_train_baseline_returns_result(tiny_synthetic_df):
    """The function returns a BaselineResult dataclass."""
    result = train_baseline(tiny_synthetic_df, target_col="default", random_state=0)
    assert isinstance(result, BaselineResult)


def test_train_baseline_metrics_are_in_unit_interval(tiny_synthetic_df):
    """Accuracy and AUC are in [0, 1]."""
    result = train_baseline(tiny_synthetic_df, target_col="default", random_state=0)
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.roc_auc <= 1.0


def test_train_baseline_model_predicts_correct_shape(tiny_synthetic_df):
    """The returned model predicts one binary label per input row."""
    result = train_baseline(tiny_synthetic_df, target_col="default", random_state=0)
    X = tiny_synthetic_df.drop(columns=["default"])
    preds = result.model.predict(X)
    assert preds.shape == (len(tiny_synthetic_df),)
    assert set(np.unique(preds)).issubset({0, 1})
