"""Tests for the Evidently DataDriftPreset wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from credit_risk.drift.reporter import DriftSummary, compute_drift_report


@pytest.fixture
def reference_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "feature_age": rng.normal(35, 10, 500),
            "feature_credit_amount": rng.normal(3000, 1000, 500),
            "feature_duration": rng.normal(36, 12, 500),
            "prediction": rng.integers(0, 2, 500),
            "probability_default": rng.uniform(0, 1, 500),
        }
    )


@pytest.fixture
def current_df_no_drift() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "feature_age": rng.normal(35, 10, 500),
            "feature_credit_amount": rng.normal(3000, 1000, 500),
            "feature_duration": rng.normal(36, 12, 500),
            "prediction": rng.integers(0, 2, 500),
            "probability_default": rng.uniform(0, 1, 500),
        }
    )


@pytest.fixture
def current_df_with_drift() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    return pd.DataFrame(
        {
            "feature_age": rng.normal(50, 10, 500),  # SHIFTED +15
            "feature_credit_amount": rng.normal(8000, 1500, 500),  # SHIFTED +5000
            "feature_duration": rng.normal(36, 12, 500),
            "prediction": rng.integers(0, 2, 500),
            "probability_default": rng.uniform(0, 1, 500),
        }
    )


def test_compute_drift_report_returns_summary(reference_df, current_df_no_drift):
    summary = compute_drift_report(reference_df, current_df_no_drift)
    assert isinstance(summary, DriftSummary)
    assert summary.html is not None
    assert len(summary.html) > 1000  # Evidently HTML is many KB


def test_no_drift_reports_low_share(reference_df, current_df_no_drift):
    summary = compute_drift_report(reference_df, current_df_no_drift)
    # Same distribution, different seed — most columns should NOT drift.
    # Allow some slack: 5 cols × 5% alpha = ~1 false positive expected.
    assert summary.drift_share <= 0.4


def test_with_drift_reports_high_share(reference_df, current_df_with_drift):
    summary = compute_drift_report(reference_df, current_df_with_drift)
    assert summary.drift_share > 0.3
    # The two shifted columns should be flagged
    drifted = set(summary.drifted_columns)
    assert "feature_age" in drifted or "feature_credit_amount" in drifted


def test_n_columns_matches_input(reference_df, current_df_no_drift):
    summary = compute_drift_report(reference_df, current_df_no_drift)
    # 5 columns in the fixtures
    assert summary.n_columns == 5
