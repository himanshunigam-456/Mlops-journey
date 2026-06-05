"""Tests for prediction capture — Parquet-friendly schema + flattening."""

from __future__ import annotations

from credit_risk.drift.capture import PredictionRow, capture_row, to_dataframe


def test_capture_row_returns_predictionrow():
    row = capture_row(
        request_id="req-1",
        features={"duration": 12.0, "credit_amount": 1500.0},
        prediction=0,
        probability_default=0.21,
        model_version="6",
    )
    assert isinstance(row, PredictionRow)
    assert row.prediction == 0
    assert row.probability_default == 0.21


def test_capture_row_records_scored_at_utc_iso():
    row = capture_row(
        request_id="req-1",
        features={"x": 1.0},
        prediction=1,
        probability_default=0.8,
        model_version="6",
    )
    assert row.scored_at.endswith("Z")  # ISO-8601 UTC convention


def test_to_dataframe_returns_one_row_per_capture():
    rows = [
        capture_row(
            request_id=f"req-{i}",
            features={"duration": float(i)},
            prediction=i % 2,
            probability_default=0.5,
            model_version="6",
        )
        for i in range(5)
    ]
    df = to_dataframe(rows)
    assert len(df) == 5
    assert "request_id" in df.columns
    assert "feature_duration" in df.columns  # features get the feature_ prefix


def test_to_dataframe_empty_input_returns_empty_df():
    df = to_dataframe([])
    assert len(df) == 0


def test_capture_row_coerces_types():
    """Defensive: callers might pass numpy types or strings."""
    row = capture_row(
        request_id="req-1",
        features={"x": 1.0},
        prediction="1",  # str
        probability_default="0.42",  # str
        model_version=6,  # int
    )
    assert row.prediction == 1
    assert row.probability_default == 0.42
    assert row.model_version == "6"
