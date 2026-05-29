"""Tests for the German Credit data loader."""

from pathlib import Path

import pandas as pd
import pytest
from credit_risk.data_loader import GERMAN_CREDIT_COLUMNS, load_german_credit

REPO_DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "german_credit.csv"


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
