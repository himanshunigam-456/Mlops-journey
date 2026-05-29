"""Tests for feature engineering + train/test split."""

import numpy as np
import pandas as pd
import pytest
from credit_risk.features import (
    FeatureSplit,
    encode_features,
    train_test_split_stratified,
)


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """16-row synthetic mimic of German Credit.

    Size matters: stratified split with test_size=0.25 needs >= n_classes test
    rows per class, so n_rows must be >= 4 * n_classes (= 8) at minimum.
    16 gives a comfortable margin (4 test rows, 2 per class).
    """
    return pd.DataFrame(
        {
            "duration": [12, 24, 36, 18] * 4,
            "credit_amount": [1000, 5000, 2500, 800] * 4,
            "checking_status": ["A11", "A12", "A11", "A13"] * 4,
            "purpose": ["A40", "A41", "A40", "A42"] * 4,
            "default": [0, 1, 0, 1] * 4,
        }
    )


def test_encode_features_returns_numeric_only(tiny_df):
    X = encode_features(tiny_df.drop(columns=["default"]))
    assert X.select_dtypes(include="object").shape[1] == 0


def test_encode_features_keeps_row_count(tiny_df):
    X = encode_features(tiny_df.drop(columns=["default"]))
    assert len(X) == len(tiny_df)


def test_encode_features_drops_one_dummy_per_category(tiny_df):
    """drop_first=True avoids the dummy variable trap.

    2 numeric cols (duration, credit_amount) +
    (3 unique checking_status - 1) +
    (3 unique purpose - 1) = 6 total columns.
    """
    X = encode_features(tiny_df.drop(columns=["default"]))
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
