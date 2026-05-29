"""Feature engineering and train/test splitting for German Credit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class FeatureSplit:
    """Outcome of train/test splitting + encoding.

    Frozen so callers can't mutate the splits after creation — any retraining
    should produce a NEW FeatureSplit, never modify an existing one.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all categorical columns with drop_first=True.

    German Credit has mixed numeric + 'A11'-style categorical columns. XGBoost
    handles numeric input fastest, so we encode upfront rather than relying on
    enable_categorical. `drop_first=True` avoids the dummy variable trap
    (perfect collinearity across one-hot columns).
    """
    return pd.get_dummies(df, drop_first=True)


def train_test_split_stratified(
    df: pd.DataFrame,
    target_col: str,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
) -> FeatureSplit:
    """Stratified train/test split + one-hot encoding in one call.

    Preserves the target's class distribution in both train and test — critical
    when the target is imbalanced (German Credit is ~30% positive).
    """
    y = df[target_col].astype(int).to_numpy()
    X = encode_features(df.drop(columns=[target_col]))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return FeatureSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
