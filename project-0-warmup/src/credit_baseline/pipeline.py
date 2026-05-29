"""Minimal credit-risk baseline.

This module is the *seed* for Project 1 (self-healing pipeline). It does the
essential vertical slice:
  - take a labeled DataFrame
  - train/test split (stratified)
  - fit a RandomForest
  - return a structured result with model + metrics

Project 1 will replace this with: DVC-tracked data, XGBoost, MLflow runs,
canary deployment, drift detection. For now: keep it boring and well-tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

DEFAULT_TEST_SIZE: Final[float] = 0.25


@dataclass(frozen=True)
class BaselineResult:
    """Outcome of one training run.

    Frozen so callers can't mutate metrics after the fact — anything that
    needs a different result should call train_baseline() again.
    """

    model: RandomForestClassifier
    accuracy: float
    roc_auc: float
    n_train: int
    n_test: int


def train_baseline(
    df: pd.DataFrame,
    target_col: str,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = 42,
) -> BaselineResult:
    """Train a baseline RandomForest classifier on the given DataFrame.

    Args:
        df: a labeled DataFrame including the target column.
        target_col: the column name to predict (must be binary 0/1).
        test_size: fraction held out for evaluation.
        random_state: seeds train_test_split and the model for reproducibility.
    """
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return BaselineResult(
        model=model,
        accuracy=accuracy_score(y_test, y_pred),
        roc_auc=roc_auc_score(y_test, y_proba),
        n_train=len(X_train),
        n_test=len(X_test),
    )
