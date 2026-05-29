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
    """Outcome of one XGBoost training run.

    Frozen so callers can't post-edit metrics or swap the model after the
    fact. Any "I want different metrics" path must call train_xgboost again.
    """

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
    """Train an XGBClassifier on the given split, return 5 metrics + model + params.

    `zero_division=0` on precision/recall avoids warnings when the predicted
    positive set is empty (which can happen on tiny test sets).
    """
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
