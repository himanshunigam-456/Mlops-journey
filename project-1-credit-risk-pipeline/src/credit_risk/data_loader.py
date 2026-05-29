"""Load the UCI German Credit dataset from a local CSV path."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

GERMAN_CREDIT_COLUMNS: Final[list[str]] = [
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_commitment",
    "personal_status",
    "other_parties",
    "residence_since",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
    "default",
]


def load_german_credit(path: Path | str) -> pd.DataFrame:
    """Load the German Credit CSV and recode the target to 0/1.

    UCI's raw labels are 1=good credit, 2=bad credit. We map bad → 1 ("default").
    """
    df = pd.read_csv(path, sep=" ", header=None, names=GERMAN_CREDIT_COLUMNS)
    df["default"] = (df["default"] == 2).astype(int)
    return df
