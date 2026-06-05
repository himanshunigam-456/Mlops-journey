"""Generate a SHIFTED-distribution customer CSV — drift detection demo input.

Same Indian-bank schema as the Phase 2.5 baseline (sample_indian_customers.csv)
but with intentionally shifted distributions, simulating "post-event production
traffic":

  - Salaried 30% (was 60%)  → Self-Employed 40% (was 20%)
  - Home Loan 15% (was 30%) → Personal Loan 40% (was 25%)
  - Rent 55% (was 40%) housing
  - Higher median income (lognormal mu 11.0 vs 10.7)
  - Older age mode (38 vs 32)
  - More existing loans + higher EMI burden

Used by the drift demo (drift_check.py) to verify the detector flags the
shifts. NOT real customer data — purely synthetic.

Usage:
    python scripts/generate_drifted_customers.py
    python scripts/generate_drifted_customers.py --num-rows 1000 --output custom.csv
"""

from __future__ import annotations

import argparse
import random
import string
from pathlib import Path

import pandas as pd
from faker import Faker

HERE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = HERE / "examples" / "sample_indian_customers_drifted.csv"

# Shifted weights — see baseline at generate_synthetic_customers.py for compare
EMPLOYMENT_TYPES = {
    "Salaried": 0.30,  # ↓ from 0.60
    "Self-Employed": 0.40,  # ↑ from 0.20
    "Business": 0.15,
    "Professional": 0.10,
    "Homemaker": 0.03,
    "Retired": 0.02,
}

LOAN_PURPOSES = {
    "Home Loan": 0.15,  # ↓ from 0.30
    "Car Loan": 0.20,
    "Personal Loan": 0.40,  # ↑ from 0.25
    "Education Loan": 0.10,
    "Business Loan": 0.10,
    "Marriage Loan": 0.05,
}

HOUSING_STATUS = {"Own": 0.30, "Rent": 0.55, "Family": 0.15}  # ↑ rent

TENURE_BY_PURPOSE = {
    "Home Loan": [120, 180, 240, 300],
    "Car Loan": [36, 48, 60, 84],
    "Personal Loan": [12, 24, 36, 48, 60],
    "Education Loan": [60, 84, 120, 180],
    "Business Loan": [12, 24, 36, 60],
    "Marriage Loan": [12, 24, 36, 48],
}

AMOUNT_BY_PURPOSE = {
    "Home Loan": (1_500_000, 15_000_000),
    "Car Loan": (300_000, 2_500_000),
    "Personal Loan": (50_000, 2_000_000),
    "Education Loan": (200_000, 5_000_000),
    "Business Loan": (500_000, 10_000_000),
    "Marriage Loan": (100_000, 1_500_000),
}


def _weighted_choice(choices: dict[str, float]) -> str:
    return random.choices(list(choices.keys()), weights=list(choices.values()), k=1)[0]


def _make_pan() -> str:
    return (
        "".join(random.choices(string.ascii_uppercase, k=5))
        + "".join(random.choices(string.digits, k=4))
        + random.choice(string.ascii_uppercase)
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-rows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=99)  # different from baseline
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()

    random.seed(args.seed)
    fake = Faker("en_IN")
    Faker.seed(args.seed)

    rows = []
    for i in range(1, args.num_rows + 1):
        # Demographics (older mode)
        age = int(random.triangular(21, 65, mode=38))
        gender = random.choice(["M", "F"])
        dependents = random.choices([0, 1, 2, 3, 4, 5], weights=[15, 30, 30, 15, 8, 2])[0]

        # Higher median income (mu shifted up)
        monthly_income_inr = int(random.lognormvariate(mu=11.0, sigma=0.7))
        monthly_income_inr = max(15_000, min(monthly_income_inr, 2_000_000))

        employment_type = _weighted_choice(EMPLOYMENT_TYPES)
        max_job_years = max(1, age - 21)
        years_in_current_job = random.randint(0, min(max_job_years, 30))

        # More existing loans + higher EMI burden
        existing_loans = random.choices([0, 1, 2, 3], weights=[35, 35, 20, 10])[0]
        existing_emi_inr = (
            random.randint(int(monthly_income_inr * 0.15), int(monthly_income_inr * 0.45))
            if existing_loans > 0
            else 0
        )

        purpose = _weighted_choice(LOAN_PURPOSES)
        amount_lo, amount_hi = AMOUNT_BY_PURPOSE[purpose]
        loan_amount_inr = random.randint(amount_lo, amount_hi)
        loan_duration_months = random.choice(TENURE_BY_PURPOSE[purpose])

        rows.append(
            {
                "customer_id": f"CUST-2026-D{i:05d}",
                "full_name": fake.name(),
                "pan": _make_pan(),
                "age": age,
                "gender": gender,
                "dependents": dependents,
                "monthly_income_inr": monthly_income_inr,
                "employment_type": employment_type,
                "years_in_current_job": years_in_current_job,
                "existing_loans": existing_loans,
                "existing_emi_inr": existing_emi_inr,
                "loan_purpose": purpose,
                "loan_amount_inr": loan_amount_inr,
                "loan_duration_months": loan_duration_months,
                "housing_status": _weighted_choice(HOUSING_STATUS),
                "pincode": f"{random.randint(110001, 999999)}",
                "city": fake.city(),
                "state": fake.state(),
            }
        )

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} DRIFTED customer rows → {args.output}")
    print()
    se_share = (df["employment_type"] == "Self-Employed").mean()
    pl_share = (df["loan_purpose"] == "Personal Loan").mean()
    rent_share = (df["housing_status"] == "Rent").mean()
    print(f"Median income:        ₹{df['monthly_income_inr'].median():,.0f} (baseline: ₹44,498)")
    print(f"Self-Employed share:  {se_share:.1%} (baseline: 19.8%)")
    print(f"Personal Loan share:  {pl_share:.1%} (baseline: 23.8%)")
    print(f"Rent housing share:   {rent_share:.1%} (baseline: ~40%)")


if __name__ == "__main__":
    main()
