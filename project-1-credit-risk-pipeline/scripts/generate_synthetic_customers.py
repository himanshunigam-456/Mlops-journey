"""Generate a synthetic Indian-bank-formatted customer CSV for the demo.

Produces a CSV with realistic-looking Indian fintech fields (PAN format, INR
amounts, employment categories, pincodes) but with NO real customer data.
Used by the batch scoring demo + Streamlit UI.

NOTE: This file is purely synthetic. It is NOT real customer data, NOT
sourced from any bank, and the schema is designed to resemble what an
Indian retail bank's loan-application pipeline would emit.

Usage:
    python scripts/generate_synthetic_customers.py
    python scripts/generate_synthetic_customers.py --num-rows 500 --output custom.csv
"""

from __future__ import annotations

import argparse
import random
import string
from pathlib import Path

import pandas as pd
from faker import Faker

EMPLOYMENT_TYPES = {
    "Salaried": 0.60,
    "Self-Employed": 0.20,
    "Business": 0.10,
    "Professional": 0.05,
    "Homemaker": 0.03,
    "Retired": 0.02,
}

LOAN_PURPOSES = {
    "Home Loan": 0.30,
    "Car Loan": 0.20,
    "Personal Loan": 0.25,
    "Education Loan": 0.10,
    "Business Loan": 0.10,
    "Marriage Loan": 0.05,
}

HOUSING_STATUS = {"Own": 0.45, "Rent": 0.40, "Family": 0.15}

# Typical bank loan tenures (months), grouped by purpose
TENURE_BY_PURPOSE = {
    "Home Loan": [120, 180, 240, 300],
    "Car Loan": [36, 48, 60, 84],
    "Personal Loan": [12, 24, 36, 48, 60],
    "Education Loan": [60, 84, 120, 180],
    "Business Loan": [12, 24, 36, 60],
    "Marriage Loan": [12, 24, 36, 48],
}

# Loan amount range by purpose (in INR)
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
    """PAN format: 5 letters + 4 digits + 1 letter (synthetic, not validated)."""
    letters = "".join(random.choices(string.ascii_uppercase, k=5))
    digits = "".join(random.choices(string.digits, k=4))
    suffix = random.choice(string.ascii_uppercase)
    return f"{letters}{digits}{suffix}"


def _customer_id(i: int) -> str:
    return f"CUST-2026-{i:05d}"


def generate(num_rows: int, seed: int) -> pd.DataFrame:
    random.seed(seed)
    fake = Faker("en_IN")
    Faker.seed(seed)

    rows = []
    for i in range(1, num_rows + 1):
        # Demographics
        age = int(random.triangular(21, 65, mode=32))
        gender = random.choice(["M", "F"])
        dependents = random.choices([0, 1, 2, 3, 4, 5], weights=[10, 25, 35, 20, 8, 2])[0]

        # Income (log-normal — realistic Indian income distribution)
        monthly_income_inr = int(random.lognormvariate(mu=10.7, sigma=0.7))
        monthly_income_inr = max(12_000, min(monthly_income_inr, 1_500_000))

        # Employment
        employment_type = _weighted_choice(EMPLOYMENT_TYPES)
        max_job_years = max(1, age - 21)
        years_in_current_job = random.randint(0, min(max_job_years, 30))

        # Existing obligations
        existing_loans = random.choices([0, 1, 2, 3], weights=[50, 30, 15, 5])[0]
        existing_emi_inr = (
            random.randint(int(monthly_income_inr * 0.10), int(monthly_income_inr * 0.40))
            if existing_loans > 0
            else 0
        )

        # The loan being requested
        purpose = _weighted_choice(LOAN_PURPOSES)
        amount_lo, amount_hi = AMOUNT_BY_PURPOSE[purpose]
        loan_amount_inr = random.randint(amount_lo, amount_hi)
        loan_duration_months = random.choice(TENURE_BY_PURPOSE[purpose])

        # Housing
        housing_status = _weighted_choice(HOUSING_STATUS)

        # Location
        pincode = f"{random.randint(110001, 999999)}"
        state = fake.state()
        city = fake.city()

        rows.append(
            {
                "customer_id": _customer_id(i),
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
                "housing_status": housing_status,
                "pincode": pincode,
                "city": city,
                "state": state,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-rows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "examples" / "sample_indian_customers.csv",
    )
    args = p.parse_args()

    df = generate(args.num_rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Wrote {len(df)} synthetic customer rows → {args.output}")
    print()
    print("Summary stats:")
    print(
        f"  age:                   median={df['age'].median():>6.0f}  "
        f"range=[{df['age'].min()}, {df['age'].max()}]"
    )
    print(f"  monthly_income_inr:    median=₹{df['monthly_income_inr'].median():>10,.0f}")
    print(f"  loan_amount_inr:       median=₹{df['loan_amount_inr'].median():>10,.0f}")
    print(f"  loan_duration_months:  median={df['loan_duration_months'].median():>6.0f}")
    print(f"  existing_loans:        mean={df['existing_loans'].mean():>4.2f}")
    print()
    print("Purpose distribution:")
    print(df["loan_purpose"].value_counts().to_string())
    print()
    print("Employment distribution:")
    print(df["employment_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
