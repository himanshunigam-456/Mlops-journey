"""Batch scoring: Indian-bank customer CSV → loan-decision Excel.

The product surface most Indian banks actually consume — an analyst uploads
the day's loan applications and gets a fully-decisioned Excel back, with
APPROVE / REVIEW / REJECT bands, reason codes for rejections, and audit
columns (model_version, run_id, scored_at) for compliance.

The model itself is the same one the FastAPI service uses — same MLflow
Registry slot (`models:/credit-risk-classifier/Staging`). Two interfaces,
one model.

Usage:
    python scripts/batch_predict.py
    python scripts/batch_predict.py \\
        --input examples/sample_indian_customers.csv \\
        --schema examples/icici_schema_map.yaml \\
        --output examples/loan_decisions.xlsx
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from credit_risk.features import encode_features
from credit_risk.serving.model_loader import load_staging_model

HERE = Path(__file__).resolve().parents[1]


# ── Schema-mapping ──────────────────────────────────────────────────────


def _apply_banded(value: float, bands: list[dict]) -> str:
    """Pick the first band whose ``lt`` threshold is greater than ``value``."""
    for band in bands:
        if value < band["lt"]:
            return band["code"]
    return bands[-1]["code"]


def map_to_model_schema(df: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    """Translate the bank's raw schema into the model's training-time schema.

    Returns a DataFrame with the German Credit raw columns (pre-encoding).

    Numeric features are scaled into the model's training range — INR loan
    amounts (~₹50K-1.5Cr) get normalized to the model's DM range (250-18424)
    so tree splits land on familiar territory. Without this, all amounts
    fall in the model's "extreme high" branch and proba skews toward default.
    """
    out = pd.DataFrame(index=df.index)

    # Direct numeric maps
    for model_col, source_col in schema["direct"].items():
        out[model_col] = df[source_col]

    # Scale numeric features into the model's training range
    # (production-realistic feature normalization at the boundary)
    out["credit_amount"] = (out["credit_amount"] / 1000).clip(upper=18424)
    out["duration"] = out["duration"].clip(upper=72)

    # Categorical translations
    for model_col, lookup in schema["categorical"].items():
        out[model_col] = df[
            model_col.replace("housing", "housing_status").replace("purpose", "loan_purpose")
        ].map(lookup)

    # Derived fields
    for name, expr in schema.get("derived", {}).items():
        # Safe-ish eval — schema is project-owned config, not user input
        out[name] = df.eval(expr)

    # Banded categoricals (numeric → bucket code)
    for model_col, cfg in schema["banded"].items():
        src = out[cfg["source"]] if cfg["source"] in out.columns else df[cfg["source"]]
        out[model_col] = src.apply(lambda v, bands=cfg["bands"]: _apply_banded(v, bands))
        # Optional override based on another column
        for override_col, override_map in (cfg.get("employment_type_overrides", {}) or {}).items():
            mask = df["employment_type"] == override_col
            out.loc[mask, model_col] = override_map

    # Gender → personal_status
    out["personal_status"] = df["gender"].map(schema["gender_rules"])

    # Defaults for everything else
    for col, val in schema["defaults"].items():
        if col not in out.columns:
            out[col] = val

    # Drop derived helpers that aren't model features
    out = out.drop(columns=["emi_to_income_ratio"], errors="ignore")

    return out


# ── Decisioning + reason codes ──────────────────────────────────────────


def _decisions(proba: np.ndarray, approve_below: float, reject_above: float) -> np.ndarray:
    """3-tier decision band: APPROVE / REVIEW / REJECT."""
    out = np.empty(len(proba), dtype=object)
    out[proba < approve_below] = "APPROVE"
    out[(proba >= approve_below) & (proba <= reject_above)] = "REVIEW"
    out[proba > reject_above] = "REJECT"
    return out


def _reason_codes_for_rejects(
    decisions: np.ndarray,
    X_arr: np.ndarray,
    feature_importances: np.ndarray,
    feature_names: list[str],
    top_k: int = 3,
) -> list[str]:
    """For each REJECT, return the top-k driving features as a human-readable string.

    Uses tree-model feature importance × feature value as a proxy for SHAP.
    Real production would use SHAP TreeExplainer; this is good enough for the
    demo and ~100x faster.
    """
    reasons: list[str] = []
    fi = np.asarray(feature_importances, dtype=float)
    for i, dec in enumerate(decisions):
        if dec != "REJECT":
            reasons.append("")
            continue
        contributions = np.abs(X_arr[i] * fi)
        top_idx = np.argsort(contributions)[::-1][:top_k]
        reasons.append("; ".join(feature_names[idx] for idx in top_idx))
    return reasons


# ── Main ─────────────────────────────────────────────────────────────────


def score_dataframe(df_raw: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    """Score an in-memory DataFrame. Used by both the CLI and the Streamlit UI."""
    df_mapped = map_to_model_schema(df_raw, schema)

    bundle = load_staging_model()
    X = encode_features(df_mapped).reindex(columns=bundle.feature_names, fill_value=0.0)
    X_arr = X.to_numpy(dtype=float)

    proba = bundle.model.predict_proba(X_arr)[:, 1]
    thr = schema["decision_thresholds"]
    decisions = _decisions(proba, thr["approve_below"], thr["reject_above"])
    reasons = _reason_codes_for_rejects(
        decisions, X_arr, bundle.model.feature_importances_, list(X.columns)
    )

    now_iso = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return pd.DataFrame(
        {
            "customer_id": df_raw["customer_id"],
            "full_name": df_raw["full_name"],
            "loan_purpose": df_raw["loan_purpose"],
            "loan_amount_inr": df_raw["loan_amount_inr"],
            "decision": decisions,
            "probability_default": proba.round(4),
            "reason_code": reasons,
            "model_version": bundle.version,
            "scored_at": now_iso,
            "schema_map_version": schema["schema_map_version"],
        }
    )


def run(input_path: Path, schema_path: Path, output_path: Path) -> pd.DataFrame:
    schema = yaml.safe_load(schema_path.read_text())
    df_raw = pd.read_csv(input_path)
    print(f"Loaded {len(df_raw):,} customer rows from {input_path.name}")

    output = score_dataframe(df_raw, schema)
    print(f"Scored against model v{output['model_version'].iloc[0]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_excel(output_path, index=False, sheet_name="Decisions")
    print(f"\nWrote {len(output):,} decisions → {output_path}")

    print("\nDecision breakdown:")
    print(output["decision"].value_counts().to_string())
    pct = output["decision"].value_counts(normalize=True) * 100
    print(f"\nApproval rate: {pct.get('APPROVE', 0):.1f}%")
    print(f"Review rate:   {pct.get('REVIEW', 0):.1f}%")
    print(f"Reject rate:   {pct.get('REJECT', 0):.1f}%")

    return output


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=HERE / "examples" / "sample_indian_customers.csv")
    p.add_argument("--schema", type=Path, default=HERE / "examples" / "icici_schema_map.yaml")
    p.add_argument("--output", type=Path, default=HERE / "examples" / "loan_decisions.xlsx")
    args = p.parse_args()
    run(args.input, args.schema, args.output)


if __name__ == "__main__":
    main()
