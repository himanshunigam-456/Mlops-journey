"""Build the training reference Parquet — drift's "left side" comparison snapshot.

Loads the German Credit training set, runs it through the Staging model, and
uploads the result to `s3://reports/reference_v{N}.parquet` matching the
production capture schema (feature_* columns + prediction + probability_default).

Run once per model version. The output stays static until the next retrain.

Usage:
    python scripts/build_reference.py
    python scripts/build_reference.py --data /path/to/custom.csv
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
from credit_risk.data_loader import load_german_credit
from credit_risk.drift.store import REPORTS_BUCKET, _s3_client, ensure_bucket
from credit_risk.features import encode_features
from credit_risk.serving.model_loader import load_staging_model

HERE = Path(__file__).resolve().parents[1]
DEFAULT_DATA = HERE / "data" / "raw" / "german_credit.csv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = p.parse_args()

    df = load_german_credit(args.data)
    print(f"Loaded {len(df):,} training rows from {args.data.name}")

    bundle = load_staging_model()
    print(f"Scoring against credit-risk-classifier v{bundle.version}")

    # Encode + reindex to match the model's training-time column set.
    X = encode_features(df.drop(columns=["default"])).reindex(
        columns=bundle.feature_names, fill_value=0.0
    )
    X_arr = X.to_numpy(dtype=float)

    proba = bundle.model.predict_proba(X_arr)[:, 1]
    pred = bundle.model.predict(X_arr)

    # Build the flat row schema — matching capture.py:to_dataframe() exactly.
    # Evidently can only compare columns present in BOTH reference + current.
    flat = pd.DataFrame(
        {
            "request_id": [f"ref-{i}" for i in range(len(X))],
            "prediction": pred.astype(int),
            "probability_default": proba.round(4),
            "model_version": str(bundle.version),
            "scored_at": "REFERENCE",
        }
    )
    for col in X.columns:
        flat[f"feature_{col}"] = X[col].astype(float).values

    print(f"Reference schema: {flat.shape[0]:,} rows × {flat.shape[1]} cols")

    # Upload to MinIO under the model's version (one reference per model).
    ensure_bucket(REPORTS_BUCKET)
    key = f"reference_v{bundle.version}.parquet"
    buf = io.BytesIO()
    flat.to_parquet(buf, index=False)
    buf.seek(0)
    s3 = _s3_client()
    s3.put_object(Bucket=REPORTS_BUCKET, Key=key, Body=buf.read())

    print(f"\nWrote reference snapshot → s3://{REPORTS_BUCKET}/{key}")
    print("\nQuick stats (first 5 feature columns):")
    feature_cols = [c for c in flat.columns if c.startswith("feature_")][:5]
    print(flat[feature_cols].describe().T[["mean", "std", "min", "max"]].round(2))


if __name__ == "__main__":
    main()
