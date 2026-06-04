"""Encode a raw German Credit row and POST it to the running API.

Demonstrates the boundary between the upstream feature-engineering layer
(this script's job) and the serving layer (the FastAPI app's job).
In production, encoding lives in a separate feature pipeline, NOT inside
the model server.

Usage:
    # Default — uses the first row of the training set + local API
    python examples/encode_and_call.py

    # Custom API URL (e.g. for the Dockerized container later)
    python examples/encode_and_call.py http://localhost:8000

    # Run the API first (in another terminal):
    uvicorn credit_risk.serving.app:app --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
from credit_risk.data_loader import load_german_credit
from credit_risk.features import encode_features

DEFAULT_URL = "http://localhost:8000"
REPO_DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "german_credit.csv"


def main(url: str = DEFAULT_URL) -> None:
    df = load_german_credit(REPO_DATA)
    raw_row = df.drop(columns=["default"]).iloc[[0]]

    encoded = encode_features(raw_row)
    features = {k: float(v) for k, v in encoded.iloc[0].to_dict().items()}

    payload = {"features": features, "request_id": "example-1"}
    print(f"POST {url}/predict  ({len(features)} encoded features)")

    r = httpx.post(f"{url}/predict", json=payload, timeout=10.0)
    r.raise_for_status()

    resp = r.json()
    print()
    print(
        f"  prediction          = {resp['prediction']}  "
        f"({'DEFAULT' if resp['prediction'] == 1 else 'non-default'})"
    )
    print(f"  probability_default = {resp['probability_default']:.4f}")
    print(f"  model_version       = {resp['model_version']}")
    print(f"  request_id          = {resp['request_id']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL)
