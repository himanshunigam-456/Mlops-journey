"""Locust load test for the credit-risk API.

Run (with the API on :8000):
    locust -f serving/locustfile.py --headless \\
        --users 10 --spawn-rate 2 --run-time 30s \\
        --host http://localhost:8000

In a real production load test, you'd:
- Replay actual traffic instead of synthetic (recorded from prod logs)
- Mix endpoint distribution (e.g., 99% /predict, 1% /healthz)
- Vary feature payloads to defeat any caching the model server adds
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

# Tiny synthetic feature dict — already-encoded space.
# Real prod load test would replay anonymized traffic samples.
SAMPLE_FEATURES = {
    "duration": 12.0,
    "credit_amount": 1500.0,
    "installment_commitment": 4.0,
    "residence_since": 4.0,
    "age": 35.0,
    "existing_credits": 1.0,
    "num_dependents": 1.0,
}


class CreditRiskUser(HttpUser):
    """Simulates a client hitting /predict with jittered features."""

    # Each simulated user pauses 0.1-0.5s between requests — realistic for
    # a human-driven client; for service-to-service traffic, set to 0.
    wait_time = between(0.1, 0.5)

    @task(10)  # weight 10 — 91% of requests are predict
    def predict(self):
        features = {k: v * random.uniform(0.8, 1.2) for k, v in SAMPLE_FEATURES.items()}
        self.client.post(
            "/predict",
            json={"features": features, "request_id": f"locust-{id(self)}"},
            timeout=5.0,
            name="/predict",  # group all predict variations under one name in stats
        )

    @task(1)  # weight 1 — 9% are healthz (mimics a load balancer probe)
    def healthz(self):
        self.client.get("/healthz", name="/healthz")
