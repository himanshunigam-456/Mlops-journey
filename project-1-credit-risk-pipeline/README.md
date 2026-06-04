# credit-risk-api

Production-grade credit-risk classification service. Trains an XGBoost model on the UCI German Credit dataset, registers it in MLflow, serves it via a Dockerized FastAPI app, and auto-publishes to GHCR.

**Image:** [`ghcr.io/himanshunigam-456/credit-risk-api:latest`](https://github.com/himanshunigam-456/Mlops-journey/pkgs/container/credit-risk-api)

## Shipped

### Training pipeline
- DVC-tracked dataset on MinIO (S3 remote)
- XGBoost trainer with 3-trial hyperparameter sweep
- 5 metrics logged per run (accuracy · precision · recall · f1 · roc_auc)
- MLflow signature + input example baked into the artifact
- Typer CLI: `train` + `register` subcommands

```bash
make p1-train      # 3-trial sweep, log to MLflow
make p1-register   # promote best run to Registry @ Staging
```

### Serving + Dockerization
- Pydantic-validated `POST /predict` (rejects bad input at the boundary with 422)
- `/healthz` and `/readyz` separated (K8s liveness vs readiness semantics)
- Model loaded once at startup via FastAPI `lifespan` from the Registry
- Multi-stage Dockerfile → slim non-root image (1.71 GB) with healthcheck
- Locust load test: **p50=14ms · p95=25ms · p99=35ms · 30 RPS · 0 failures**
- GitHub Actions auto-publishes to GHCR on every push to `main`

```bash
make p2-serve            # local uvicorn on :8000 (auto-reload)
make p2-docker-build     # build image as credit-risk-api:dev
make p2-docker-run       # run container against compose stack
make p2-load-test        # 30s Locust scenario (10 users)
make p2-test             # serving tests only
```

Or pull and run the public image directly:

```bash
docker pull ghcr.io/himanshunigam-456/credit-risk-api:latest
docker run --rm -p 8000:8000 ghcr.io/himanshunigam-456/credit-risk-api:latest
```

Then:
- Interactive API docs: http://localhost:8000/docs
- Alternative API docs: http://localhost:8000/redoc

### Batch decisioning + Streamlit demo

![BlueLeaf Bank — Loan Decision Service](../docs/screenshots/streamlit-demo.png)

A second consumer of the same model — a batch UI for analysts. Upload an Indian-bank-formatted customer CSV (2,000 synthetic rows included), get back **APPROVE / REVIEW / REJECT** decisions with reason codes per rejection and an audit trail. Same model, two interfaces:

- **Real-time:** FastAPI `/predict` for software clients (the public Docker image above)
- **Batch:** Streamlit UI + `batch_predict.py` for analysts (this section)

```bash
make p2-streamlit          # opens http://localhost:8501
make p2-batch              # CLI: writes examples/loan_decisions.xlsx
make p2-demo-data          # regenerates the 2,000-row synthetic CSV
```

**Schema mapping in YAML:** the bank's schema (PAN, monthly_income_inr, employment_type, ...) is translated to the model's training schema via [`examples/icici_schema_map.yaml`](examples/icici_schema_map.yaml). Onboarding a new bank means editing the YAML, not the code.

**Decision bands** (also configurable in YAML):
- `APPROVE` — probability of default < 0.25
- `REVIEW` — between 0.25 and 0.50 (manual underwriter)
- `REJECT` — > 0.50, with top-3 feature-importance-based reason codes

**Output Excel columns:** `customer_id · full_name · loan_purpose · loan_amount_inr · decision · probability_default · reason_code · model_version · scored_at · schema_map_version`.

> All data is synthetic; the model is trained on the open UCI German Credit dataset for reproducibility.

## Roadmap

- **Drift detection** — Evidently AI feature/target drift reports + auto-retrain on drift signal
- **Canary deploy + Prometheus** — gradient rollout with metric-driven rollback
- **Production demo** — AWS EC2 deployment + Streamlit UI for the portfolio

## Tests

```bash
make p1-test    # all 33 tests
```

33 tests across data loader · features · trainer · registry · schemas · app · live smoke.
