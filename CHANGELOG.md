# Changelog

All notable shipped work on this platform. Newest first.

## v0.2.5 — Batch decisioning + Streamlit demo UI

**Released 2026-06-04.**

- Batch loan-decisioning pipeline: Indian-bank CSV → Excel with audit columns
- Streamlit demo UI at port 8501, fronts the same `models:/credit-risk-classifier/Staging`
- YAML-driven schema mapping (Indian-bank fields → model training schema)
- 3-band decisions: APPROVE / REVIEW / REJECT with reason codes per REJECT
- 2,000-row synthetic Indian customer dataset (Faker `en_IN`, PAN format, INR amounts)
- Feature-importance-based reason codes (proxy for SHAP, ~100x faster)
- Audit columns per decision: `model_version`, `schema_map_version`, `scored_at`
- New `make p2-demo-data` / `p2-batch` / `p2-streamlit` shortcuts

## v0.2.0 — Serving + Docker image (Project 1)

**Released 2026-06-04.**

- `credit-risk-api` FastAPI service with `/predict`, `/healthz`, `/readyz`
- Pydantic v2 schemas at the request boundary (422 on malformed input)
- Lifespan-managed model load from MLflow Registry (`models:/credit-risk-classifier/Staging`)
- Multi-stage Dockerfile → 1.71 GB slim image, non-root, container healthcheck
- Public Docker image: [`ghcr.io/himanshunigam-456/credit-risk-api`](https://github.com/himanshunigam-456/Mlops-journey/pkgs/container/credit-risk-api)
- GitHub Actions auto-publish on every push to `main` (build cache via GHA)
- Locust load test: **p50=14ms · p95=25ms · p99=35ms · 30 RPS · 0 failures**
- 33/33 tests across data loader, features, trainer, registry, schemas, app, smoke
- Registry hardening: test-name isolation + `archive_existing_versions=True` invariant

## v0.1.0 — Training pipeline + Model Registry (Project 1)

**Released 2026-06-02.**

- DVC-tracked UCI German Credit dataset on MinIO S3 remote
- XGBoost trainer with 3-trial hyperparameter sweep + 5 metrics
- MLflow Model Registry integration with `Staging` promotion
- Typer CLI: `credit-risk train` + `credit-risk register`
- Model signature + input example logged at training time
- Best model promoted to `credit-risk-classifier` v6 @ `Staging` (`roc_auc=0.8005`)

## v0.0.1 — Platform foundation

**Released 2026-05-27.**

- Docker Compose managed-service tier: MLflow + MinIO + Postgres + Redis
- k3d workload-tier cluster (1 server + 2 agents)
- Ollama GPU LLM inference with `llama3.1:8b` + `nomic-embed-text`
- Monorepo with uv workspace, pre-commit hooks, gitleaks
- CI: ruff + pytest on every push
- 18-probe `verify.sh` smoke test (always 18/18 in published releases)
