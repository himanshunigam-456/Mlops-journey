# project-1-credit-risk-pipeline

**Domain:** Fintech · **Phase:** Week 2-6 of the MLOps journey.

A production-grade credit-risk classification pipeline. Trains an XGBoost
model on the UCI German Credit dataset, registers the best model in MLflow,
serves predictions via a Dockerized FastAPI service, detects drift in
production traffic, and auto-retrains via GitHub Actions.

## Phase 1 (Week 2): Data + XGBoost + Registry ✅

- DVC-tracked raw data in MinIO
- XGBoost trainer with hyperparameter sweep + 5 metrics + MLflow signature
- Best model promoted to MLflow Model Registry @ `Staging`

```bash
make p1-train      # train + log to MLflow (3 trials)
make p1-register   # promote best run to Model Registry @ Staging
```

## Phase 2 (Week 3): FastAPI Serving + Dockerization ✅

- Pydantic-validated `/predict` endpoint (rejects bad input at the boundary with 422)
- `/healthz` + `/readyz` separation (K8s livenessProbe vs readinessProbe semantics)
- Model loaded once at startup via FastAPI `lifespan` from `models:/credit-risk-classifier/Staging`
- Multi-stage Dockerfile → slim non-root image (1.71 GB) with container healthcheck
- Locust load test: **p50=14ms · p95=25ms · p99=35ms · 30 RPS · 0 failures**
- GitHub Actions auto-publishes to GHCR on every push to `main`
- Public image: [`ghcr.io/himanshunigam-456/credit-risk-api`](https://github.com/himanshunigam-456/Mlops-journey/pkgs/container/credit-risk-api)

```bash
make p2-serve            # local uvicorn on :8000 (auto-reload)
make p2-docker-build     # build image as credit-risk-api:dev
make p2-docker-run       # run container against compose stack
make p2-load-test        # 30s Locust scenario (10 users)
make p2-test             # run only the serving tests
```

Or pull the public image directly:

```bash
docker pull ghcr.io/himanshunigam-456/credit-risk-api:latest
docker run --rm -p 8000:8000 ghcr.io/himanshunigam-456/credit-risk-api:latest
```

Then visit:
- Interactive API docs: http://localhost:8000/docs (Swagger UI)
- Alternative docs: http://localhost:8000/redoc

## Phases (planned)

- Phase 3 (Week 4): Evidently AI drift detection + auto-retrain
- Phase 4 (Week 5): Canary deploy + Prometheus monitoring
- Phase 5 (Week 6): AWS EC2 production demo + Streamlit UI

## Tests

```bash
make p1-test    # all 33 project-1 tests
```

33 tests: 4 data_loader · 6 features · 6 train · 2 registry · 8 schemas · 5 app · 2 smoke.
