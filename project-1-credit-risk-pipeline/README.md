# project-1-credit-risk-pipeline

**Domain:** Fintech · **Phase:** Week 2-6 of the MLOps journey.

A self-healing credit-risk classification pipeline. Trains an XGBoost model
on the UCI German Credit dataset, registers the best model in MLflow,
detects drift in production traffic, and auto-retrains via GitHub Actions.

## Phase 1 (Week 2): Data + XGBoost + Registry

- DVC-tracked raw data in MinIO
- XGBoost trainer with hyperparameter sweep
- Best model promoted to MLflow Model Registry @ `Staging`

## Run

```bash
make p1-train      # train + log to MLflow
make p1-register   # promote best run to Model Registry
```

## Phases (planned)

- Phase 2 (Week 3): FastAPI serving + Dockerization
- Phase 3 (Week 4): Evidently AI drift detection + auto-retrain
- Phase 4 (Week 5): Canary deploy + Prometheus monitoring
- Phase 5 (Week 6): AWS EC2 production demo + portfolio polish
