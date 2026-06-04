# STATUS

Single source of truth for "where am I right now?" Updated at the end of every session.

## Current

- **Week:** 3 of 26
- **Project:** project-1-credit-risk-pipeline — Phase 2 ✅ shipped
- **Infrastructure:** 18 / 18 verify checks passing (survived cold restart)
- **Model Registry:** `credit-risk-classifier` v6 @ `Staging` (with MLflow signature + input_example)
- **Public image:** [`ghcr.io/himanshunigam-456/credit-risk-api:latest`](https://github.com/himanshunigam-456/Mlops-journey/pkgs/container/credit-risk-api) (pullable globally, no auth)
- **Measured SLA:** p50=14ms · p95=25ms · p99=35ms · 30 RPS sustained · 0 failures across 873 requests
- **Tests:** 33 / 33 passing (4 data_loader · 6 features · 6 train · 2 registry · 8 schemas · 5 app · 2 smoke)
- **Blockers:** none

## Week 1 milestones (all green)

- [x] Day 1 — Toolchain installed (uv, direnv, Ollama with GPU verified)
- [x] Day 2 — k3d-mlops cluster created + smoke-tested
- [x] Day 3 — Docker Compose stack live (MLflow + MinIO + Postgres + Redis)
- [x] Day 4 — Monorepo skeleton + pre-commit + README + GitHub push
- [x] Day 5 — sklearn baseline notebook + tested module + MLflow run logged

## Project 1 — Phases (5 weeks)

- [x] Phase 1 (Week 2) — Data + XGBoost + MLflow Model Registry
- [x] Phase 2 (Week 3) — FastAPI serving + Dockerization + GHCR auto-publish
- [ ] Phase 3 (Week 4) — Evidently AI drift detection + auto-retrain
- [ ] Phase 4 (Week 5) — Canary deploy + Prometheus monitoring
- [ ] Phase 5 (Week 6) — AWS EC2 production demo + Streamlit UI

## Phase 2 deliverables (live on `main`)

| Artifact | Path / Location |
|----------|-----------------|
| Pydantic schemas (request/response/health) | `src/credit_risk/serving/schemas.py` |
| Lifespan-managed model loader | `src/credit_risk/serving/model_loader.py` |
| FastAPI app — `/predict` `/healthz` `/readyz` | `src/credit_risk/serving/app.py` |
| Multi-stage Dockerfile (1.71 GB, non-root, healthcheck) | `serving/Dockerfile` |
| Locust load-test scenario | `serving/locustfile.py` |
| GHCR auto-publish workflow | `.github/workflows/docker-publish.yml` |
| Encode-and-call helper | `examples/encode_and_call.py` |
| Makefile shortcuts | `p2-serve` · `p2-docker-build` · `p2-docker-run` · `p2-docker-stop` · `p2-load-test` · `p2-test` |
| Registry hardening (test pollution fix) | `name=` param + `archive_existing_versions=True` |
| Public Docker image | `ghcr.io/himanshunigam-456/credit-risk-api` (tags: `latest`, `main`, `<sha>`) |

## Phase 1 deliverables (live on `main`)

| Artifact | Path / Location |
|----------|-----------------|
| DVC-tracked dataset | `data/raw/german_credit.csv.dvc` → MinIO bucket `datasets/` |
| XGBoost trainer (5 metrics) | `src/credit_risk/train.py` |
| Feature engineering (stratified split + one-hot) | `src/credit_risk/features.py` |
| Registry helper | `src/credit_risk/registry.py` |
| Typer CLI | `src/credit_risk/cli.py` — `train` + `register` subcommands |
| Makefile shortcuts | `p1-data` · `p1-train` · `p1-register` · `p1-test` |
| Registered model | `credit-risk-classifier` v6 @ `Staging` (with signature) |

## Up next — Phase 3 (Week 4)

**Evidently AI drift detection + auto-retrain**

Phase 2's `/predict` endpoint now logs every request. Phase 3 wires
[Evidently AI](https://github.com/evidentlyai/evidently) to compute
feature/target drift between training data and production traffic, publish
an HTML drift report at `/drift`, and add a GitHub Actions cron that
auto-retrains on drift signal.

Next session command: ask Claude to *"plan project 1 phase 3"* — invokes
the writing-plans skill for the Week-4 plan.

## Memory

Conversational context lives at:
`~/.claude/projects/-home-himanshu-learning-mlops-journey/memory/`

`MEMORY.md` is the index. Future Claude sessions read it first.

## Surprises / lessons logged so far

See `memory/project_progress.md` "Day N surprise lessons" sections + the
consolidated cheat sheet at `memory/reference_interview_soundbites.md`.
