# STATUS

Single source of truth for "where am I right now?" Updated at the end of every session.

## Current

- **Week:** 2 of 26
- **Project:** project-1-credit-risk-pipeline — Phase 1 ✅ shipped
- **Infrastructure:** 18 / 18 verify checks passing
- **Model Registry:** `credit-risk-classifier` v5 @ `Staging` (best `roc_auc` 0.8005 from 3-trial sweep)
- **Tests:** 17 / 17 passing (4 data_loader · 6 features · 5 train · 2 registry)
- **Blockers:** none

## Week 1 milestones (all green)

- [x] Day 1 — Toolchain installed (uv, direnv, Ollama with GPU verified)
- [x] Day 2 — k3d-mlops cluster created + smoke-tested
- [x] Day 3 — Docker Compose stack live (MLflow + MinIO + Postgres + Redis)
- [x] Day 4 — Monorepo skeleton + pre-commit + README + GitHub push
- [x] Day 5 — sklearn baseline notebook + tested module + MLflow run logged

## Project 1 — Phases (5 weeks)

- [x] Phase 1 (Week 2) — Data + XGBoost + MLflow Model Registry
- [ ] Phase 2 (Week 3) — FastAPI serving + Dockerization
- [ ] Phase 3 (Week 4) — Evidently AI drift detection + auto-retrain
- [ ] Phase 4 (Week 5) — Canary deploy + Prometheus monitoring
- [ ] Phase 5 (Week 6) — AWS EC2 production demo + portfolio polish

## Phase 1 deliverables (live on `main`)

| Artifact | Path / Location |
|----------|-----------------|
| DVC-tracked dataset | `project-1-credit-risk-pipeline/data/raw/german_credit.csv.dvc` → MinIO bucket `datasets/` |
| Trainer module | `src/credit_risk/train.py` (XGBoost + 5 metrics) |
| Feature engineering | `src/credit_risk/features.py` (stratified split + one-hot) |
| Data loader | `src/credit_risk/data_loader.py` (UCI schema + target recode) |
| Registry helper | `src/credit_risk/registry.py` (`register_model_from_run`) |
| Typer CLI | `src/credit_risk/cli.py` — `train` + `register` subcommands |
| Makefile shortcuts | `p1-data` · `p1-train` · `p1-register` · `p1-test` |
| Registered model | `credit-risk-classifier` v5 @ `Staging` (MLflow Registry) |
| Tests | 17 / 17 green |

## Up next — Phase 2 (Week 3)

**FastAPI serving + Dockerization**

The `Staging` model artifact (`models:/credit-risk-classifier/Staging`) becomes
the input contract for a FastAPI app: `/predict` endpoint with Pydantic schema,
Dockerized image pushed to GHCR, smoke + load tests wired into CI.

Next session command: ask Claude to *"plan project 1 phase 2"* — invokes the
writing-plans skill for a 5-day Phase 2 plan, then we execute task-by-task.

## Memory

Conversational context lives at:
`~/.claude/projects/-home-himanshu-learning-mlops-journey/memory/`

`MEMORY.md` is the index. Future Claude sessions read it first.

## Surprises / lessons logged so far

See `memory/project_progress.md` "Day N surprise lessons" sections — these are
the interview-grade gotchas worth re-reading before any platform-engineering
interview. Also see `memory/reference_interview_soundbites.md` for the
consolidated cheat sheet.
