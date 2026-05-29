# STATUS

Single source of truth for "where am I right now?" Updated at the end of every session.

## Current

- **Week:** 1 of 26 — ✅ COMPLETE
- **Project:** project-0-warmup — ✅ shipped
- **Infrastructure:** 18 / 18 verify checks passing
- **First MLflow run:** `credit-baseline / warmup-randomforest` (acc 0.752, auc 0.788)
- **Blockers:** none

## Week 1 milestones (all green)

- [x] Day 1 — Toolchain installed (uv, direnv, Ollama with GPU verified)
- [x] Day 2 — k3d-mlops cluster created + smoke-tested
- [x] Day 3 — Docker Compose stack live (MLflow + MinIO + Postgres + Redis)
- [x] Day 4 — Monorepo skeleton + pre-commit + README + GitHub push
- [x] Day 5 — sklearn baseline notebook + tested module + MLflow run logged

## Up next — Week 2 begins Project 1

**Project 1: Self-Healing Credit-Risk Pipeline** (Fintech, weeks 2-6)

The warmup module + notebook becomes the *seed* — Project 1 wraps it with
DVC for data versioning, switches to XGBoost, adds Evidently AI for drift
detection, GitHub Actions retraining workflow, FastAPI serving with canary
deployment, and Prometheus/Grafana monitoring.

Next session command: ask Claude to "plan Project 1" — invokes writing-plans
skill for a 5-week implementation plan, then we execute task-by-task.

## Memory

Conversational context lives at:
`~/.claude/projects/-home-himanshu-learning-mlops-journey/memory/`

`MEMORY.md` is the index. Future Claude sessions read it first.

## Surprises / lessons logged so far

See `memory/project_progress.md` "Day N surprise lessons" sections — these are
the interview-grade gotchas worth re-reading before any platform-engineering
interview.
