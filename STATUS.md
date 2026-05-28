# STATUS

Single source of truth for "where am I right now?" Updated at the end of every session.

## Current

- **Week:** 1 of 26
- **Day:** Day 4 / 5 — in progress
- **Project:** project-0-warmup (Week 1 foundations)
- **Infrastructure:** 18 / 18 verify checks passing
- **Blockers:** none
- **Last commit:** `git log -1 --oneline`

## Week 1 milestones (Mon–Fri)

- [x] Day 1 — Toolchain installed (uv, direnv, Ollama with GPU verified)
- [x] Day 2 — k3d-mlops cluster created + smoke-tested
- [x] Day 3 — Docker Compose stack live (MLflow + MinIO + Postgres + Redis)
- [x] Day 4 — Monorepo skeleton + Makefile + pre-commit + README + GitHub push
- [ ] Day 5 — sklearn baseline notebook + tested module + MLflow run logged

## Up next

**Day 5:** sklearn baseline. Build a TDD'd `train_baseline()` function in
`project-0-warmup/src/credit_baseline/pipeline.py`, then a Jupyter notebook
that loads the German Credit dataset, calls it, and logs the run to MLflow.

End-of-Friday gate: `make demo` runs end-to-end and an experiment shows up
in the MLflow UI with the model stored in MinIO.

After that: **Week 2 starts Project 1** (Self-Healing Credit-Risk Pipeline).

## Memory

Conversational context lives at:
`~/.claude/projects/-home-himanshu-learning-mlops-journey/memory/`

`MEMORY.md` is the index. Future Claude sessions read it first.

## Surprises / lessons logged so far

See `memory/project_progress.md` "Day N surprise lessons" sections — these are
the interview-grade gotchas worth re-reading before any platform-engineering
interview.
