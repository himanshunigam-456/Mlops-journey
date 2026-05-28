# project-0-warmup

Week-1 warmup. Loads the German Credit dataset, trains a baseline RandomForest,
logs the run to MLflow, and exposes a `train_baseline()` function that Project 1
will extend into a full self-healing pipeline.

## Run

```bash
make demo    # at repo root — boots stack + executes the notebook
```

## Why this exists

A vertical slice from raw data → trained model → MLflow-tracked run, in ~150 lines.
Confirms the local MLOps environment is fully wired before Project 1's complexity.
