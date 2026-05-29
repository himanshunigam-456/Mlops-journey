# Data

## Raw

| File | Source | Rows | Target | DVC-tracked |
|------|--------|------|--------|-------------|
| `raw/german_credit.csv` | [UCI Statlog](https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/) | 1000 | `default` (1=bad credit recoded to 1=default) | ✅ |

20 mixed-type features (numeric + categorical). See [src/credit_risk/data_loader.py](../src/credit_risk/data_loader.py)
for the schema and [src/credit_risk/features.py](../src/credit_risk/features.py) for preprocessing.

## Reproduce on a fresh checkout

```bash
cd project-1-credit-risk-pipeline
dvc pull              # fetches data/raw/german_credit.csv from MinIO
```

`dvc pull` reads the MD5 hash from `data/raw/german_credit.csv.dvc` and pulls
exactly that content from the `datasets/project-1` bucket. Reproducibility is
guaranteed at the byte level.
