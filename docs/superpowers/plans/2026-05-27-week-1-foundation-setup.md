# Week 1 Foundation Setup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get Himanshu's laptop ready to build all 5 portfolio projects: install toolchain, run local K3s + Docker stack, verify Ollama GPU acceleration, create the monorepo skeleton, and ship a baseline ML notebook to prove the full path (data → model → MLflow → tested module) works end-to-end.

**Architecture:** Local-first development on Ubuntu 22.04 with K3s as the lightweight K8s, Docker Compose for stateful services (MLflow, Postgres, MinIO, Redis), Ollama for local LLM inference (GPU-accelerated on GTX 1650), and uv for Python package management. The Week 1 artifact is a *configured, smoke-tested development environment* + a working sklearn baseline that becomes the seed for Project 1.

**Tech Stack:** uv · Docker + Docker Compose · K3s · Helm · kubectl · Ollama (CUDA) · MLflow · MinIO · PostgreSQL · Redis · scikit-learn · Jupyter · pytest · pre-commit

**Time budget:** 10 hrs across 5 weekday sessions (Mon-Fri, 2 hrs each).

**Repo:** `/home/himanshu/learning/mlops-journey/`

---

## Daily Plan At-A-Glance

| Day | 2-hr session focus | Tasks |
|-----|--------------------|-------|
| Mon | Python toolchain + Ollama + GPU verification | T1-T5 |
| Tue | K3s + Helm + smoke test cluster | T6-T9 |
| Wed | Docker Compose stack (MLflow/Postgres/MinIO/Redis) | T10-T13 |
| Thu | Repo skeleton + Makefile + pre-commit + README + STATUS | T14-T18 |
| Fri | sklearn baseline notebook + tested module + push | T19-T23 |

---

## File Structure Created This Week

```
mlops-journey/
├── .envrc                                 # direnv: env vars per-project
├── .gitignore                             # extended
├── .pre-commit-config.yaml                # ruff, black, secret-detect hooks
├── Makefile                               # common commands (make verify, make demo)
├── README.md                              # portfolio landing page
├── STATUS.md                              # week/project tracker
├── pyproject.toml                         # uv workspace root
├── infra/
│   ├── docker-compose.yml                 # MLflow + Postgres + MinIO + Redis
│   ├── k3s-install.sh                     # K3s installer wrapper
│   ├── ollama-pull-models.sh              # pulls Llama 3.1 8B + nomic embed
│   ├── verify.sh                          # smoke-test all services
│   └── hello-world.yaml                   # K3s smoke-test workload
├── project-0-warmup/
│   ├── pyproject.toml                     # project-local uv config
│   ├── notebooks/
│   │   └── credit_baseline.ipynb          # sklearn end-to-end refresher
│   ├── src/credit_baseline/
│   │   ├── __init__.py
│   │   └── pipeline.py                    # tiny baseline module (seeds Project 1)
│   └── tests/
│       └── test_pipeline.py               # smoke test
└── docs/
    └── superpowers/
        ├── specs/2026-05-27-mlops-agentic-ai-learning-path-design.md  # exists
        └── plans/2026-05-27-week-1-foundation-setup.md                # this file
```

---

# DAY 1 (MON) — Python Toolchain + Ollama GPU Verification

## Task 1: Verify and capture system prerequisites

**Files:**
- Create: `infra/verify.sh`

- [ ] **Step 1: Create infra directory and stub verify script**

```bash
mkdir -p /home/himanshu/learning/mlops-journey/infra
cd /home/himanshu/learning/mlops-journey
```

- [ ] **Step 2: Write `infra/verify.sh`**

Create file `infra/verify.sh` with:

```bash
#!/usr/bin/env bash
# Smoke-test all Week 1 infra. Re-run any time you suspect environment rot.
set -uo pipefail

pass=0
fail=0

check() {
  local name="$1"; shift
  if "$@" >/dev/null 2>&1; then
    printf "  ✅ %s\n" "$name"
    pass=$((pass+1))
  else
    printf "  ❌ %s\n" "$name"
    fail=$((fail+1))
  fi
}

echo "── Toolchain ──"
check "docker"        docker --version
check "docker compose" docker compose version
check "uv"            uv --version
check "direnv"        direnv --version
check "kubectl"       kubectl version --client
check "helm"          helm version --short
check "k3s"           k3s --version
check "ollama"        ollama --version
check "nvidia-smi"    nvidia-smi

echo ""
echo "── Services ──"
check "docker daemon"     docker info
check "k3s nodes ready"   bash -c "kubectl get nodes --no-headers 2>/dev/null | grep -q ' Ready'"
check "mlflow UI :5000"   curl -sf http://localhost:5000/health
check "minio   :9000"     curl -sf http://localhost:9000/minio/health/live
check "postgres :5432"    bash -c "echo > /dev/tcp/localhost/5432"
check "redis    :6379"    bash -c "echo > /dev/tcp/localhost/6379"
check "ollama   :11434"   curl -sf http://localhost:11434/api/tags

echo ""
echo "── Ollama models ──"
check "llama3.1:8b pulled"     bash -c "ollama list | grep -q llama3.1:8b"
check "nomic-embed-text pulled" bash -c "ollama list | grep -q nomic-embed-text"

echo ""
echo "Result: $pass passed, $fail failed"
exit $fail
```

- [ ] **Step 3: Make executable and run (expect many failures — that's fine)**

```bash
chmod +x infra/verify.sh
./infra/verify.sh
```

Expected: most checks fail (we haven't installed anything yet). `docker` likely passes, `nvidia-smi` should pass (his GTX 1650 is already there).

- [ ] **Step 4: Commit**

```bash
git add infra/verify.sh
git commit -m "chore: add infra verify smoke-test script"
```

---

## Task 2: Install uv (Python package manager) and direnv

**Files:**
- Modify: shell rc (~/.bashrc)

- [ ] **Step 1: Install uv via official installer**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then reload shell or `source ~/.bashrc`.

- [ ] **Step 2: Verify uv**

```bash
uv --version
```

Expected output: `uv 0.x.x` (any 0.x is fine).

- [ ] **Step 3: Install direnv**

```bash
sudo apt-get update && sudo apt-get install -y direnv
```

- [ ] **Step 4: Hook direnv into bash**

```bash
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
source ~/.bashrc
```

- [ ] **Step 5: Verify direnv**

```bash
direnv --version
```

Expected: `2.x.x`.

- [ ] **Step 6: Re-run verify script — uv & direnv should now pass**

```bash
./infra/verify.sh 2>&1 | grep -E "uv|direnv"
```

Expected: both show ✅.

---

## Task 3: Install NVIDIA Container Toolkit (so Docker/Ollama can use the GPU)

**Files:** none (system install)

- [ ] **Step 1: Verify GPU is visible at the OS level**

```bash
nvidia-smi
```

Expected: a table showing `GeForce GTX 1650` with VRAM `4096MiB`. If this fails, install drivers first:
```bash
sudo apt-get install -y nvidia-driver-535 && sudo reboot
```

- [ ] **Step 2: Add NVIDIA container toolkit repo**

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

- [ ] **Step 3: Install and configure**

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

- [ ] **Step 4: Verify Docker can see GPU**

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

Expected: the same `nvidia-smi` output as host. If it fails — fix before proceeding (Ollama will run CPU-only otherwise, which is too slow on a 4GB GPU machine).

---

## Task 4: Install Ollama and pull models

**Files:**
- Create: `infra/ollama-pull-models.sh`

- [ ] **Step 1: Install Ollama**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

The installer auto-detects the NVIDIA GPU and configures CUDA support.

- [ ] **Step 2: Verify Ollama service is running**

```bash
systemctl status ollama --no-pager | head -5
curl -sf http://localhost:11434/api/tags
```

Expected: service `active (running)`. Curl returns `{"models":[]}` (no models yet).

- [ ] **Step 3: Create model-pull script**

Create `infra/ollama-pull-models.sh`:

```bash
#!/usr/bin/env bash
# Pulls the local LLMs we need for Projects 2, 3, 5.
# All models picked to fit on a 4GB GPU (GTX 1650) via Q4 quantization.
set -euo pipefail

MODELS=(
  "llama3.1:8b"            # primary chat / agent model — ~4.7GB quantized
  "nomic-embed-text"       # embeddings — ~270MB, used by Project 3 RAG
  "phi3:mini"              # small fallback for fast dev iterations — ~2.3GB
)

for m in "${MODELS[@]}"; do
  echo "── Pulling $m ──"
  ollama pull "$m"
done

echo ""
echo "── All pulled ──"
ollama list
```

- [ ] **Step 4: Run it (this is a one-time download, ~7GB total)**

```bash
chmod +x infra/ollama-pull-models.sh
./infra/ollama-pull-models.sh
```

Expected: progress bars for each pull, then `ollama list` showing all three models.

- [ ] **Step 5: Verify GPU is actually being used**

In one terminal, run:
```bash
ollama run llama3.1:8b "Say hello in one sentence."
```

In another terminal **while the response is generating**:
```bash
nvidia-smi
```

Expected: a process named `ollama` shows up in `nvidia-smi`'s GPU process list with several hundred MB of VRAM used. **If no process appears, Ollama is running CPU-only — re-check Task 3.**

- [ ] **Step 6: Commit**

```bash
git add infra/ollama-pull-models.sh
git commit -m "feat(infra): add ollama model pull script (llama3.1 + nomic + phi3)"
```

---

## Task 5: End-of-Day-1 commit and STATUS update

**Files:** none new (commit only)

- [ ] **Step 1: Re-run verify script — confirm Day 1 progress**

```bash
./infra/verify.sh
```

Expected ✅ on: docker, docker compose, uv, direnv, ollama, nvidia-smi, docker daemon, ollama :11434, llama3.1:8b, nomic-embed-text. The remaining ❌ are Day 2+ tasks.

- [ ] **Step 2: Commit if anything outstanding (e.g., bashrc tweaks aren't tracked, that's fine)**

```bash
git status
# If clean, skip. If something tracked changed, commit:
git commit -am "chore: day-1 setup complete"
```

---

# DAY 2 (TUE) — K3s + Helm + kubectl

## Task 6: Install kubectl and helm

**Files:** none

- [ ] **Step 1: Install kubectl (official binary)**

```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
rm kubectl
kubectl version --client
```

Expected: client version printed; no error about missing server (yet).

- [ ] **Step 2: Install helm**

```bash
curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null
sudo apt-get install -y apt-transport-https
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
sudo apt-get update
sudo apt-get install -y helm
helm version --short
```

Expected: `v3.x.x+gXXXX`.

---

## Task 7: Install K3s and wire kubeconfig

**Files:**
- Create: `infra/k3s-install.sh`

- [ ] **Step 1: Create wrapper script**

Create `infra/k3s-install.sh`:

```bash
#!/usr/bin/env bash
# Installs K3s with sensible defaults for local MLOps dev.
# - disables traefik (we'll use plain NodePort / port-forward)
# - disables servicelb (saves ~150MB RAM)
# - writes kubeconfig readable by the current user
set -euo pipefail

curl -sfL https://get.k3s.io | sh -s - \
  --disable=traefik \
  --disable=servicelb \
  --write-kubeconfig-mode=644 \
  --write-kubeconfig=/etc/rancher/k3s/k3s.yaml

mkdir -p "$HOME/.kube"
sudo cp /etc/rancher/k3s/k3s.yaml "$HOME/.kube/config"
sudo chown "$(id -u):$(id -g)" "$HOME/.kube/config"

echo ""
echo "K3s installed. Verify:"
echo "  kubectl get nodes"
```

- [ ] **Step 2: Run it**

```bash
chmod +x infra/k3s-install.sh
./infra/k3s-install.sh
```

Expected: pulling images, then K3s service running. ~2-4 minutes.

- [ ] **Step 3: Wait for the node to be Ready**

```bash
kubectl get nodes -w
```

Wait until status shows `Ready`, then Ctrl-C.

- [ ] **Step 4: Verify low memory footprint**

```bash
free -h
```

Expected: total RAM used by K3s + system should be **under 4 GB**. If higher, something else is eating RAM — investigate before continuing.

- [ ] **Step 5: Commit**

```bash
git add infra/k3s-install.sh
git commit -m "feat(infra): add k3s install script (traefik+servicelb disabled)"
```

---

## Task 8: Smoke-test the K3s cluster with a hello-world pod

**Files:**
- Create: `infra/hello-world.yaml`

- [ ] **Step 1: Write hello-world manifest**

Create `infra/hello-world.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hello
  labels: { app: hello }
spec:
  restartPolicy: Never
  containers:
    - name: hello
      image: busybox:1.36
      command: ["sh", "-c", "echo 'k3s is alive at' $(date) && sleep 2"]
```

- [ ] **Step 2: Apply and watch**

```bash
kubectl apply -f infra/hello-world.yaml
kubectl wait --for=condition=Ready pod/hello --timeout=60s || true
kubectl logs hello
```

Expected logs: `k3s is alive at <timestamp>`.

- [ ] **Step 3: Clean up**

```bash
kubectl delete -f infra/hello-world.yaml
```

- [ ] **Step 4: Commit**

```bash
git add infra/hello-world.yaml
git commit -m "test(infra): add k3s hello-world smoke test"
```

---

## Task 9: End-of-Day-2 verification

- [ ] **Step 1: Re-run verify script**

```bash
./infra/verify.sh
```

Expected: K3s checks now ✅. Day-3 docker compose checks still ❌.

---

# DAY 3 (WED) — Docker Compose Stack

## Task 10: Write `docker-compose.yml` for local services

**Files:**
- Create: `infra/docker-compose.yml`
- Modify: `.gitignore` (add data dirs)

- [ ] **Step 1: Create the compose file**

Create `infra/docker-compose.yml`:

```yaml
services:
  postgres:
    image: postgres:16-alpine
    container_name: mlops-postgres
    environment:
      POSTGRES_USER: mlops
      POSTGRES_PASSWORD: REDACTED-OLD-DEFAULT
      POSTGRES_DB: mlflow
    ports:
      - "5432:5432"
    volumes:
      - ./data/postgres:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mlops -d mlflow"]
      interval: 5s
      timeout: 5s
      retries: 5

  minio:
    image: minio/minio:RELEASE.2024-08-17T01-24-54Z
    container_name: mlops-minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: REDACTED-OLD-DEFAULT
      MINIO_ROOT_PASSWORD: REDACTED-OLD-DEFAULT
    ports:
      - "9000:9000"   # S3 API
      - "9001:9001"   # console UI
    volumes:
      - ./data/minio:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 5s
      timeout: 5s
      retries: 5

  minio-init:
    # one-shot: creates the 'mlflow' bucket on first start
    image: minio/mc:RELEASE.2024-08-17T11-33-50Z
    depends_on:
      minio: { condition: service_healthy }
    entrypoint: >
      sh -c "
      mc alias set local http://minio:9000 REDACTED-OLD-DEFAULT REDACTED-OLD-DEFAULT &&
      mc mb -p local/mlflow &&
      mc mb -p local/datasets &&
      echo 'buckets ready'
      "

  redis:
    image: redis:7-alpine
    container_name: mlops-redis
    ports:
      - "6379:6379"
    volumes:
      - ./data/redis:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.16.2
    container_name: mlops-mlflow
    depends_on:
      postgres: { condition: service_healthy }
      minio-init: { condition: service_completed_successfully }
    environment:
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: REDACTED-OLD-DEFAULT
      AWS_SECRET_ACCESS_KEY: REDACTED-OLD-DEFAULT
    ports:
      - "5000:5000"
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri postgresql://REDACTED:REDACTED@postgres:5432/mlflow
      --artifacts-destination s3://mlflow/
      --serve-artifacts
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 10s
      timeout: 5s
      retries: 10
```

- [ ] **Step 2: Extend `.gitignore` to ignore data directories**

Append to `.gitignore`:

```
# Local infra data (mounted volumes — not committed)
infra/data/
```

- [ ] **Step 3: Bring the stack up**

```bash
cd infra
docker compose up -d
docker compose ps
```

Expected: 4 services in state `running` or `healthy` (`minio-init` exits cleanly with state `exited (0)` — that's expected, it's a one-shot bucket creator).

- [ ] **Step 4: Wait for MLflow to be healthy**

```bash
docker compose ps mlflow
# Repeat until State shows "(healthy)" — usually 30-60s on first start.
```

- [ ] **Step 5: Commit**

```bash
cd /home/himanshu/learning/mlops-journey
git add infra/docker-compose.yml .gitignore
git commit -m "feat(infra): add docker-compose stack (postgres+minio+redis+mlflow)"
```

---

## Task 11: Smoke-test the stack from a host browser/CLI

**Files:** none new

- [ ] **Step 1: Smoke-test MLflow API**

```bash
curl -sf http://localhost:5000/health
```

Expected: `OK` or `200 OK` (no body — that's normal).

- [ ] **Step 2: Smoke-test MinIO**

```bash
curl -sf http://localhost:9000/minio/health/live
```

Expected: HTTP 200 (no body).

- [ ] **Step 3: Verify the `mlflow` bucket exists in MinIO**

Open `http://localhost:9001` in a browser. Login: `REDACTED-OLD-DEFAULT` / `REDACTED-OLD-DEFAULT`. You should see two buckets: `mlflow` and `datasets`.

- [ ] **Step 4: Verify MLflow UI loads**

Open `http://localhost:5000` in a browser. You should see the MLflow UI, with the default `Experiments` view and no experiments yet.

- [ ] **Step 5: Verify Postgres + Redis are reachable**

```bash
# Postgres
docker exec mlops-postgres psql -U mlops -d mlflow -c "SELECT 1"
# Redis
docker exec mlops-redis redis-cli ping
```

Expected: Postgres returns `1`, Redis returns `PONG`.

---

## Task 12: Add a Makefile target to start/stop the stack

**Files:**
- Create: `Makefile`

- [ ] **Step 1: Write a minimal Makefile**

Create `Makefile`:

```makefile
.PHONY: help verify up down logs clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

verify:  ## Run infrastructure smoke tests
	@bash infra/verify.sh

up:  ## Start docker-compose stack (postgres, minio, redis, mlflow)
	cd infra && docker compose up -d
	@echo ""
	@echo "MLflow UI: http://localhost:5000"
	@echo "MinIO UI:  http://localhost:9001  (REDACTED-OLD-DEFAULT / REDACTED-OLD-DEFAULT)"

down:  ## Stop docker-compose stack
	cd infra && docker compose down

logs:  ## Tail logs from docker-compose stack
	cd infra && docker compose logs -f --tail=50

clean:  ## Stop stack and remove data volumes (DESTRUCTIVE)
	cd infra && docker compose down -v
	rm -rf infra/data/
```

- [ ] **Step 2: Test all targets**

```bash
make help
make verify
make down
make up
make verify  # MLflow check may still fail for ~30s while MLflow boots
```

Expected: `make help` shows colored help. `make verify` reflects current state. `make up` reports the URLs.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "feat: add top-level Makefile (verify/up/down/logs/clean)"
```

---

## Task 13: End-of-Day-3 verification

- [ ] **Step 1: Run verify; should be all green except for any K3s pods we haven't deployed**

```bash
make verify
```

Expected: all 14+ checks pass.

---

# DAY 4 (THU) — Monorepo Skeleton + GitHub Profile

## Task 14: Set up Python workspace root with uv

**Files:**
- Create: `pyproject.toml`
- Create: `.envrc`

- [ ] **Step 1: Initialize uv workspace**

Create `pyproject.toml` (at repo root):

```toml
[project]
name = "mlops-journey"
version = "0.0.1"
description = "Himanshu's 6-month MLOps + Agentic AI portfolio journey."
requires-python = ">=3.11"
readme = "README.md"

[tool.uv.workspace]
members = ["project-0-warmup", "project-1-credit-risk-pipeline", "project-2-sre-copilot",
           "project-3-medical-rag", "project-4-ml-platform-k8s", "project-5-pricing-agent"]
# Note: projects not yet created. uv ignores missing members until they exist.
exclude = []

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B"]

[tool.pytest.ini_options]
testpaths = ["project-0-warmup/tests", "project-1-credit-risk-pipeline/tests"]
addopts = "-ra --strict-markers"
```

- [ ] **Step 2: Create .envrc for direnv**

Create `.envrc`:

```bash
# direnv: loads on `cd` into this repo.
# Auto-activates uv venv and exposes service endpoints.
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=REDACTED-OLD-DEFAULT
export AWS_SECRET_ACCESS_KEY=REDACTED-OLD-DEFAULT
export AWS_DEFAULT_REGION=us-east-1
export OLLAMA_HOST=http://localhost:11434
export REDIS_URL=redis://localhost:6379/0

# Use the repo's uv-managed virtualenv if it exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi
```

- [ ] **Step 3: Allow direnv to load it**

```bash
direnv allow
```

- [ ] **Step 4: Verify env vars are set**

```bash
echo "$MLFLOW_TRACKING_URI"
```

Expected: `http://localhost:5000`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .envrc
git commit -m "feat: add uv workspace root + direnv config"
```

---

## Task 15: Create `project-0-warmup` package skeleton

**Files:**
- Create: `project-0-warmup/pyproject.toml`
- Create: `project-0-warmup/src/credit_baseline/__init__.py`
- Create: `project-0-warmup/src/credit_baseline/pipeline.py`
- Create: `project-0-warmup/tests/__init__.py`
- Create: `project-0-warmup/tests/test_pipeline.py`
- Create: `project-0-warmup/README.md`

- [ ] **Step 1: Create directory tree**

```bash
mkdir -p project-0-warmup/src/credit_baseline
mkdir -p project-0-warmup/tests
mkdir -p project-0-warmup/notebooks
```

- [ ] **Step 2: Write `project-0-warmup/pyproject.toml`**

```toml
[project]
name = "credit-baseline"
version = "0.0.1"
description = "Week-1 sklearn refresher; seed for Project 1 (credit risk)."
requires-python = ">=3.11"
dependencies = [
  "scikit-learn>=1.5",
  "pandas>=2.2",
  "numpy>=2.0",
  "mlflow>=2.16",
  "boto3>=1.34",       # MLflow S3 artifact upload
  "joblib>=1.4",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.6", "jupyter>=1.0", "ipykernel>=6.29"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/credit_baseline"]
```

- [ ] **Step 3: Write placeholder module `src/credit_baseline/__init__.py`**

```python
"""Credit baseline package — Week 1 warmup that seeds Project 1."""
```

- [ ] **Step 4: Write placeholder `src/credit_baseline/pipeline.py` (implementation lands Friday)**

```python
"""Minimal credit-risk baseline. Implementation arrives Friday (Day 5)."""
from __future__ import annotations
```

- [ ] **Step 5: Write empty `tests/__init__.py`**

```python
```

- [ ] **Step 6: Write placeholder test (real test lands Day 5)**

`tests/test_pipeline.py`:

```python
"""Tests for the warmup baseline. Real assertions added Day 5."""

def test_placeholder():
    assert True
```

- [ ] **Step 7: Write `project-0-warmup/README.md`**

```markdown
# project-0-warmup

Week-1 warmup. Loads the German Credit dataset, trains a baseline RandomForest,
logs the run to MLflow, and exposes a `train_baseline()` function used by Project 1.

## Run

```bash
make demo    # at repo root — boots stack + runs the notebook
```
```

- [ ] **Step 8: Sync the venv with uv from repo root**

```bash
cd /home/himanshu/learning/mlops-journey
uv venv .venv
uv pip install -e ./project-0-warmup[dev]
```

Expected: a `.venv/` directory with all deps installed in <30 seconds (uv is fast).

- [ ] **Step 9: Run the placeholder test to confirm wiring**

```bash
source .venv/bin/activate
pytest project-0-warmup -v
```

Expected: `1 passed`.

- [ ] **Step 10: Commit**

```bash
git add project-0-warmup pyproject.toml
git commit -m "feat: scaffold project-0-warmup package + placeholder tests"
```

---

## Task 16: Add `.pre-commit-config.yaml` for code hygiene

**Files:**
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Write config**

`.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ["--maxkb=2000"]
      - id: check-merge-conflict
      - id: detect-private-key

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

- [ ] **Step 2: Install pre-commit and activate hooks**

```bash
uv pip install pre-commit
pre-commit install
```

- [ ] **Step 3: Run against all files (first run will fix many things)**

```bash
pre-commit run --all-files
```

Expected: some files reformatted (trailing whitespace, etc.). Re-run if needed; second run should be clean.

- [ ] **Step 4: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "chore: add pre-commit hooks (ruff, ruff-format, basic hygiene)"
```

---

## Task 17: Write `README.md` (portfolio landing page) and `STATUS.md` (tracker)

**Files:**
- Create: `README.md`
- Create: `STATUS.md`

- [ ] **Step 1: Write top-level README.md**

```markdown
# mlops-journey

A 6-month, project-driven push from DevOps engineering into Senior MLOps Platform
Engineering with Agentic AI specialization. Five portfolio projects spanning fintech,
DevOps tooling, healthcare, platform engineering, and e-commerce.

## Projects

| # | Project | Domain | Status |
|---|---------|--------|--------|
| 0 | Warmup — sklearn baseline | — | 🟡 Week 1 |
| 1 | Self-Healing Credit-Risk Pipeline | Fintech | 📋 Planned |
| 2 | Agentic SRE Co-Pilot | DevOps Tooling | 📋 Planned |
| 3 | Medical-Lit RAG with Continuous Eval | Healthcare | 📋 Planned |
| 4 | Mini ML Platform on Kubernetes ★ | Platform Engineering | 📋 Planned |
| 5 | Autonomous Pricing Agent (Capstone) | E-commerce | 📋 Planned |

★ = portfolio crown jewel.

## Stack (local-first, free)

K3s · Docker Compose · MLflow · MinIO · PostgreSQL · Redis · Ollama (CUDA) ·
LangGraph · Phoenix · Qdrant · Prometheus + Grafana · Terraform · ArgoCD · KServe.

## Quick start

```bash
make help     # show commands
make up       # start MLflow / MinIO / Postgres / Redis
make verify   # smoke-test all services
```

See [`docs/superpowers/specs/`](docs/superpowers/specs/) for the full design and
[`STATUS.md`](STATUS.md) for current week / project progress.
```

- [ ] **Step 2: Write STATUS.md**

```markdown
# STATUS

Single source of truth for "where am I right now?" Updated at the end of every session.

## Current

- **Week:** 1 of 26
- **Project:** project-0-warmup (Week 1 foundations)
- **Day:** TBD — update at end-of-day
- **Blockers:** none
- **Last commit:** see `git log -1 --oneline`

## Week 1 milestones (Mon–Fri)

- [ ] Mon — Toolchain installed (uv, Ollama, GPU verified)
- [ ] Tue — K3s + Helm + kubectl, hello-world ran
- [ ] Wed — Docker Compose stack up; MLflow + MinIO + Postgres + Redis healthy
- [ ] Thu — Repo skeleton + Makefile + pre-commit + README
- [ ] Fri — Sklearn baseline notebook + tested module + MLflow run logged

## Up next (Week 2)

Project 1 — Self-Healing Credit-Risk Pipeline. Plan to be written at end of Week 1.

## Memory

Conversational context lives at:
`~/.claude/projects/-home-himanshu-learning-mlops-journey/memory/`

`MEMORY.md` is the index. Future Claude sessions read it first.
```

- [ ] **Step 3: Commit**

```bash
git add README.md STATUS.md
git commit -m "docs: add portfolio README and STATUS tracker"
```

---

## Task 18: GitHub remote setup (manual / web action)

**Files:** none

- [ ] **Step 1: Create the GitHub repo (web)**

Go to https://github.com/new. Settings:
- Repository name: `mlops-journey`
- Description: `7-yr DevOps engineer → Senior MLOps + Agentic AI. 5 portfolio projects in 6 months.`
- Visibility: **Public** (this is for showcasing)
- Do **NOT** check "Add a README" — we have one locally.

- [ ] **Step 2: Add remote and push**

```bash
cd /home/himanshu/learning/mlops-journey
git remote add origin git@github.com:<your-github-username>/mlops-journey.git
git branch -M main
git push -u origin main
```

If you don't have an SSH key set up for GitHub, use HTTPS:
```bash
git remote add origin https://github.com/<your-github-username>/mlops-journey.git
```

- [ ] **Step 3 (optional but recommended): Pin this repo on your GitHub profile**

GitHub.com → Your profile → Customize your pins → select `mlops-journey`.

- [ ] **Step 4: Update profile bio (web)**

Set your GitHub bio to: *"7-yr DevOps engineer building production MLOps & agentic AI systems."*

---

# DAY 5 (FRI) — sklearn Baseline + MLflow End-to-End

## Task 19: Implement the `train_baseline()` function (TDD)

**Files:**
- Modify: `project-0-warmup/src/credit_baseline/pipeline.py`
- Modify: `project-0-warmup/tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test first**

Replace contents of `project-0-warmup/tests/test_pipeline.py` with:

```python
"""Smoke tests for the warmup baseline."""
import numpy as np
import pandas as pd
import pytest

from credit_baseline.pipeline import train_baseline, BaselineResult


@pytest.fixture
def tiny_synthetic_df():
    """A 100-row synthetic credit-like dataset. Just enough to train end-to-end."""
    rng = np.random.default_rng(seed=42)
    n = 100
    df = pd.DataFrame({
        "age": rng.integers(18, 75, size=n),
        "income": rng.normal(50_000, 20_000, size=n).clip(min=5_000),
        "loan_amount": rng.normal(15_000, 8_000, size=n).clip(min=500),
        "credit_history_years": rng.integers(0, 30, size=n),
        # binary target: rough heuristic so the model has signal
        "default": ((rng.normal(0, 1, size=n) - 0.5) > 0).astype(int),
    })
    return df


def test_train_baseline_returns_result(tiny_synthetic_df):
    result = train_baseline(tiny_synthetic_df, target_col="default", random_state=0)
    assert isinstance(result, BaselineResult)


def test_train_baseline_metrics_are_in_unit_interval(tiny_synthetic_df):
    result = train_baseline(tiny_synthetic_df, target_col="default", random_state=0)
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.roc_auc <= 1.0


def test_train_baseline_model_predicts_correct_shape(tiny_synthetic_df):
    result = train_baseline(tiny_synthetic_df, target_col="default", random_state=0)
    X = tiny_synthetic_df.drop(columns=["default"])
    preds = result.model.predict(X)
    assert preds.shape == (len(tiny_synthetic_df),)
    assert set(np.unique(preds)).issubset({0, 1})
```

- [ ] **Step 2: Run the test — verify it fails for the right reason**

```bash
cd /home/himanshu/learning/mlops-journey
source .venv/bin/activate
pytest project-0-warmup -v
```

Expected: failure with `ImportError: cannot import name 'train_baseline'` or `BaselineResult`. That's the *right* failure mode — pipeline isn't implemented yet.

- [ ] **Step 3: Implement the minimal code to make it pass**

Replace contents of `project-0-warmup/src/credit_baseline/pipeline.py` with:

```python
"""Minimal credit-risk baseline.

This module is the *seed* for Project 1 (self-healing pipeline). It does the
essential vertical slice:
  - take a labeled DataFrame
  - train/test split
  - fit a RandomForest
  - return a structured result with model + metrics

Project 1 will replace this with: DVC-tracked data, XGBoost, MLflow runs,
canary deployment, drift detection. For now: keep it boring and well-tested.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

DEFAULT_TEST_SIZE: Final[float] = 0.25


@dataclass(frozen=True)
class BaselineResult:
    """Outcome of one training run."""
    model: RandomForestClassifier
    accuracy: float
    roc_auc: float
    n_train: int
    n_test: int


def train_baseline(
    df: pd.DataFrame,
    target_col: str,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = 42,
) -> BaselineResult:
    """Train a baseline RandomForest classifier on the given DataFrame.

    Args:
        df: a labeled DataFrame including the target column.
        target_col: the column name to predict (must be binary 0/1).
        test_size: fraction held out for evaluation.
        random_state: seeds train_test_split and the model for reproducibility.
    """
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return BaselineResult(
        model=model,
        accuracy=accuracy_score(y_test, y_pred),
        roc_auc=roc_auc_score(y_test, y_proba),
        n_train=len(X_train),
        n_test=len(X_test),
    )
```

- [ ] **Step 4: Run the tests; expect all 3 to pass**

```bash
pytest project-0-warmup -v
```

Expected output:
```
test_train_baseline_returns_result PASSED
test_train_baseline_metrics_are_in_unit_interval PASSED
test_train_baseline_model_predicts_correct_shape PASSED
```

- [ ] **Step 5: Commit**

```bash
git add project-0-warmup/src project-0-warmup/tests
git commit -m "feat(warmup): implement train_baseline + smoke tests (TDD green)"
```

---

## Task 20: Write the end-to-end notebook that loads German Credit + logs to MLflow

**Files:**
- Create: `project-0-warmup/notebooks/credit_baseline.ipynb`

We'll create it as a Python file and convert, since we're writing it from a script.

- [ ] **Step 1: Install Jupyter kernel for the venv (if not already)**

```bash
uv pip install jupyter ipykernel
python -m ipykernel install --user --name mlops-journey --display-name "Python (mlops-journey)"
```

- [ ] **Step 2: Create the notebook as a Python file first**

Create `project-0-warmup/notebooks/_credit_baseline.py`:

```python
# %% [markdown]
# # Credit Baseline — Week 1 End-to-End
#
# Loads the German Credit dataset from UCI (the LendingClub dataset is too big
# for a warmup), trains the baseline RandomForest, logs the run to MLflow,
# and saves the model as an MLflow artifact.
#
# This is the **vertical slice** for Project 1: data → model → tracked run.

# %%
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from credit_baseline.pipeline import train_baseline

# %% [markdown]
# ## Load German Credit dataset
# UCI provides it at a stable URL. ~1000 rows, 20 features, binary target.

# %%
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data"
)
COLS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job", "num_dependents",
    "own_telephone", "foreign_worker", "default",
]

df = pd.read_csv(UCI_URL, sep=" ", header=None, names=COLS)
# Target in this dataset is 1=good, 2=bad. Recode so 1 = default.
df["default"] = (df["default"] == 2).astype(int)

# One-hot encode the categorical columns (RandomForest needs numeric input)
df = pd.get_dummies(df, drop_first=True)
print(f"Shape: {df.shape}, default rate: {df['default'].mean():.2%}")
df.head()

# %% [markdown]
# ## Train + log to MLflow

# %%
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("credit-baseline")

with mlflow.start_run(run_name="warmup-randomforest"):
    result = train_baseline(df, target_col="default", random_state=42)

    mlflow.log_params({
        "model": "RandomForestClassifier",
        "n_estimators": 200, "max_depth": 8, "random_state": 42,
    })
    mlflow.log_metrics({
        "accuracy": result.accuracy,
        "roc_auc": result.roc_auc,
        "n_train": result.n_train,
        "n_test": result.n_test,
    })
    mlflow.sklearn.log_model(result.model, artifact_path="model")

    print(f"accuracy: {result.accuracy:.3f}  roc_auc: {result.roc_auc:.3f}")

# %% [markdown]
# ## Verify the run lands in MLflow
# Open http://localhost:5000 — you should see one experiment "credit-baseline"
# with one run, including the model artifact in MinIO under s3://mlflow/.
```

- [ ] **Step 3: Convert the script to an actual notebook**

```bash
uv pip install jupytext
cd project-0-warmup/notebooks
jupytext --to ipynb _credit_baseline.py -o credit_baseline.ipynb
rm _credit_baseline.py
cd /home/himanshu/learning/mlops-journey
```

- [ ] **Step 4: Bring the stack up if it isn't already**

```bash
make up
make verify
```

Expected: all services healthy.

- [ ] **Step 5: Execute the notebook headlessly to prove it works end-to-end**

```bash
uv pip install nbconvert
jupyter nbconvert --to notebook --execute project-0-warmup/notebooks/credit_baseline.ipynb \
  --output credit_baseline.ipynb
```

Expected: no errors. The notebook now has populated output cells (a printed DataFrame head, accuracy/roc_auc numbers).

- [ ] **Step 6: Verify the run shows up in MLflow UI**

Open http://localhost:5000. You should see:
- Experiment named `credit-baseline`
- One run named `warmup-randomforest`
- Metrics: `accuracy`, `roc_auc`, `n_train`, `n_test`
- An `model/` artifact stored under MinIO

Click the run → Artifacts → `model/` → you should see `MLmodel`, `model.pkl`, `conda.yaml`, etc.

- [ ] **Step 7: Commit**

```bash
git add project-0-warmup/notebooks/
git commit -m "feat(warmup): add end-to-end German Credit notebook (data→model→MLflow)"
```

---

## Task 21: Add a `make demo` target that runs the whole vertical slice

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Append demo target to Makefile**

Add to `Makefile`:

```makefile
demo:  ## Warmup end-to-end: ensure stack is up, run the baseline notebook
	@make up
	@echo "── Waiting for MLflow ──"
	@for i in 1 2 3 4 5 6 7 8 9 10; do \
		curl -sf http://localhost:5000/health && break || sleep 5; \
	done
	@echo "── Running baseline notebook ──"
	jupyter nbconvert --to notebook --execute \
		project-0-warmup/notebooks/credit_baseline.ipynb \
		--output credit_baseline.ipynb
	@echo ""
	@echo "✅ Done. Open http://localhost:5000 to see the run."
```

- [ ] **Step 2: Test it**

```bash
make demo
```

Expected: stack comes up, notebook runs, success message. ~30 seconds if everything is already up.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "feat: add 'make demo' for end-to-end warmup run"
```

---

## Task 22: Update STATUS.md and push

**Files:**
- Modify: `STATUS.md`

- [ ] **Step 1: Tick the Week 1 checkboxes**

Replace the milestones section in `STATUS.md` with:

```markdown
## Week 1 milestones (Mon–Fri)

- [x] Mon — Toolchain installed (uv, Ollama, GPU verified)
- [x] Tue — K3s + Helm + kubectl, hello-world ran
- [x] Wed — Docker Compose stack up; MLflow + MinIO + Postgres + Redis healthy
- [x] Thu — Repo skeleton + Makefile + pre-commit + README
- [x] Fri — Sklearn baseline notebook + tested module + MLflow run logged
```

Also update `Current` section:

```markdown
## Current

- **Week:** 1 of 26 — ✅ complete
- **Project:** project-0-warmup — ✅ complete
- **Next:** Week 2 — Project 1 (Credit-Risk Pipeline). Plan to be written.
- **Blockers:** none
- **Last commit:** see `git log -1 --oneline`
```

- [ ] **Step 2: Commit and push everything**

```bash
git add STATUS.md
git commit -m "docs(status): Week 1 foundation setup complete"
git push origin main
```

- [ ] **Step 3: Verify GitHub shows the activity**

Open `https://github.com/<your-username>/mlops-journey`. You should see:
- ~20-25 commits from this week
- README rendering correctly
- The folder structure visible
- Green contribution squares for the past 5 days

---

## Task 23: Final smoke-test pass

- [ ] **Step 1: Stop all containers**

```bash
make down
```

- [ ] **Step 2: Start fresh**

```bash
make up
```

- [ ] **Step 3: Wait ~30s for services to settle, then run full verify**

```bash
sleep 30 && make verify
```

Expected: every check ✅. If anything fails, **fix before Week 2** — Project 1 assumes this baseline works.

- [ ] **Step 4: End-of-week reflection in a journal entry**

Append to `STATUS.md`:

```markdown
## Week 1 retrospective

- **Time logged:** ~10 hours
- **What worked:**
- **What surprised me:**
- **What I want to remember:**
```

Fill in the bullets honestly (1-2 lines each). Commit:

```bash
git add STATUS.md
git commit -m "docs(status): Week 1 retrospective"
git push
```

---

# Done — Week 1 Complete

**You've shipped:**
- A working local MLOps dev environment (K3s + Docker Compose stack)
- GPU-accelerated local LLM via Ollama
- A monorepo skeleton with Python workspace, pre-commit hooks, Makefile
- A tested baseline module + a working end-to-end notebook (data → model → MLflow → MinIO)
- The portfolio repo live on GitHub with daily commits

**Week 2 starts with Plan 2 (Project 1 — Credit-Risk Pipeline).** Tell Claude: *"Let's plan Project 1."*
