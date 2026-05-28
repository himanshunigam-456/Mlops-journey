# mlops-journey

A 6-month, project-driven push from DevOps engineering into Senior **MLOps
Platform Engineering with Agentic AI** specialization. Five portfolio projects
spanning fintech, DevOps tooling, healthcare, platform engineering, and
e-commerce — all built local-first on free/OSS tooling.

## Projects

| # | Project | Domain | Status |
|---|---------|--------|--------|
| 0 | Warmup — sklearn baseline + MLflow vertical slice | — | 🟡 Week 1 |
| 1 | Self-Healing Credit-Risk Pipeline | Fintech | 📋 Planned |
| 2 | Agentic SRE Co-Pilot (Autonomous Incident Investigator) | DevOps Tooling | 📋 Planned |
| 3 | Medical-Literature RAG with Continuous Evaluation | Healthcare | 📋 Planned |
| 4 | Mini ML Platform on Kubernetes ★ | Platform Engineering | 📋 Planned |
| 5 | Autonomous Pricing & Inventory Agent (Capstone) | E-commerce | 📋 Planned |

★ = portfolio crown jewel.

## Stack (local-first, free)

K3d · Docker Compose · MLflow · MinIO · PostgreSQL · Redis · Ollama (CUDA) ·
LangGraph · Phoenix · Qdrant · Prometheus + Grafana · Terraform · ArgoCD · KServe.

A full "Paid Tool → Free Alternative" mapping is in the design doc.

## Quick start

```bash
# 1. Smoke-test all infrastructure
make verify          # expect 18/18 ✅

# 2. Start the local MLOps stack (MLflow, MinIO, Postgres, Redis)
make up

# 3. Browse services
#    MLflow UI: http://localhost:5000
#    MinIO UI:  http://localhost:9001  (REDACTED-OLD-DEFAULT / REDACTED-OLD-DEFAULT)
```

## Repository layout

```
mlops-journey/
├── infra/                              ← local infrastructure
│   ├── docker-compose.yml              ← MLflow + MinIO + Postgres + Redis
│   ├── k3d-create.sh                   ← K8s cluster bootstrap
│   ├── mlflow-entrypoint.sh            ← runtime deps installer
│   ├── hello-world.yaml                ← cluster smoke test
│   └── verify.sh                       ← end-to-end infra check
├── docs/superpowers/specs/             ← design docs + plans
├── project-0-warmup/                   ← Week 1 sklearn refresher
├── project-1-credit-risk-pipeline/     ← Fintech (planned)
├── project-2-sre-copilot/              ← DevOps Tooling (planned)
├── project-3-medical-rag/              ← Healthcare (planned)
├── project-4-ml-platform-k8s/          ← Platform Engineering ★ (planned)
├── project-5-pricing-agent/            ← E-commerce capstone (planned)
└── STATUS.md                           ← current week / blockers
```

## Author

**Himanshu Nigam** — 7-yr DevOps engineer building production MLOps & agentic AI systems.

---

*Design + plans live in [`docs/superpowers/specs/`](docs/superpowers/specs/). Currently executing Week 1 of 26.*
