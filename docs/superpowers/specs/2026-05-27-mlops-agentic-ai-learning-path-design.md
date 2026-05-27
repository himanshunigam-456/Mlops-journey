# MLOps + Agentic AI Learning Path — Design Document

**Author:** Himanshu Nigam (with Claude as mentor/copilot)
**Created:** 2026-05-27
**Duration:** 6 months (with 2-week buffer)
**Status:** Draft pending user approval

---

## 1. Goal

Transform a 7-year DevOps engineer into a **Senior MLOps Platform Engineer with Agentic AI specialization** ("unicorn profile"), positioned to land:

- A senior MLOps role in India at ₹70L-1.2Cr, OR
- A remote US/EU MLOps contract at $130-200K, OR
- Freelance contracts at $80-150/hr via Toptal/Arc/Braintrust/direct.

**Income target:** ₹40 LPA → ₹1 Cr+, projected to cross between Year 1.5 and Year 2 post-completion.

---

## 2. Starting Context

| Factor | Reality |
|--------|---------|
| **Background** | 7 yrs DevOps. Strong in Linux, Docker, K8s, CI/CD, AWS, observability, IaC, Python scripting. |
| **ML/AI level** | Basic — has used pandas/sklearn but never shipped production ML. |
| **Schedule** | 2 hrs/day weekday minimum (10 hrs/week baseline). Weekends optional. |
| **Hardware** | Ubuntu 22.04, i5-9300H (8 cores), 23 GB RAM, **NVIDIA GTX 1650 4GB VRAM** (CUDA), 112 GB free disk. |
| **Budget** | Strict low-budget. Free / OSS / local-first tools only. Estimated total spend over 6 months: ₹2000-5000. |
| **Preferred cloud** | AWS (most familiar from DevOps work). |
| **Learning style** | Doing-first. No long lectures. Senior-engineer pair-programming with Claude. |

---

## 3. Chosen Approach: Project-Driven, Parallel Learning

**Why this approach (over Foundations-First or Sprint):**

1. Six-month timeline matches a 10 hrs/week sustainable pace without burnout.
2. First GitHub-pushable artifact by **end of Week 4** — keeps motivation high.
3. Theory is pulled in *as needed for the next thing being built*, not front-loaded.
4. Every project is independently sellable as a freelance pitch — income can start before completion.
5. The DevOps moat (K8s, infra, observability) is leveraged in every project. Projects do not ask the user to compete with ML PhDs on accuracy.

**What this approach explicitly avoids:**

- No multi-week ML theory courses (Andrew Ng, fast.ai's full Part 1, etc.). Only targeted 20-25 min videos when a specific concept is needed.
- No Kaggle accuracy obsession.
- No DSA refreshers, math refreshers, or unrelated CS coursework.

---

## 4. Six-Month Roadmap

```
Month 1 ─ Foundations + Project 1 starts (Self-Healing Credit-Risk Pipeline)
  Week 1   : Setup + Python prod patterns + sklearn end-to-end refresher
  Week 2-4 : Build Project 1 → first GitHub push by end of Week 4

Month 2 ─ Project 1 finish + Project 2 starts (Agentic SRE Co-Pilot)
  Week 5-6 : Polish Project 1, write BLOG.md, publish to dev.to + LinkedIn
  Week 7-8 : Begin Project 2 — LangGraph fundamentals + agent design

Month 3 ─ Project 2 deep dive (multi-agent investigation system)

Month 4 ─ Project 3 (Medical Lit RAG with Continuous Eval)
  ↑ Start applying for ₹70L+ Senior MLOps roles in India.

Month 5 ─ Project 4 (Mini ML Platform on Kubernetes) — CROWN JEWEL

Month 6 ─ Project 5 (Autonomous Pricing Agent on the platform) + freelance launch
  ↑ Apply on Toptal/Arc/Braintrust. Aggressive job interviews.
```

**Total buffer:** ~2 weeks distributed across months for slip / illness / family.

---

## 5. Weekly Cadence (2 hrs/day weekday × 5 days = 10 hrs/week)

```
Mon-Fri  2 hrs/day :  20 min  – read official docs / 1 focused video / Claude Q&A
                       1h 40m – build, debug, push code
                       End of session: 1 commit minimum (even a README polish).

Sat-Sun           :  OPTIONAL bonus deep-work block (4-8 hrs total when available).
                      Used for harder tasks (e.g., K8s operator wiring).
                      Never required for plan to succeed.

End of Friday     :  Must hit that week's deliverable. Weekend is bonus only.
```

**"I'm tired" backup plan:** If a weekday slot is dead, push 1 commit — even a doc improvement. Don't break the streak. The GitHub contribution graph is part of the portfolio.

**Per-concept learning workflow (when hitting a new tool/idea):**

1. Read official docs for 20 min — get the high-level shape.
2. Watch one focused YouTube tutorial — max 25 min.
3. Ask Claude to explain it in DevOps analogues (e.g., *"KServe = ML's Ingress controller"*, *"MLflow = artifactory for models"*, *"Feature Store = Redis for ML features"*).
4. Build a toy version in ≤ 1 hour.
5. Apply it to the current project — that's where real retention happens.

**Maximum total time from "first heard of X" to "shipped X in code": ~1.5 hours.**

---

## 6. The 5 Projects (Domain-Diversified Portfolio)

Each project is from a **different industry domain** to demonstrate cross-vertical capability.

### Project 1 — Self-Healing Credit-Risk ML Pipeline
**Domain:** Fintech · **Weeks:** 2-6

**Real industry problem.** Banks and fintechs deploy ML models for credit scoring and then suffer silent model drift when customer behavior shifts. Most teams discover the problem months later via lawsuits, regulatory action, or P&L loss.

**What's built.** A complete production-style pipeline:

- Ingest a public credit-risk dataset (LendingClub or German Credit) via DVC versioning
- Train an XGBoost / LightGBM model; track all runs in MLflow (experiments + model registry)
- Evaluate against a holdout set; reject promotion if metrics regress
- Serve via FastAPI containerized + deployed to AWS EC2 t2.micro (free tier)
- Monitor production traffic with Evidently AI for data drift + prediction drift
- On drift detection: trigger GitHub Actions retraining workflow
- New model deploys as canary (10% traffic), promoted to 100% if metrics hold
- Prometheus scrapes inference metrics; Grafana dashboard shows model health

**Stack.** Python (uv) · scikit-learn · XGBoost · MLflow · DVC (S3/MinIO backend) · FastAPI · Evidently AI · Prometheus + Grafana · GitHub Actions · Docker · AWS EC2 t2.micro (production demo).

**Unfair angle.** Most ML tutorials end at "I trained a model." This shows the full lifecycle including drift detection — the *one thing* every MLOps hiring manager asks about in interviews.

**Freelance pitch.** *"I deploy ML systems that self-heal — your model doesn't go silently stale."* Target clients: fintechs, e-commerce churn teams, fraud detection teams.

---

### Project 2 — Agentic SRE Co-Pilot (Autonomous Incident Investigator)
**Domain:** DevOps / SaaS Tooling · **Weeks:** 7-12

**Real industry problem.** Senior SREs (₹50L+) spend 40% of incident time on grunt-work investigation: log spelunking, metric pulling, "what changed in the last 2 hours" forensics. This is pure burn of expensive engineering time. The user has lived this pain personally.

**What's built.** A multi-agent system that, on receiving an alert:

- Webhook ingests a PagerDuty / Slack alert (or simulated alert)
- LangGraph coordinator spawns parallel investigator agents:
  - **Logs agent** — queries Loki (real or fake) for anomalies in the affected service
  - **Metrics agent** — pulls Prometheus for spikes, drops, saturation
  - **Deploys agent** — checks recent GitHub releases / ArgoCD sync events
  - **Runbook agent** — semantic-searches a markdown runbook knowledge base
- Each investigator returns structured findings (timeline, evidence, hypotheses)
- Coordinator synthesizes → drafts incident timeline + likely root cause + suggested mitigation
- Posts back to Slack with a button for human override (which becomes a training signal)
- Replayable on historical incidents → measure diagnosis accuracy as eval metric

**Stack.** LangGraph · Ollama (local Llama 3.1 8B Q4 with GPU acceleration on his GTX 1650) for dev · Anthropic Claude Sonnet 4.6 / Groq Llama 3 for production quality · FastAPI · PostgreSQL (incident memory) · Loki + Prometheus (containerized for demo) · Slack webhook.

**Unfair angle.** Only a 7-year DevOps engineer could imagine and design this workflow at the right fidelity. ML engineers build chatbots; he's building tooling that ships AI into the workflow he uniquely understands. **Nobody else has this in their portfolio.**

**Freelance pitch.** *"I build AI agents for DevOps/SRE teams that actually save on-call time."* Target clients: SaaS companies, DevOps tooling vendors, MSPs, observability vendors.

---

### Project 3 — Medical-Literature RAG with Continuous Evaluation
**Domain:** Healthcare · **Weeks:** 13-16

**Real industry problem.** Every company is shipping a "GPT for our docs" feature. 80% have no idea if it's getting better or worse over time. In healthcare specifically — where a wrong drug-interaction answer is a lawsuit — the *eval* is the product. The differentiation here is not "I built RAG"; it's "I built RAG with a regression-detecting eval pipeline."

**What's built.** A RAG over PubMed abstracts (free public dataset, no PHI/compliance issues):

- Ingest ~50K PubMed abstracts via DVC; chunk + embed using `sentence-transformers/all-mpnet-base-v2` (local, free)
- Store in Qdrant vector DB
- Retrieval + reranking + LLM answer synthesis
- **Offline eval pipeline:** RAGAS metrics (faithfulness, context precision, answer relevance) against a curated test set of 200 medical questions
- **Online eval:** every query logged + LLM-as-judge auto-scoring + thumbs up/down endpoint
- **A/B testing:** 2 prompt variants × 2 embedding models with statistical significance tracking via SciPy
- **Cost controls:** semantic caching reduces costs ~60% — measured and shown
- **Observability:** Phoenix (Arize OSS) traces every query
- **Load testing:** Locust proves p95 latency holds under 50 RPS

**Stack.** LangChain · Qdrant · `sentence-transformers` (local embeddings, no API cost) · Phoenix (OSS) · Ollama for dev + Claude Sonnet for final quality benchmark · FastAPI · Locust · GitHub Actions for eval-on-PR.

**Unfair angle.** Anyone can build a RAG demo. Almost nobody builds it like a real service with **regression-detecting evals + cost dashboards + statistical A/B**. That's the senior-engineering signal.

**Freelance pitch.** *"I take your RAG demo to production with proper evals so you actually know it's working."* Target clients: any company shipping a v0 GenAI feature that is now panicking about quality.

---

### Project 4 — Mini ML Platform on Kubernetes (Vertex AI / SageMaker Clone)
**Domain:** Platform Engineering (cross-vertical) · **Weeks:** 17-22 · **★ Crown Jewel**

**Real industry problem.** Mid-size companies (50-500 engineers) can't afford SageMaker / Vertex AI ($50K+/yr base) but their data scientists want a self-serve ML platform. So they hire 3-engineer "ML platform teams" at ₹2 Cr+/yr to build internal Vertex AI clones. **This project shows the user can do that team's job alone.**

**What's built.** A self-service mini ML platform that runs on a K8s cluster:

- **Notebook-as-a-Service** — JupyterHub on K8s, users get GPU/CPU notebooks on demand
- **Training jobs** — Submit YAML, Kubeflow Training Operator runs it, logs auto-flow to MLflow
- **Model registry & serving** — MLflow + KServe (Knative-based); promote a model → auto-deploy as REST endpoint with autoscaling
- **Feature store** — Feast for offline (BigQuery / DuckDB) + online (Redis) features
- **GitOps** — ArgoCD watches a config repo; all infra changes applied via PR
- **Cost & resource tracking** — Prometheus + Grafana dashboard showing platform cost per team
- **All packaged** — Terraform module + Helm umbrella chart; `helm install my-mlplatform` works on any K8s cluster
- **Local + cloud demo** — Runs on K3s locally; documented 1-day spin-up on AWS EKS for the portfolio demo video

**Stack.** K3s (local) + AWS EKS (1-day cloud demo) · Helm · Terraform · ArgoCD · MLflow · KServe · Kubeflow Training Operator · Feast · JupyterHub · Prometheus / Grafana / Loki · DuckDB.

**Unfair angle.** This is the resume-defining repo. When a hiring manager sees a working `helm install` command in the README that spins up an entire ML platform — interview is fast-tracked. **₹70L+ jobs hinge on this exact capability.**

**Freelance pitch.** *"I build internal ML platforms. Stop paying SageMaker/Vertex AI."* Target clients: mid-size companies starting their AI journey. Engagements are often 6-month at $80-150/hr.

---

### Project 5 — Autonomous Pricing & Inventory Agent for E-commerce
**Domain:** E-commerce / Retail · **Weeks:** 23-26 · **Capstone**

**Real industry problem.** Dynamic pricing and inventory teams at e-commerce companies cost ₹5 Cr+ to staff. An autonomous agent that monitors competitor prices, demand signals, and inventory — and recommends/applies price changes within configured guardrails — is a high-ROI product.

**What's built.** A multi-agent pricing system **deployed on the Project 4 ML Platform** (this is what ties everything together):

- Forecasting model — predicts demand from historical data (public dataset: Instacart Market Basket or Olist Brazilian E-commerce). Trained on the platform, registered in MLflow, served via KServe.
- Multi-agent system (LangGraph):
  - **Market agent** — monitors simulated competitor price feeds
  - **Demand agent** — queries the forecasting model for next-7-day demand prediction
  - **Inventory agent** — checks stock levels and reorder lead times
  - **Pricing agent** — synthesizes signals; proposes price changes within guardrails (e.g., never below cost, max ±15%)
  - **Audit agent** — explains every decision in human-readable form
- Continuous evaluation — backtest agent decisions against historical sales data; measure simulated revenue lift
- Production-style ops — LangSmith/Langfuse traces, Grafana dashboards, A/B test of agent strategies
- All monitored via *his own* Project 4 platform

**Stack.** Everything from Projects 1-4, synthesized.

**Unfair angle.** This is the "hire me" project. Synthesis matters more than any single feature. The story: *"I built the platform. Then I built and deployed an agent on it. Then I evaluated and improved it over 2 weeks of production traffic."* That's a senior staff ML platform engineer's full resume in a single repo.

**Freelance pitch.** *"I build production-grade AI agents end-to-end — platform, deployment, evaluation."* Target clients: premium clients via Toptal/Arc. Rate: ₹150-200/hr or $100-150/hr.

---

## 7. Per-Project Deliverable Checklist

Every project must ship with:

- ✅ Working `make demo` command that spins up the system
- ✅ README with architecture diagram (Mermaid or Excalidraw)
- ✅ `BLOG.md` (800-1500 words) explaining decisions, tradeoffs, and what failed
- ✅ A 90-second Loom or GIF demo embedded in README
- ✅ Tests (at minimum smoke tests — most ML repos lack these; this is a differentiator)
- ✅ Cost breakdown section ("Running this costs ₹X/mo on cloud, ₹0 locally")
- ✅ Blog post published to **dev.to + LinkedIn**, linking back to the repo

---

## 8. Tooling Stack

**Default principle:** local-first, OSS, free-tier. Paid tools only where unavoidable and only for final-polish portfolio demos.

### 8.1 Default Chosen Stack

**Local infrastructure**
- K3s (lightweight Kubernetes) — runs the platform, agents, model serving
- Docker + Docker Compose
- MinIO (local S3-compatible)
- PostgreSQL + Redis (in Docker)
- uv (modern Python package manager — way faster than pip/poetry)
- direnv (per-project env vars)

**MLOps stack (all OSS, runs in K3s)**
- MLflow — experiment tracking + model registry
- DVC — data version control
- Feast — feature store
- KServe — model serving
- Kubeflow Training Operator — distributed training jobs
- JupyterHub on K8s — multi-user notebooks
- Argo Workflows — pipeline orchestration
- Evidently AI — drift detection

**Observability (OSS)**
- Prometheus + Grafana + Loki + Tempo

**LLM / Agentic stack**
- Ollama with **GPU acceleration on GTX 1650** — local Llama 3.1 8B, Mistral, Phi-3 for dev
- Groq API (free tier) — fast Llama 3 inference for dev
- Google AI Studio (free tier) — Gemini Flash
- Anthropic Claude API — only for final production-quality demos (~$5-10 total)
- LangGraph + LangChain — agent orchestration
- Phoenix (Arize OSS) — LLM tracing
- Langfuse OSS — self-hosted LangSmith alternative
- Qdrant — vector DB (local)
- `sentence-transformers` — local embeddings (free, no API)

**CI/CD & deployment**
- GitHub Actions (free for public repos)
- GHCR — GitHub Container Registry (free)
- ArgoCD — GitOps in K3s
- Terraform (CLI free; Terraform Cloud free tier for state if needed)

### 8.2 Paid Tool → Free Alternative Cheatsheet

| Paid / Industry-Standard | Free Alternative | Notes |
|--------------------------|------------------|-------|
| AWS S3 | **MinIO** (local) or S3 free tier (5GB / 12 mo) | API-compatible; code stays the same |
| AWS SageMaker / Vertex AI | **MLflow + KServe + Feast on K3s** | *This IS Project 4* |
| OpenAI / Claude API | **Ollama** + Groq + Google AI Studio free | Paid only for final benchmarks |
| GitHub Copilot | **Continue.dev** + Ollama, or Claude Code free tier | Plug into VSCode |
| Cloud GPU rental | Local **GTX 1650** + Google Colab + **Kaggle 30 hr/wk free GPU** | His GTX handles most dev |
| Pinecone vector DB | **Qdrant** or Chroma | Both production-grade OSS |
| LangSmith | **Phoenix (Arize OSS)** or **Langfuse OSS** | Self-hosted, near-identical features |
| Datadog / New Relic | **Prometheus + Grafana + Loki** | Industry-standard OSS |
| Weights & Biases | **MLflow** | MLflow does experiment tracking too |
| Snowflake | **DuckDB** | Blazing fast for analytics on parquet |
| Astronomer (Airflow Cloud) | Self-hosted Airflow / **Prefect free tier** / **Dagster OSS** | Dagster is the modern pick |
| AWS EKS (~$73/mo + nodes) | **K3s locally** (free) | EKS only for 1-day Project 4 demo |
| AWS RDS | PostgreSQL in Docker | RDS only for portfolio demo |
| AWS ECR | **GHCR** | Free for public images |
| AWS Lambda | Cloud Run free tier / local cron | Or FastAPI on EC2 t2.micro |
| PagerDuty | **Grafana OnCall** or Opsgenie free | For Project 2 integration |
| Hex / Mode | Local Jupyter via JupyterHub on K3s | *This is Project 4* |
| Determined AI / Modal | Kubeflow Training Operator | Self-hosted training orchestration |
| Replicate / HF Inference | KServe on K3s | Self-host model endpoints |

### 8.3 Budget Guardrails

- **Monthly spend cap during learning:** ₹1000 (~$12 USD).
- If a week's plan would exceed ₹300 in spend, flag to user before proceeding.
- Free tiers to register Week 1: AWS, Groq, Google AI Studio, Anthropic ($5 starter), LangSmith free, HuggingFace, Kaggle.

---

## 9. GitHub Profile Strategy

### 9.1 Profile-Level (Week 1)

- Pinned repos in order: **Project 5, Project 4, Project 2, Project 3, Project 1** (impressive-first)
- Profile README with "Shipping in 2026" status section: ✅ done · 🟡 in progress · 📋 planned
- One-line bio: *"7-yr DevOps engineer building production MLOps & agentic AI systems."*

### 9.2 Per-Repo Requirements

Each project repo includes:

- **Architecture diagram** (Mermaid or Excalidraw) in README
- **"Why this exists"** section — business problem in 2 sentences
- **Working `make demo`** — recruiters will try this
- **Loom/GIF demo** (90 seconds max)
- **`BLOG.md`** with decisions, tradeoffs, failures
- **Tests** (at least smoke)
- **Cost breakdown** section

### 9.3 Commit Hygiene

- Conventional commits (`feat:`, `fix:`, `docs:`, `chore:`)
- 1 commit per ~30 min of work, descriptive messages
- Daily commits during build weeks — contribution graph becomes a magnet

### 9.4 Blog-as-Moat Strategy

After each project:

1. Publish `BLOG.md` to **dev.to** (developer-first audience)
2. LinkedIn post — short hook + link to dev.to
3. Title formula: *"I built [X] in 6 weeks — here's what broke"*
4. Every post links back to GitHub repo

Each post earns 200-2000 reads from MLOps engineers. **This is how recruiters find him.**

---

## 10. Freelance & Job Application Timeline

```
Month 1-2  ── Heads down. NO applying yet. Build Project 1 quietly.
              Polish LinkedIn profile; don't announce career change yet.

Month 3    ── Project 2 done → publish blog → first LinkedIn post.
              Update headline to "DevOps Engineer | building MLOps systems".
              Start collecting connections from ML engineers in target companies.

Month 4    ── Project 3 done. Start applying to ₹70L+ Senior MLOps roles.
              Even if not switching, interviewing reveals gaps to fix.

Month 5    ── Project 4 (Crown Jewel) shipping → Twitter/X thread.
              Cold outreach to ML platform engineers at target companies. Most reply.

Month 6    ── Project 5 done. THE portfolio is complete.

              Two-track strategy:
                (A) Interview for ₹70L-1Cr Senior MLOps roles in India
                (B) Apply on Toptal / Arc.dev / Braintrust for freelance ($80-120/hr)

              Realistic outcome by Month 8-9:
                Signed offer at ₹70-90L OR 2-3 freelance clients @ ~$80/hr.

Year 1.5-2 ── ₹1 Cr crossed via:
                (Path A) Senior MLOps role + 5-10 hrs/wk freelance side income, OR
                (Path B) Remote US/EU MLOps contract directly.
```

**Target companies (Month 4+):**

- *India product:* Razorpay, Atlassian India, Stripe India, Swiggy, Zomato, Hasura, Postman, Freshworks, Vercel India
- *Indian unicorns:* Cred, PhonePe, Meesho, Acko, Zerodha
- *Remote US/EU MLOps-focused:* TruEra, Comet, Arize AI, Determined AI, dbt Labs, Modal, Replicate
- *Freelance platforms:* Toptal, Arc.dev, Braintrust

**Critical mindset note:** He will not *feel* ready to apply at Month 4. Apply anyway. The interview process *is* learning. He'll land the right role around the time he stops feeling like an impostor.

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Stall on Project 4 (it's the hardest) | Break into 6 weekly milestones, each independently GitHub-pushable. |
| LLM API costs blow up during Project 3 | Heavy prompt + semantic caching + Ollama for dev + cost tracking from day 1. |
| Burnout around Month 3-4 | 2 rest days/week protected. Buffer weeks baked in. Pause-don't-quit principle. |
| Feels "behind" vs Kaggle grandmasters | Not the game being played. MLOps engineer ≠ Kaggler. Different specialty. |
| No client takes him seriously without ML PhD | Project 4 (platform engineering) bypasses this entirely. |
| Missed weekday → guilt → quit | "1 commit minimum" rule. Even a README polish counts. Streak > perfection. |
| Local hardware can't run Project 4 fully | All components individually fit. Test resource budget per service in Week 1 setup. |

---

## 12. Success Metrics (Month 6 Gate)

All must hit:

- ✅ 5 polished GitHub projects, each with `make demo`, BLOG.md, architecture diagram
- ✅ 5 blog posts published (dev.to + LinkedIn)
- ✅ 500+ LinkedIn followers from organic ML/DevOps content
- ✅ At least 1 GitHub repo with >50 stars (Project 4 candidate)
- ✅ 3+ active interview pipelines at target roles (₹70L+)
- ✅ Comfortable explaining each architecture decision in a 30-min technical interview

---

## 13. Repository Structure

```
mlops-journey/
├── docs/
│   └── superpowers/specs/
│       └── 2026-05-27-mlops-agentic-ai-learning-path-design.md  ← this file
├── project-1-credit-risk-pipeline/         (Fintech)
├── project-2-sre-copilot/                  (DevOps Tooling)
├── project-3-medical-rag/                  (Healthcare)
├── project-4-ml-platform-k8s/              (Platform Engineering — Crown Jewel)
├── project-5-pricing-agent/                (E-commerce)
├── STATUS.md                               ← current week, project, blockers
└── README.md                               ← portfolio landing page
```

Each project subdirectory is later optionally split into its own repo when polishing for portfolio (pinned repos look better than a monorepo).

---

## 14. Persistence (Cross-Session Memory)

The user has explicitly requested that all context persist across Claude sessions. Memory is maintained in:

`/home/himanshu/.claude/projects/-home-himanshu-learning-mlops-journey/memory/`

Files:
- `MEMORY.md` — index of all memory files
- `user_profile.md`, `user_career_goal.md`, `user_hardware.md`
- `feedback_learning_style.md`, `feedback_schedule.md`, `feedback_budget.md`, `feedback_persistent_context.md`
- `project_mlops_overview.md`, `project_curriculum_plan.md`, `project_five_projects.md`, `project_role_target.md`, `project_design_spec.md`
- `reference_tooling_stack.md`

**At session start:** Claude reads `MEMORY.md` and relevant files, checks `STATUS.md` in repo for current week, and resumes without forcing the user to re-explain context.

---

## 15. Next Step (After This Spec Is Approved)

Invoke the `superpowers:writing-plans` skill to produce a detailed week-by-week implementation plan (Week 1 setup tasks, exact files to create, exact commands to run). That plan will live alongside this spec at:

`docs/superpowers/specs/2026-05-27-mlops-agentic-ai-learning-path-plan.md`

After plan approval, Week 1 begins immediately.

---

*End of design document.*
