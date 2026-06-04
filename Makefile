.PHONY: help verify up down logs clean ps

help:  ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

verify:  ## Run infrastructure smoke tests (target: 18/18)
	@bash infra/verify.sh

up:  ## Start docker-compose stack (postgres, minio, redis, mlflow)
	cd infra && docker compose up -d
	@echo ""
	@echo "  MLflow UI: http://localhost:5000"
	@if [ -f infra/.env ]; then \
		MINIO_USER=$$(grep '^MINIO_ROOT_USER=' infra/.env | cut -d= -f2); \
		MINIO_PASS=$$(grep '^MINIO_ROOT_PASSWORD=' infra/.env | cut -d= -f2); \
		echo "  MinIO UI:  http://localhost:9001  ($$MINIO_USER / $$MINIO_PASS)"; \
	else \
		echo "  MinIO UI:  http://localhost:9001  (see infra/.env for credentials)"; \
	fi

down:  ## Stop docker-compose stack
	cd infra && docker compose down

logs:  ## Tail logs from docker-compose stack
	cd infra && docker compose logs -f --tail=50

ps:  ## Show running containers in the stack
	cd infra && docker compose ps

demo:  ## End-to-end: start stack, train baseline model, log to MLflow
	@$(MAKE) up
	@echo "── Waiting for MLflow ──"
	@for i in 1 2 3 4 5 6 7 8 9 10; do \
		curl -sf http://localhost:5000/health > /dev/null && break || sleep 5; \
	done
	@echo "── Running baseline notebook ──"
	.venv/bin/jupyter nbconvert --to notebook --execute \
		project-0-warmup/notebooks/credit_baseline.ipynb \
		--output credit_baseline.ipynb
	@echo ""
	@echo "✅ Done. Browse:"
	@echo "   MLflow UI: http://localhost:5000  (experiment 'credit-baseline')"
	@echo "   MinIO UI:  http://localhost:9001  (bucket 'mlflow/')"

# ── Project 1 — Credit Risk Pipeline ──

p1-data:  ## Pull project-1 raw data from MinIO (DVC)
	cd project-1-credit-risk-pipeline && ../.venv/bin/dvc pull

p1-train:  ## Run the 3-trial hyperparameter sweep, log all to MLflow
	cd project-1-credit-risk-pipeline && ../.venv/bin/python -m credit_risk.cli train

p1-register:  ## Promote the best run to Model Registry @ Staging
	cd project-1-credit-risk-pipeline && ../.venv/bin/python -m credit_risk.cli register

p1-test:  ## Run all project-1 tests
	.venv/bin/pytest project-1-credit-risk-pipeline -v

# ── Project 1 Phase 2 — Serving ──

p2-serve:  ## Run the FastAPI app locally on :8000 against live MLflow
	cd project-1-credit-risk-pipeline && \
	  ../.venv/bin/uvicorn credit_risk.serving.app:app --host 0.0.0.0 --port 8000 --reload

p2-docker-build:  ## Build the credit-risk-api Docker image
	cd project-1-credit-risk-pipeline && \
	  docker build -f serving/Dockerfile -t credit-risk-api:dev .

p2-docker-run:  ## Run the container against the compose stack
	docker run --rm -d --name credit-risk-api \
	  --network infra_default \
	  -p 8000:8000 \
	  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
	  -e AWS_ACCESS_KEY_ID=$$(grep '^MINIO_ROOT_USER=' infra/.env | cut -d= -f2) \
	  -e AWS_SECRET_ACCESS_KEY=$$(grep '^MINIO_ROOT_PASSWORD=' infra/.env | cut -d= -f2) \
	  -e MLFLOW_S3_ENDPOINT_URL=http://minio:9000 \
	  credit-risk-api:dev
	@echo "Container started — http://localhost:8000/docs"

p2-docker-stop:  ## Stop the credit-risk-api container
	-docker stop credit-risk-api

p2-load-test:  ## Run a 30s Locust load test against http://localhost:8000
	cd project-1-credit-risk-pipeline && \
	  ../.venv/bin/locust -f serving/locustfile.py --headless \
	    --users 10 --spawn-rate 2 --run-time 30s \
	    --host http://localhost:8000

p2-test:  ## Run only the Phase 2 (serving) tests
	.venv/bin/pytest project-1-credit-risk-pipeline/tests/test_serving_schemas.py \
	                 project-1-credit-risk-pipeline/tests/test_serving_app.py \
	                 project-1-credit-risk-pipeline/tests/test_serving_smoke.py -v

clean:  ## Stop stack AND remove data volumes (DESTRUCTIVE)
	cd infra && docker compose down -v
	# Bind-mounted data dirs are root-owned (created by containerized
	# Postgres/MinIO). Use Docker (also root) to remove them instead of sudo.
	docker run --rm -v $$(pwd)/infra/data:/data alpine sh -c "rm -rf /data/* /data/.[!.]*" 2>/dev/null || true
	@echo "Local data wiped. Stack is at fresh-install state."
