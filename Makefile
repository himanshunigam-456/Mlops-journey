.PHONY: help verify up down logs clean ps

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

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

clean:  ## Stop stack AND remove data volumes (DESTRUCTIVE)
	cd infra && docker compose down -v
	# Bind-mounted data dirs are root-owned (created by containerized
	# Postgres/MinIO). Use Docker (also root) to remove them instead of sudo.
	docker run --rm -v $$(pwd)/infra/data:/data alpine sh -c "rm -rf /data/* /data/.[!.]*" 2>/dev/null || true
	@echo "Local data wiped. Stack is at fresh-install state."
