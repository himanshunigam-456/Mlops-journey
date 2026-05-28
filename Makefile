.PHONY: help verify up down logs clean ps

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

verify:  ## Run infrastructure smoke tests (target: 18/18)
	@bash infra/verify.sh

up:  ## Start docker-compose stack (postgres, minio, redis, mlflow)
	cd infra && docker compose up -d
	@echo ""
	@echo "  MLflow UI: http://localhost:5000"
	@echo "  MinIO UI:  http://localhost:9001  (REDACTED-OLD-DEFAULT / REDACTED-OLD-DEFAULT)"

down:  ## Stop docker-compose stack
	cd infra && docker compose down

logs:  ## Tail logs from docker-compose stack
	cd infra && docker compose logs -f --tail=50

ps:  ## Show running containers in the stack
	cd infra && docker compose ps

clean:  ## Stop stack AND remove data volumes (DESTRUCTIVE)
	cd infra && docker compose down -v
	rm -rf infra/data/
	@echo "Local data wiped. Stack is at fresh-install state."
