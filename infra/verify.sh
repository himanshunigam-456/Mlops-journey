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
