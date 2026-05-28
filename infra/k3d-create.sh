#!/usr/bin/env bash
# Create the k3d-mlops cluster used by all 5 portfolio projects.
# Topology: 1 control-plane + 2 workers, ingress on host port 8080.
# Re-runnable: if the cluster already exists, this script no-ops.
set -euo pipefail

CLUSTER="${CLUSTER:-mlops}"

if k3d cluster list | grep -qE "^${CLUSTER}\b"; then
  echo "Cluster '${CLUSTER}' already exists. Nothing to do."
  exit 0
fi

echo "Creating k3d cluster '${CLUSTER}' ..."
k3d cluster create "${CLUSTER}" \
  --servers 1 \
  --agents 2 \
  --port "8080:80@loadbalancer" \
  --port "8443:443@loadbalancer"

echo ""
echo "Waiting for all nodes Ready ..."
kubectl wait --for=condition=Ready nodes --all --timeout=120s

echo ""
echo "Cluster ready. Context: $(kubectl config current-context)"
kubectl get nodes
