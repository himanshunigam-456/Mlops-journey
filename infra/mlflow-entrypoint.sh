#!/bin/sh
# Install Postgres + S3 drivers at container start, with retry for transient DNS issues.
# Then exec the MLflow server command passed as args.
set -e

for i in $(seq 1 8); do
  if pip install --no-cache-dir --quiet psycopg2-binary boto3 2>&1 ; then
    echo "[entrypoint] deps installed on attempt $i"
    break
  fi
  echo "[entrypoint] pip install attempt $i failed, retrying in 5s..."
  sleep 5
done

echo "[entrypoint] starting: $@"
exec "$@"
