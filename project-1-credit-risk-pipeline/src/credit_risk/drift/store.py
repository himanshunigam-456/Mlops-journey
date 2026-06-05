"""Read/write Parquet + HTML to MinIO via boto3.

Two buckets:
  - `predictions` — daily partitions of captured production scoring rows
  - `reports`     — training reference snapshot + latest drift report

Reuses the same AWS env vars our FastAPI container uses
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `MLFLOW_S3_ENDPOINT_URL`).
"""

from __future__ import annotations

import io
import os
from datetime import date

import boto3
import pandas as pd

PREDICTIONS_BUCKET = "predictions"
REPORTS_BUCKET = "reports"


def _s3_client():
    """boto3 S3 client pointed at our MinIO endpoint."""
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def ensure_bucket(name: str) -> None:
    """Create the bucket if it does not exist. Idempotent."""
    s3 = _s3_client()
    existing = {b["Name"] for b in s3.list_buckets().get("Buckets", [])}
    if name not in existing:
        s3.create_bucket(Bucket=name)


def write_predictions_parquet(df: pd.DataFrame, day: date | None = None) -> str:
    """Append today's predictions to s3://predictions/YYYY-MM-DD.parquet.

    If the day's file already exists, read it, append, and overwrite — keeps
    one Parquet per day. Simpler than partitioned writes for our scale.
    """
    ensure_bucket(PREDICTIONS_BUCKET)
    day = day or date.today()
    key = f"{day.isoformat()}.parquet"
    s3 = _s3_client()

    try:
        existing = read_predictions_for_day(day)
        df = pd.concat([existing, df], ignore_index=True)
    except Exception:
        # First write of the day → no prior file, that's fine
        pass

    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=PREDICTIONS_BUCKET, Key=key, Body=buf.read())
    return f"s3://{PREDICTIONS_BUCKET}/{key}"


def read_predictions_for_day(day: date) -> pd.DataFrame:
    """Read a single day's predictions Parquet. Raises if missing."""
    s3 = _s3_client()
    obj = s3.get_object(Bucket=PREDICTIONS_BUCKET, Key=f"{day.isoformat()}.parquet")
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def read_predictions_range(start: date, end: date) -> pd.DataFrame:
    """Read all daily Parquets in [start, end] inclusive; concat. Missing days are skipped."""
    days = pd.date_range(start, end).date
    parts = []
    for d in days:
        try:
            parts.append(read_predictions_for_day(d))
        except Exception:
            continue
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def write_report_html(name: str, html: str) -> str:
    """Write an HTML report to s3://reports/<name>."""
    ensure_bucket(REPORTS_BUCKET)
    s3 = _s3_client()
    s3.put_object(Bucket=REPORTS_BUCKET, Key=name, Body=html.encode("utf-8"))
    return f"s3://{REPORTS_BUCKET}/{name}"


def read_report_html(name: str) -> str:
    """Read an HTML report from s3://reports/<name>."""
    s3 = _s3_client()
    obj = s3.get_object(Bucket=REPORTS_BUCKET, Key=name)
    return obj["Body"].read().decode("utf-8")
