"""R2 upload helper. S3-compatible; credentials come from env (handler-only)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import boto3
from botocore.client import Config

log = logging.getLogger("sam3.storage")


def _client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
        region_name="auto",
    )


def upload(path: Path, key: str, content_type: str | None = None) -> str:
    """Upload file to R2 and return its public CDN URL."""
    bucket = os.environ["R2_OUTPUT_BUCKET"]
    base = os.environ.get("R2_PUBLIC_BASE", "").rstrip("/")
    extra: dict = {}
    if content_type:
        extra["ContentType"] = content_type
    try:
        _client().upload_file(str(path), bucket, key, ExtraArgs=extra or None)
    except Exception as e:
        # Surface as retryable upstream
        from pipeline import PipelineFailure
        raise PipelineFailure("ERR_UPLOAD_FAILED", f"R2 upload failed: {e}", {"key": key}) from e
    return f"{base}/{key}" if base else f"r2://{bucket}/{key}"
