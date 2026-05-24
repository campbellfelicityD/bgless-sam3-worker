"""R2 upload helper. S3-compatible.

Credential resolution order (per key):
  1. env var
  2. `/runpod-volume/.r2_env`  (KEY=VALUE lines, populated via RunPod S3 API)
  3. `/workspace/.r2_env`
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import boto3
from botocore.client import Config

log = logging.getLogger("sam3.storage")


_FALLBACK_PATHS = (
    "/runpod-volume/.r2_env",
    "/workspace/.r2_env",
)
_R2_KEYS = (
    "R2_ENDPOINT",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_OUTPUT_BUCKET",
    "R2_PUBLIC_BASE",
)


def _load_fallback_env() -> None:
    """Populate missing R2_* env vars from /runpod-volume/.r2_env if available.

    The fallback file is a KEY=VALUE dotenv-style file. Existing env vars win.
    """
    missing = [k for k in _R2_KEYS if not os.environ.get(k)]
    if not missing:
        return
    for p in _FALLBACK_PATHS:
        try:
            content = Path(p).read_text()
        except Exception:  # noqa: BLE001
            continue
        loaded = 0
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k in _R2_KEYS and not os.environ.get(k):
                os.environ[k] = v
                loaded += 1
        if loaded:
            log.info("loaded %d R2 vars from %s", loaded, p)
            break


def _require(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(
            f"{key} missing from env and from {_FALLBACK_PATHS}"
        )
    return val


def _client():
    _load_fallback_env()
    return boto3.client(
        "s3",
        endpoint_url=_require("R2_ENDPOINT"),
        aws_access_key_id=_require("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=_require("R2_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
        region_name="auto",
    )


def upload(path: Path, key: str, content_type: str | None = None) -> str:
    """Upload file to R2 and return its public CDN URL."""
    _load_fallback_env()
    bucket = _require("R2_OUTPUT_BUCKET")
    base = os.environ.get("R2_PUBLIC_BASE", "").rstrip("/")
    extra: dict = {}
    if content_type:
        extra["ContentType"] = content_type
    try:
        _client().upload_file(str(path), bucket, key, ExtraArgs=extra or None)
    except Exception as e:
        from pipeline import PipelineFailure
        raise PipelineFailure("ERR_UPLOAD_FAILED", f"R2 upload failed: {e}", {"key": key}) from e
    return f"{base}/{key}" if base else f"r2://{bucket}/{key}"
