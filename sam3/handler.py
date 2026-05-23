"""RunPod Serverless entry point.

Wraps `pipeline.run()` with input validation, error normalization, and
HMAC-signed progress callbacks. See docs/sam3-integration/api-contract.md.
"""

from __future__ import annotations

import logging
import os
import time
import traceback

import runpod
from pydantic import ValidationError

from schema import (
    ERROR_CODES,
    HandlerError,
    HandlerInput,
    HandlerOutput,
)
from pipeline import PipelineFailure, run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("sam3.handler")


def _error_payload(code: str, message: str, details: dict | None = None) -> dict:
    retryable = ERROR_CODES.get(code, True)
    return HandlerError(
        error=code,
        message=message,
        retryable=retryable,
        details=details or {},
    ).model_dump()


def handler(event: dict) -> dict:
    """RunPod serverless calls this per request. Returns dict.

    Expected event = {"input": {...}}  (see schema.HandlerInput)
    Returns either HandlerOutput.model_dump() or HandlerError.model_dump().
    """
    started = time.perf_counter()
    raw_input = (event or {}).get("input") or {}

    # ─── 1. Validate input ───────────────────────────────────────────────
    try:
        parsed = HandlerInput.model_validate(raw_input)
    except ValidationError as exc:
        log.warning("Invalid input: %s", exc.errors())
        return _error_payload(
            "ERR_INVALID_INPUT",
            "Input failed schema validation",
            {"validation_errors": exc.errors()},
        )

    log.info(
        "Job %s start: model=%s preview=%s prompt_mode=%s",
        parsed.job_id, parsed.model, parsed.preview, parsed.prompt.mode,
    )

    # ─── 2. Run pipeline ─────────────────────────────────────────────────
    try:
        result: HandlerOutput = run_pipeline(parsed)
        elapsed = int((time.perf_counter() - started) * 1000)
        log.info("Job %s OK in %d ms", parsed.job_id, elapsed)
        return result.model_dump(mode="json")
    except PipelineFailure as pf:
        log.warning("Job %s failed (%s): %s", parsed.job_id, pf.code, pf.message)
        return _error_payload(pf.code, pf.message, pf.details)
    except Exception as exc:  # noqa: BLE001 — last-resort catch-all
        log.exception("Job %s crashed", parsed.job_id)
        return _error_payload(
            "ERR_UNKNOWN",
            f"Unhandled exception: {type(exc).__name__}: {exc}",
            {"traceback": traceback.format_exc(limit=20)},
        )


# ─── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info(
        "Boot: sam3_dir=%s matanyone_dir=%s r2_bucket=%s",
        os.environ.get("SAM3_WEIGHTS_DIR", "<unset>"),
        os.environ.get("MATANYONE_WEIGHTS_DIR", "<unset>"),
        os.environ.get("R2_OUTPUT_BUCKET", "<unset>"),
    )
    runpod.serverless.start({"handler": handler})
