"""End-to-end SAM3 + MatAnyone pipeline.

  download → SAM3 first-frame mask (from text/auto prompt)
           → MatAnyone alpha refinement (mp4 alpha matte + fgr)
           → FFmpeg compose with background + encode to chosen format
           → R2 upload
           → return public CDN URL

Stages are timed and reported in HandlerOutput.stats. Errors are wrapped in
PipelineFailure so the RunPod handler maps them to consistent ERR_* codes.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from schema import HandlerInput, HandlerOutput, PipelineStats, ResultPayload

log = logging.getLogger("sam3.pipeline")


# ─── Errors ───────────────────────────────────────────────────────────────────

class PipelineFailure(Exception):
    """Carries an ERR_* code that maps to schema.ERROR_CODES."""

    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


# ─── Stage tracking ──────────────────────────────────────────────────────────

@dataclass
class StageTimings:
    timings_ms: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    low_conf_frames: list[int] = field(default_factory=list)
    gpu_peak_mb: int = 0
    model_version: str = ""

    def time(self, name: str):
        return _StageTimer(self, name)


class _StageTimer:
    def __init__(self, holder: StageTimings, name: str):
        self.holder = holder
        self.name = name

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed = int((time.perf_counter() - self._t0) * 1000)
        self.holder.timings_ms[self.name] = elapsed
        log.info("stage=%s elapsed_ms=%d", self.name, elapsed)


# ─── Constants ───────────────────────────────────────────────────────────────

DOWNLOAD_TIMEOUT_S = 60
MAX_INPUT_BYTES = 500 * 1024 * 1024  # 500 MB
DEFAULT_AUTO_PROMPT = "the main subject"


def _download(url: str, dest: Path) -> None:
    try:
        with requests.get(str(url), stream=True, timeout=DOWNLOAD_TIMEOUT_S) as r:
            r.raise_for_status()
            cl = r.headers.get("Content-Length")
            if cl and int(cl) > MAX_INPUT_BYTES:
                raise PipelineFailure(
                    "ERR_VIDEO_TOO_LARGE",
                    f"Source exceeds {MAX_INPUT_BYTES} bytes ({cl} declared)",
                )
            total = 0
            with dest.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_INPUT_BYTES:
                        raise PipelineFailure(
                            "ERR_VIDEO_TOO_LARGE",
                            f"Source streamed past {MAX_INPUT_BYTES} bytes",
                        )
                    f.write(chunk)
    except requests.Timeout as e:
        raise PipelineFailure("ERR_DOWNLOAD_TIMEOUT", str(e)) from e
    except requests.HTTPError as e:
        raise PipelineFailure(
            "ERR_DOWNLOAD_FAILED",
            f"HTTP {e.response.status_code} fetching source",
            {"status": e.response.status_code},
        ) from e
    except requests.RequestException as e:
        raise PipelineFailure("ERR_DOWNLOAD_FAILED", str(e)) from e


def _maybe_trim_to_preview(src: Path, dest: Path, preview_seconds: float) -> Path:
    """If preview=True, transcode the first preview_seconds to dest and return dest;
    otherwise return src unchanged."""
    if preview_seconds <= 0:
        return src
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-t", f"{preview_seconds:.2f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-c:a", "aac", "-b:a", "96k",
        str(dest),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        raise PipelineFailure(
            "ERR_FFMPEG_ENCODE_FAILED",
            f"Preview trim failed: {p.stderr[-800:]}",
        )
    return dest


# ─── Lazy model loaders (singletons per container) ───────────────────────────

_SAM3 = None  # type: ignore[var-annotated]
_MATANYONE = None  # type: ignore[var-annotated]


_HF_TOKEN_FALLBACK_PATHS = (
    "/runpod-volume/.hf_token",
    "/workspace/.hf_token",
    "/etc/hf_token",
)


def _ensure_hf_login() -> None:
    """Force huggingface_hub to authenticate even if RunPod env var injection
    failed to propagate HF_TOKEN to the worker process.

    Resolution order:
      1. `HF_TOKEN` env var
      2. `/runpod-volume/.hf_token` (provisioned via RunPod S3 API)
      3. `/workspace/.hf_token`
      4. `/etc/hf_token`
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        from pathlib import Path
        for p in _HF_TOKEN_FALLBACK_PATHS:
            try:
                t = Path(p).read_text().strip()
                if t:
                    token = t
                    log.info("loaded HF token from fallback file %s", p)
                    break
            except Exception:  # noqa: BLE001
                continue
    if not token:
        raise PipelineFailure(
            "ERR_MODEL_LOAD_FAILED",
            "HF_TOKEN missing from env and all fallback files "
            f"({list(_HF_TOKEN_FALLBACK_PATHS)})",
        )
    os.environ["HF_TOKEN"] = token
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        log.info("hf login OK (token len=%d)", len(token))
    except Exception as e:  # noqa: BLE001
        log.warning("hf login failed (non-fatal): %s", e)


def _load_sam3() -> Any:
    global _SAM3
    if _SAM3 is not None:
        return _SAM3
    _ensure_hf_login()
    try:
        from sam3.model_builder import build_sam3_video_predictor
    except ImportError as e:
        raise PipelineFailure("ERR_MODEL_LOAD_FAILED", f"sam3 import failed: {e}") from e
    try:
        _SAM3 = build_sam3_video_predictor()
        log.info("SAM3 video predictor loaded")
        return _SAM3
    except Exception as e:  # noqa: BLE001
        raise PipelineFailure(
            "ERR_MODEL_LOAD_FAILED",
            f"SAM3 build failed: {type(e).__name__}: {e}",
        ) from e


def _load_matanyone() -> Any:
    global _MATANYONE
    if _MATANYONE is not None:
        return _MATANYONE
    _ensure_hf_login()
    try:
        from matanyone import InferenceCore
    except ImportError as e:
        raise PipelineFailure(
            "ERR_MODEL_LOAD_FAILED", f"matanyone import failed: {e}"
        ) from e
    try:
        # HF repo id; weights cached under HF_HOME on the network volume
        _MATANYONE = InferenceCore("PeiqingYang/MatAnyone")
        log.info("MatAnyone loaded")
        return _MATANYONE
    except Exception as e:  # noqa: BLE001
        raise PipelineFailure(
            "ERR_MODEL_LOAD_FAILED",
            f"MatAnyone build failed: {type(e).__name__}: {e}",
        ) from e


# ─── SAM3 first-frame mask extraction ────────────────────────────────────────

def _resolve_prompt_text(inp: HandlerInput) -> str:
    """Map auto/text prompt modes to a text string SAM3 can consume.

    box/point/mask modes are not supported in v0.1 — degrade to auto.
    """
    p = inp.prompt
    if p.mode == "text" and p.text:
        return p.text.strip()
    if p.mode == "auto":
        return DEFAULT_AUTO_PROMPT
    # v0.1: degrade box/point/mask to auto with a warning surfaced upstream.
    return DEFAULT_AUTO_PROMPT


def _extract_first_frame_mask(
    video_path: Path, prompt_text: str, frame_index: int, score_threshold: float
) -> np.ndarray:
    """Run SAM3 over `video_path`, add text prompt at frame_index,
    union all matching instance masks, return a uint8 mask {0,255} of shape (H, W).
    """
    predictor = _load_sam3()

    # 1. Start session — SAM3 indexes frames of the video
    sess = predictor.handle_request({
        "type": "start_session",
        "resource_path": str(video_path),
    })
    session_id = sess["session_id"]
    log.info("SAM3 session=%s", session_id)

    try:
        # 2. Add text prompt at requested frame (typically 0)
        resp = predictor.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_index,
            "text": prompt_text,
        })
        out = resp.get("outputs") or {}
        binary_masks = out.get("out_binary_masks")
        if binary_masks is None or len(binary_masks) == 0:
            raise PipelineFailure(
                "ERR_NO_SUBJECT_FOUND",
                f"SAM3 found no instances matching prompt {prompt_text!r}",
            )
        probs = out.get("out_probs")
        # Filter by stability threshold if probs available
        if probs is not None:
            keep = [i for i, p in enumerate(np.asarray(probs).tolist()) if p >= score_threshold]
            if keep:
                binary_masks = np.asarray(binary_masks)[keep]
        # Union all instance masks
        masks_np = np.asarray(binary_masks)
        if masks_np.ndim == 3:
            union = np.any(masks_np.astype(bool), axis=0)
        else:
            union = masks_np.astype(bool)
        if not union.any():
            raise PipelineFailure(
                "ERR_NO_SUBJECT_FOUND",
                f"SAM3 returned an empty mask for prompt {prompt_text!r}",
            )
        return (union.astype(np.uint8) * 255)
    finally:
        # Free session memory so subsequent jobs don't blow up VRAM
        try:
            predictor.handle_request({
                "type": "close_session",
                "session_id": session_id,
            })
        except Exception:  # noqa: BLE001
            pass


# ─── MatAnyone alpha refinement ──────────────────────────────────────────────

def _refine_with_matanyone(
    video_path: Path, mask_png: Path, workdir: Path, max_dim: int
) -> Path:
    """Run MatAnyone given source + first-frame mask, return path to alpha-matte mp4."""
    processor = _load_matanyone()
    try:
        # Cap dimension to fit GPU memory; -1 means keep original.
        max_size = max_dim if 0 < max_dim < 4097 else -1
        _, alpha_path = processor.process_video(
            input_path=str(video_path),
            mask_path=str(mask_png),
            output_path=str(workdir),
            n_warmup=10,
            r_erode=10,
            r_dilate=10,
            max_size=max_size,
            save_image=False,
        )
        return Path(alpha_path)
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        if "out of memory" in msg.lower() or "OOM" in msg:
            raise PipelineFailure("ERR_GPU_OOM", msg) from e
        raise PipelineFailure("ERR_MODEL_LOAD_FAILED", msg) from e


# ─── Main pipeline ───────────────────────────────────────────────────────────

def run_pipeline(inp: HandlerInput) -> HandlerOutput:
    timings = StageTimings()
    workdir = Path(tempfile.mkdtemp(prefix=f"sam3-{inp.job_id}-"))
    src_path = workdir / "input.bin"
    final_src = src_path

    try:
        # 1. Download
        with timings.time("download"):
            _download(str(inp.video_url), src_path)

        # 1b. Trim to preview if requested
        if inp.preview:
            with timings.time("preview_trim"):
                final_src = _maybe_trim_to_preview(
                    src_path, workdir / "preview.mp4", inp.preview_duration
                )

        # Probe source for output metadata + audio passthrough decision
        from encoder import has_audio_stream, probe_dimensions, encode
        src_w, src_h, src_fps, src_frames = probe_dimensions(final_src)
        has_audio = has_audio_stream(final_src)
        log.info("source: %dx%d @ %.2ffps frames=%d audio=%s",
                 src_w, src_h, src_fps, src_frames, has_audio)

        # 2. SAM3 first-frame mask
        with timings.time("sam3_inference"):
            prompt_text = _resolve_prompt_text(inp)
            if inp.prompt.mode in {"box", "point", "mask"}:
                timings.warnings.append(
                    f"prompt.mode={inp.prompt.mode} not yet supported; fell back to text '{prompt_text}'"
                )
            mask_arr = _extract_first_frame_mask(
                final_src,
                prompt_text=prompt_text,
                frame_index=inp.prompt.frame_index,
                score_threshold=inp.refine.stability_threshold,
            )
            mask_png = workdir / "first_mask.png"
            Image.fromarray(mask_arr).save(mask_png)
            log.info("first-frame mask: %dx%d", mask_arr.shape[1], mask_arr.shape[0])

        # 3. Matting refinement
        alpha_mp4: Path
        if inp.refine.matting_model == "matanyone" and inp.model != "sam3-tiny":
            with timings.time("matting"):
                alpha_mp4 = _refine_with_matanyone(
                    final_src, mask_png, workdir, inp.output.max_dimension
                )
        else:
            # Fast path: replicate first-frame mask for every frame (poor quality but fast)
            with timings.time("matting_fast"):
                alpha_mp4 = _replicate_mask_as_video(
                    mask_png, src_fps or 30.0, src_frames or 30, workdir
                )

        # 4. Compose + encode
        out_ext = {
            "webm": "webm", "mov": "mov", "mp4": "mp4",
            "gif": "gif", "webp": "webp", "png_sequence": "zip",
        }[inp.output.format]
        out_path = workdir / f"output.{out_ext}"
        with timings.time("compose_encode"):
            encode(
                source=final_src,
                alpha=alpha_mp4,
                background=inp.background,
                output=inp.output,
                bg_image=None,  # TODO: download bg_image_url to a local path
                bg_video=None,
                dest=out_path,
                has_audio=has_audio,
            )

        # 5. Upload
        with timings.time("upload"):
            from storage import upload
            mime_map = {
                "webm": "video/webm", "mov": "video/quicktime",
                "mp4": "video/mp4", "gif": "image/gif",
                "webp": "image/webp", "png_sequence": "application/zip",
            }
            key = f"outputs/{inp.job_id}.{out_ext}"
            output_url = upload(out_path, key=key, content_type=mime_map[inp.output.format])

        # GPU peak (best-effort)
        try:
            import torch
            if torch.cuda.is_available():
                timings.gpu_peak_mb = int(torch.cuda.max_memory_allocated() / 1024 / 1024)
        except Exception:  # noqa: BLE001
            pass

        timings.model_version = (
            f"sam3@hf:facebook/sam3+matanyone@hf:PeiqingYang/MatAnyone"
            if inp.refine.matting_model == "matanyone" else
            "sam3@hf:facebook/sam3"
        )

        file_size = out_path.stat().st_size
        has_alpha = inp.background.type == "transparent" and inp.output.format in {
            "webm", "mov", "gif", "webp", "png_sequence"
        }

        return HandlerOutput(
            job_id=inp.job_id,
            result=ResultPayload(
                output_url=output_url,
                output_format=inp.output.format,
                duration_seconds=(src_frames / src_fps) if src_fps else 0.0,
                frame_count=src_frames,
                width=src_w,
                height=src_h,
                file_size_bytes=file_size,
                has_alpha=has_alpha,
                preview=inp.preview,
            ),
            stats=PipelineStats(
                pipeline_ms=timings.timings_ms,
                gpu_peak_memory_mb=timings.gpu_peak_mb,
                model_version=timings.model_version,
                low_confidence_frames=timings.low_conf_frames,
                warnings=timings.warnings,
            ),
        )
    finally:
        # Best-effort cleanup
        shutil.rmtree(workdir, ignore_errors=True)


def _replicate_mask_as_video(
    mask_png: Path, fps: float, frame_count: int, workdir: Path
) -> Path:
    """For sam3-tiny / matting=none: stretch the first-frame mask into a
    grayscale mp4 that the encoder can treat like a MatAnyone alpha output."""
    out = workdir / "fake_alpha.mp4"
    n = max(1, frame_count)
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-framerate", f"{fps:.3f}", "-t", f"{n / max(fps, 1):.2f}",
        "-i", str(mask_png),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-tune", "stillimage",
        "-crf", "12",
        str(out),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        raise PipelineFailure(
            "ERR_FFMPEG_ENCODE_FAILED",
            f"alpha replication failed: {p.stderr[-800:]}",
        )
    return out
