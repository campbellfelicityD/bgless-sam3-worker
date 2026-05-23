"""Pydantic input/output models — mirrors docs/sam3-integration/api-contract.md §1, §3."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, model_validator


# ─── Prompt ───────────────────────────────────────────────────────────────────

class PromptInput(BaseModel):
    mode: Literal["auto", "text", "box", "point", "mask"] = "auto"
    text: Optional[str] = Field(default=None, max_length=200)
    box: Optional[list[list[float]]] = None  # [[x1,y1,x2,y2], ...] normalized 0-1
    points: Optional[list[dict]] = None       # [{"x":.., "y":.., "label": 0|1}, ...]
    mask_url: Optional[HttpUrl] = None
    frame_index: int = 0
    negative_text: Optional[str] = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _require_mode_payload(self) -> "PromptInput":
        if self.mode == "text" and not self.text:
            raise ValueError("prompt.text required when mode='text'")
        if self.mode == "box" and not self.box:
            raise ValueError("prompt.box required when mode='box'")
        if self.mode == "point" and not self.points:
            raise ValueError("prompt.points required when mode='point'")
        if self.mode == "mask" and not self.mask_url:
            raise ValueError("prompt.mask_url required when mode='mask'")
        if self.box:
            for b in self.box:
                if len(b) != 4 or not (b[0] < b[2] and b[1] < b[3]):
                    raise ValueError("prompt.box entries must be [x1,y1,x2,y2] with x2>x1 & y2>y1")
                if not all(0.0 <= v <= 1.0 for v in b):
                    raise ValueError("prompt.box coordinates must be normalized 0..1")
        return self


# ─── Output / Background / Refine ─────────────────────────────────────────────

class OutputSpec(BaseModel):
    format: Literal["webm", "mov", "mp4", "gif", "webp", "png_sequence"] = "webm"
    max_dimension: int = Field(default=1920, ge=256, le=4096)
    fps: Optional[float] = Field(default=None, ge=6, le=120)
    preserve_audio: bool = True
    quality: Literal["draft", "standard", "high", "lossless"] = "high"


class BackgroundSpec(BaseModel):
    type: Literal["transparent", "color", "image", "video"] = "transparent"
    color: Optional[list[float]] = None  # [r,g,b] 0-1
    image_url: Optional[HttpUrl] = None
    video_url: Optional[HttpUrl] = None
    fit: Literal["cover", "contain", "stretch"] = "cover"
    blur_original: bool = False

    @model_validator(mode="after")
    def _consistency(self) -> "BackgroundSpec":
        if self.type == "color" and (not self.color or len(self.color) != 3):
            raise ValueError("background.color required as [r,g,b] when type='color'")
        if self.type == "image" and not self.image_url:
            raise ValueError("background.image_url required when type='image'")
        if self.type == "video" and not self.video_url:
            raise ValueError("background.video_url required when type='video'")
        return self


class RefineSpec(BaseModel):
    matting_model: Literal["none", "guided_filter", "matanyone"] = "matanyone"
    edge_smoothing: float = Field(default=0.5, ge=0.0, le=1.0)
    feather_px: int = Field(default=0, ge=0, le=32)
    trimap_dilate: int = Field(default=8, ge=1, le=64)
    stability_threshold: float = Field(default=0.9, ge=0.0, le=1.0)


# ─── Top-level input ─────────────────────────────────────────────────────────

class HandlerInput(BaseModel):
    version: Literal["1"] = "1"
    video_url: HttpUrl
    job_id: str

    model: Literal["sam3-tiny", "sam3-base", "sam3-pro", "sam3-human", "rvm-light"] = "sam3-pro"
    preview: bool = False
    preview_duration: float = Field(default=2.0, gt=0, le=30)

    prompt: PromptInput = Field(default_factory=PromptInput)
    output: OutputSpec = Field(default_factory=OutputSpec)
    background: BackgroundSpec = Field(default_factory=BackgroundSpec)
    refine: RefineSpec = Field(default_factory=RefineSpec)

    callback_url: Optional[HttpUrl] = None
    progress_callback: bool = False

    @model_validator(mode="after")
    def _model_specific(self) -> "HandlerInput":
        # sam3-pro forces matanyone
        if self.model == "sam3-pro" and self.refine.matting_model == "none":
            self.refine.matting_model = "matanyone"
        return self


# ─── Output types ─────────────────────────────────────────────────────────────

class PipelineStats(BaseModel):
    pipeline_ms: dict[str, int]
    gpu_peak_memory_mb: int
    model_version: str
    low_confidence_frames: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ResultPayload(BaseModel):
    output_url: HttpUrl
    output_format: str
    duration_seconds: float
    frame_count: int
    width: int
    height: int
    file_size_bytes: int
    has_alpha: bool
    preview: bool


class HandlerOutput(BaseModel):
    version: Literal["1"] = "1"
    job_id: str
    result: ResultPayload
    stats: PipelineStats
    debug: Optional[dict] = None


# ─── Error codes ──────────────────────────────────────────────────────────────

class HandlerError(BaseModel):
    error: str  # one of ERR_* — see api-contract.md §4
    message: str
    retryable: bool
    details: dict = Field(default_factory=dict)


ERROR_CODES = {
    "ERR_VIDEO_TOO_LARGE": False,
    "ERR_VIDEO_TOO_LONG": False,
    "ERR_DOWNLOAD_TIMEOUT": True,
    "ERR_DOWNLOAD_FAILED": True,
    "ERR_DOWNLOAD_BACKGROUND_FAILED": True,
    "ERR_INVALID_PROMPT": False,
    "ERR_INVALID_INPUT": False,
    "ERR_NO_SUBJECT_FOUND": False,
    "ERR_MODEL_LOAD_FAILED": True,
    "ERR_GPU_OOM": True,
    "ERR_FFMPEG_ENCODE_FAILED": True,
    "ERR_UPLOAD_FAILED": True,
    "ERR_UNKNOWN": True,
}
