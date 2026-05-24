"""FFmpeg-based encoder + compositor.

Given a source video and an alpha-matte video (single-channel grayscale MP4
produced by MatAnyone), produces the final output in the requested format.

Strategy by output format:
- webm: VP9 with yuva420p — uses ffmpeg's alphamerge to fuse source RGB + alpha.
- mov:  ProRes 4444 yuva444p10le — same alphamerge but mov container.
- mp4:  No alpha. Composites RGB onto bg.type (color/image/video).
- gif:  Palette-quantized; supports alpha for transparent type.
- webp: Animated WebP with alpha.
- png_sequence: extract per-frame RGBA PNGs, zip them up.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from schema import BackgroundSpec, OutputSpec

log = logging.getLogger("sam3.encoder")


QUALITY_CRF = {"draft": 30, "standard": 23, "high": 18, "lossless": 0}
QUALITY_VPX_CRF = {"draft": 36, "standard": 30, "high": 24, "lossless": 4}


def _run(cmd: list[str]) -> None:
    log.info("ffmpeg: %s", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=15 * 60)
    if p.returncode != 0:
        # Surface stderr (last 2k) so failures are diagnosable in RunPod logs
        raise RuntimeError(
            f"ffmpeg failed (rc={p.returncode}):\n--- stderr (tail) ---\n{p.stderr[-2000:]}"
        )


def _resize_filter(max_dim: int | None) -> str:
    """Return a scale filter that constrains the longest edge while keeping aspect."""
    if not max_dim or max_dim <= 0:
        return ""
    return f"scale='if(gt(iw,ih),min({max_dim},iw),-2)':'if(gt(iw,ih),-2,min({max_dim},ih))'"


def _bg_color_hex(rgb: list[float] | None) -> str:
    if not rgb or len(rgb) != 3:
        return "0x000000"
    r, g, b = (int(max(0, min(1, v)) * 255) for v in rgb)
    return f"0x{r:02X}{g:02X}{b:02X}"


def encode(
    *,
    source: Path,
    alpha: Path,
    background: BackgroundSpec,
    output: OutputSpec,
    bg_image: Path | None,
    bg_video: Path | None,
    dest: Path,
    has_audio: bool,
) -> None:
    """Compose source+alpha into dest in the requested format.

    `alpha` is a grayscale MP4 from MatAnyone (R=G=B=alpha).
    `source` is the original color video; `has_audio` says whether to passthrough.
    """
    fmt = output.format
    crf = str(QUALITY_CRF.get(output.quality, 18))
    vpx_crf = str(QUALITY_VPX_CRF.get(output.quality, 24))
    scale = _resize_filter(output.max_dimension)
    audio_args: list[str] = []
    if output.preserve_audio and has_audio and fmt in {"webm", "mov", "mp4"}:
        # Map source audio (input #0 audio) and re-encode to a container-safe codec.
        audio_args = [
            "-map", "0:a:0?",
            "-c:a", "libopus" if fmt == "webm" else "aac",
            "-b:a", "128k",
        ]

    # Alpha-bearing transparent output: webm/mov/png_sequence/gif/webp + bg.type=transparent
    transparent = background.type == "transparent"

    base = ["ffmpeg", "-y", "-i", str(source), "-i", str(alpha)]

    if fmt == "webm" and transparent:
        # Take source RGB, alpha from alpha video luma → yuva420p
        # Scale alpha to match source dims (MatAnyone may downscale alpha).
        filt = (
            "[1:v][0:v]scale2ref=w=iw:h=ih[a0][c0];"
            "[c0]format=gbrap,setpts=PTS-STARTPTS[c];"
            "[a0]format=gray,setpts=PTS-STARTPTS[a];"
            "[c][a]alphamerge"
        )
        if scale:
            filt += f",{scale}"
        filt += ",format=yuva420p[v]"
        cmd = base + [
            "-filter_complex", filt,
            "-map", "[v]", *audio_args,
            "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
            "-crf", vpx_crf, "-b:v", "0", "-auto-alt-ref", "0",
            str(dest),
        ]
    elif fmt == "mov" and transparent:
        filt = (
            "[1:v][0:v]scale2ref=w=iw:h=ih[a0][c0];"
            "[c0]format=gbrap,setpts=PTS-STARTPTS[c];"
            "[a0]format=gray,setpts=PTS-STARTPTS[a];"
            "[c][a]alphamerge"
        )
        if scale:
            filt += f",{scale}"
        filt += ",format=yuva444p10le[v]"
        cmd = base + [
            "-filter_complex", filt,
            "-map", "[v]", *audio_args,
            "-c:v", "prores_ks", "-profile:v", "4444",
            str(dest),
        ]
    elif fmt == "mp4" or not transparent:
        # No alpha. Composite RGB onto background.
        bg_filter = _build_bg_filter(background, bg_image, bg_video, scale)
        # bg_filter ends with [bg]; we then overlay [fg] with alpha mask
        filt = (
            f"{bg_filter};"
            "[0:v]format=yuv420p,setpts=PTS-STARTPTS[fg];"
            "[1:v]format=gray,setpts=PTS-STARTPTS[mask];"
            "[fg][mask]alphamerge[fga];"
            "[bg][fga]overlay=shortest=1:format=auto"
        )
        if scale and "scale=" not in filt:
            filt += f",{scale}"
        filt += ",format=yuv420p[v]"
        cmd = base + (
            ["-i", str(bg_image) if bg_image else (str(bg_video) if bg_video else "")]
            if bg_image or bg_video else []
        ) + [
            "-filter_complex", filt,
            "-map", "[v]", *audio_args,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", crf, "-preset", "medium",
            str(dest),
        ]
    elif fmt == "gif":
        filt = (
            "[0:v]format=rgba[c];"
            "[1:v]format=gray[a];"
            "[c][a]alphamerge,split[s1][s2];"
            "[s1]palettegen=reserve_transparent=1[pal];"
            "[s2][pal]paletteuse=alpha_threshold=128"
        )
        cmd = base + [
            "-filter_complex", filt,
            str(dest),
        ]
    elif fmt == "webp":
        filt = (
            "[1:v][0:v]scale2ref=w=iw:h=ih[a0][c0];"
            "[c0]format=rgba[c];"
            "[a0]format=gray[a];"
            "[c][a]alphamerge"
        )
        if scale:
            filt += f",{scale}"
        cmd = base + [
            "-filter_complex", filt,
            "-vcodec", "libwebp", "-lossless", "0",
            "-quality", "80", "-loop", "0", "-an",
            str(dest),
        ]
    elif fmt == "png_sequence":
        # Extract per-frame RGBA PNGs, then zip
        with tempfile.TemporaryDirectory(prefix="png-seq-") as td:
            tdp = Path(td)
            filt = (
                "[0:v]format=rgba[c];"
                "[1:v]format=gray[a];"
                "[c][a]alphamerge"
            )
            if scale:
                filt += f",{scale}"
            _run(
                base + [
                    "-filter_complex", filt,
                    str(tdp / "frame_%06d.png"),
                ]
            )
            with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for p in sorted(tdp.glob("*.png")):
                    zf.write(p, arcname=p.name)
        return
    else:
        raise ValueError(f"Unsupported output format: {fmt}")

    _run(cmd)


def _build_bg_filter(
    bg: BackgroundSpec,
    bg_image: Path | None,
    bg_video: Path | None,
    scale: str,
) -> str:
    """Return a filter graph fragment producing a [bg] stream.

    Always sized to the source's resolution (no explicit scaling here — overlay
    handles dimension matching via the source's stream as anchor).
    """
    if bg.type == "color":
        color = _bg_color_hex(bg.color)
        return f"color=c={color}:size=1920x1080,format=yuv420p,setpts=PTS-STARTPTS[bg]"
    if bg.type == "image" and bg_image:
        return f"movie={bg_image},loop=loop=-1:size=1,setpts=N/FRAME_RATE/TB[bg]"
    if bg.type == "video" and bg_video:
        return f"movie={bg_video},setpts=PTS-STARTPTS,loop=loop=-1:size=99999[bg]"
    # Default to black
    return "color=c=black:size=1920x1080,format=yuv420p[bg]"


def has_audio_stream(path: Path) -> bool:
    """Probe whether the source has any audio stream."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                str(path),
            ],
            text=True, timeout=30,
        )
        return "audio" in out
    except Exception:
        return False


def probe_dimensions(path: Path) -> tuple[int, int, float, int]:
    """Return (width, height, fps, frame_count) for path."""
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
            "-of", "default=nw=1:nk=1",
            str(path),
        ],
        text=True, timeout=30,
    ).strip().splitlines()
    width = int(out[0]) if len(out) > 0 else 0
    height = int(out[1]) if len(out) > 1 else 0
    fps_raw = out[2] if len(out) > 2 else "0/1"
    nb_frames = int(out[3]) if len(out) > 3 and out[3].isdigit() else 0
    num, den = (fps_raw.split("/") + ["1"])[:2]
    try:
        fps = float(num) / float(den) if float(den) else 0
    except ValueError:
        fps = 0
    return width, height, fps, nb_frames
