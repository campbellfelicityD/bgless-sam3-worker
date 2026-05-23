#!/usr/bin/env bash
# Initialize RunPod Network Volume with SAM3 + MatAnyone weights.
#
# Run this ONCE on a temporary CPU Pod with the target Network Volume
# mounted at /runpod-volume. After it finishes, all Serverless workers
# in the same region can read the weights without re-downloading.
#
# Usage on a Pod terminal:
#   export HF_TOKEN=hf_xxx          # SAM3 access required (gated repo)
#   bash init_volume.sh             # idempotent — re-runs only fetch missing files
#
# Layout produced under /runpod-volume:
#   /runpod-volume/sam3/sam3-tiny.pt
#   /runpod-volume/sam3/sam3-base.pt
#   /runpod-volume/sam3/sam3-large.pt
#   /runpod-volume/matanyone/matanyone.pth
#   /runpod-volume/hf-cache/        (HF_HOME, auto-populated on first model use)

set -euo pipefail

VOLUME_ROOT="${VOLUME_ROOT:-/runpod-volume}"
SAM3_DIR="${SAM3_WEIGHTS_DIR:-$VOLUME_ROOT/sam3}"
MATANYONE_DIR="${MATANYONE_WEIGHTS_DIR:-$VOLUME_ROOT/matanyone}"
HF_CACHE="${HF_HOME:-$VOLUME_ROOT/hf-cache}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN env var is required (huggingface.co/facebook/sam3 must be gated-approved for this token)" >&2
  exit 2
fi

if [[ ! -d "$VOLUME_ROOT" ]]; then
  echo "ERROR: $VOLUME_ROOT not mounted. Attach the Network Volume when launching the Pod." >&2
  exit 2
fi

mkdir -p "$SAM3_DIR" "$MATANYONE_DIR" "$HF_CACHE"
export HF_HOME="$HF_CACHE"

echo "==> Volume root: $VOLUME_ROOT"
df -h "$VOLUME_ROOT" | tail -n +2

# Install minimal deps needed to fetch from HuggingFace.
if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "==> Installing huggingface_hub CLI"
  pip install --quiet --upgrade "huggingface_hub[cli]>=0.24"
fi

huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true

# ─── SAM3 weights ────────────────────────────────────────────────────────────
# repo IDs may change; pin once Meta publishes stable filenames.
# Current expectation (Nov 2025 release):
#   facebook/sam3        → sam3.pt        (largest, ~3 GB)
#   facebook/sam3-base   → sam3-base.pt   (~1.2 GB)
#   facebook/sam3-tiny   → sam3-tiny.pt   (~400 MB)

fetch_sam3 () {
  local repo="$1"
  local local_name="$2"
  local target="$SAM3_DIR/$local_name"
  if [[ -s "$target" ]]; then
    echo "  skip ${local_name} (already present, $(du -h "$target" | cut -f1))"
    return
  fi
  echo "==> Downloading $repo → $local_name"
  huggingface-cli download "$repo" \
    --local-dir "$SAM3_DIR/.tmp-$local_name" \
    --local-dir-use-symlinks False
  # Find the .pt/.safetensors in the downloaded snapshot and move it.
  local found
  found="$(find "$SAM3_DIR/.tmp-$local_name" -maxdepth 3 -type f \
    \( -name '*.pt' -o -name '*.pth' -o -name '*.safetensors' \) | head -1)"
  if [[ -z "$found" ]]; then
    echo "ERROR: no weight file found inside $repo snapshot" >&2
    exit 3
  fi
  mv "$found" "$target"
  rm -rf "$SAM3_DIR/.tmp-$local_name"
}

fetch_sam3 facebook/sam3       sam3-large.pt
fetch_sam3 facebook/sam3-base  sam3-base.pt
fetch_sam3 facebook/sam3-tiny  sam3-tiny.pt

# ─── MatAnyone weights ───────────────────────────────────────────────────────
MATANYONE_REPO="${MATANYONE_REPO:-PeiqingYang/MatAnyone}"
MATANYONE_FILE="${MATANYONE_FILE:-matanyone.pth}"
if [[ ! -s "$MATANYONE_DIR/$MATANYONE_FILE" ]]; then
  echo "==> Downloading $MATANYONE_REPO → $MATANYONE_FILE"
  huggingface-cli download "$MATANYONE_REPO" \
    --local-dir "$MATANYONE_DIR" \
    --local-dir-use-symlinks False
  # Some repos pack weights under subdirs; normalize.
  if [[ ! -s "$MATANYONE_DIR/$MATANYONE_FILE" ]]; then
    found="$(find "$MATANYONE_DIR" -maxdepth 3 -type f \
      \( -name 'matanyone*.pth' -o -name '*.safetensors' \) | head -1)"
    if [[ -n "$found" ]]; then
      mv "$found" "$MATANYONE_DIR/$MATANYONE_FILE"
    else
      echo "WARN: MatAnyone weight file not found; check repo layout" >&2
    fi
  fi
else
  echo "  skip $MATANYONE_FILE (already present, $(du -h "$MATANYONE_DIR/$MATANYONE_FILE" | cut -f1))"
fi

# ─── Verify ──────────────────────────────────────────────────────────────────
echo ""
echo "==> Final inventory"
ls -lh "$SAM3_DIR" || true
echo ""
ls -lh "$MATANYONE_DIR" || true
echo ""
echo "==> Disk usage on volume"
du -sh "$VOLUME_ROOT"/* 2>/dev/null || true

# Write a small manifest the handler can read on boot for sanity checking.
cat >"$VOLUME_ROOT/manifest.json" <<EOF
{
  "initialized_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "sam3_dir": "$SAM3_DIR",
  "matanyone_dir": "$MATANYONE_DIR",
  "hf_cache": "$HF_CACHE",
  "weights": {
    "sam3-tiny":  "$(stat -c%s "$SAM3_DIR/sam3-tiny.pt"  2>/dev/null || echo 0)",
    "sam3-base":  "$(stat -c%s "$SAM3_DIR/sam3-base.pt"  2>/dev/null || echo 0)",
    "sam3-large": "$(stat -c%s "$SAM3_DIR/sam3-large.pt" 2>/dev/null || echo 0)",
    "matanyone":  "$(stat -c%s "$MATANYONE_DIR/$MATANYONE_FILE" 2>/dev/null || echo 0)"
  }
}
EOF
echo ""
echo "==> Done. Manifest written to $VOLUME_ROOT/manifest.json"
cat "$VOLUME_ROOT/manifest.json"
