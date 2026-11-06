#!/usr/bin/env bash
# Voice Lab — one-command bootstrap.
#
# Creates both virtualenvs (Python 3.11, fetched by uv if your system Python is
# different), installs pinned dependencies, and clones the Seed-VC engine at a
# known-good commit.
#
# Both venvs pin the *same* torch 2.6.0/CUDA 12.4 build, so uv hardlinks the
# ~5 GB of CUDA wheels once instead of storing two copies.
#
# Models download automatically from Hugging Face on first use and are cached
# in ./checkpoints — they are NOT stored in this repository.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SEED_VC_PIN="51383efd921027683c89e5348211d93ff12ac2a8"
# Chatterbox master @ 2026-07-21: adds Multilingual V3 + Nano over PyPI 0.1.7.
CHATTERBOX_PIN="5de7a54aa4e5e2baadb0182dde554908b48b85c2"
PY_VERSION="3.11"

fail() { echo "ERROR: $*" >&2; exit 1; }

echo "==> Preflight checks"
command -v git    >/dev/null || fail "git not found"
command -v ffmpeg >/dev/null || fail "ffmpeg not found (needed to decode .opus/.mp3/.m4a input)"

if command -v uv >/dev/null; then
    # Hardlinks only work within one filesystem: if uv's cache lives elsewhere,
    # keep a project-local cache so both venvs still share one copy of torch.
    UV_CACHE="$(uv cache dir 2>/dev/null || true)"
    if [[ -z "${UV_CACHE_DIR:-}" && -d "$UV_CACHE" && "$(stat -c %d "$UV_CACHE")" != "$(stat -c %d "$ROOT")" ]]; then
        export UV_CACHE_DIR="$ROOT/.uv-cache"
        echo "    uv cache is on another filesystem; using $UV_CACHE_DIR for hardlinking"
    fi
    mkvenv() { [[ -x "$1/bin/python" ]] || uv venv --quiet --python "$PY_VERSION" "$1"; }
    pipi()   { local venv=$1; shift; uv pip install --quiet --python "$venv/bin/python" "$@"; }
    echo "    uv $(uv --version | cut -d' ' -f2) (Python $PY_VERSION will be fetched if needed)"
else
    PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    case "$PYVER" in
        3.10|3.11|3.12) : ;;
        *) fail "Python 3.10-3.12 required (found $PYVER). Install uv instead and re-run:
       curl -LsSf https://astral.sh/uv/install.sh | sh" ;;
    esac
    mkvenv() { [[ -x "$1/bin/python" ]] || python3 -m venv "$1"; "$1/bin/pip" install --quiet --upgrade pip; }
    pipi()   { local venv=$1; shift; "$venv/bin/pip" install --quiet "$@"; }
    echo "    python $PYVER (tip: install uv for a much faster, deduplicated install)"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "    GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
    echo "    WARNING: no NVIDIA GPU detected — everything will run on CPU (slow; use the Nano TTS model)."
fi

FREE_GB=$(df -BG . | awk 'NR==2 {print int($4)}')
MIN_GB="${VOICE_LAB_MIN_GB:-12}"
if (( FREE_GB < MIN_GB )); then
    fail "only ${FREE_GB}GB free; ${MIN_GB}GB needed for the venvs (models need 2-4 GB more each)"
fi
echo "    disk: ${FREE_GB}GB free"

echo "==> Cloning Seed-VC @ ${SEED_VC_PIN:0:8}"
if [[ -d seed-vc/.git ]]; then
    echo "    already present, skipping clone"
else
    rm -rf seed-vc
    git clone --quiet https://github.com/Plachtaa/seed-vc.git
fi
git -C seed-vc checkout --quiet "$SEED_VC_PIN"

echo "==> seed-vc venv (torch 2.6.0 + cu124)"
mkvenv seed-vc/.venv
pipi seed-vc/.venv -r requirements-seedvc.txt

echo "==> chatterbox venv (chatterbox @ ${CHATTERBOX_PIN:0:8} + torch 2.6.0)"
mkvenv chatterbox/.venv
pipi chatterbox/.venv -r requirements-chatterbox.txt
# Dependencies are pinned above; --no-deps keeps the PyPI watermarker build.
pipi chatterbox/.venv --no-deps \
    "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git@${CHATTERBOX_PIN}"

echo "==> Verifying installs"
seed-vc/.venv/bin/python -c 'import torch; assert torch.__version__.startswith("2.6.0"); print("    seed-vc:", torch.__version__, "cuda" if torch.cuda.is_available() else "cpu")'
chatterbox/.venv/bin/python - <<'EOF'
import warnings; warnings.filterwarnings("ignore")
# perth (watermarker) imports pkg_resources, removed in setuptools >= 81.
from perth import PerthImplicitWatermarker
from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
assert PerthImplicitWatermarker is not None
print("    chatterbox (original/turbo/nano/multilingual) + perth watermarker: OK")
EOF

mkdir -p samples outputs checkpoints voices

cat <<'EOF'

Voice Lab is ready!

    ./voice-ui.sh        # start the web UI (prints the local URL)
    ./convert_voice.sh   # or use the CLI tools — see README.md

Each model downloads once (2-4 GB) the first time you use it.
EOF
