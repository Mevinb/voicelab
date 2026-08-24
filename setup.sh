#!/usr/bin/env bash
# Voice Lab — one-command bootstrap.
#
# Creates both virtualenvs, installs pinned dependencies (CUDA 12.4 PyTorch),
# clones the Seed-VC engine at a known-good commit, and applies the
# setuptools<81 fix required by Chatterbox's watermarking library.
#
# Models (~2 GB) download automatically from Hugging Face on first use and are
# cached in ./checkpoints — they are NOT stored in this repository.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SEED_VC_PIN="51383efd921027683c89e5348211d93ff12ac2a8"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

fail() { echo "ERROR: $*" >&2; exit 1; }

echo "==> Preflight checks"
command -v python3 >/dev/null || fail "python3 not found"
command -v git      >/dev/null || fail "git not found"
command -v ffmpeg   >/dev/null || fail "ffmpeg not found (needed for .opus/.mp3 input)"

PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
case "$PYVER" in
    3.10|3.11|3.12) : ;;
    *) fail "Python 3.10-3.12 required (found $PYVER)" ;;
esac
echo "    python $PYVER, git, ffmpeg: OK"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "    GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    echo "    WARNING: no NVIDIA GPU detected — everything will run on CPU (very slow)."
fi

FREE_GB=$(df -BG . | awk 'NR==2 {print int($4)}')
if (( FREE_GB < 20 )); then
    fail "only ${FREE_GB}GB free; ~20GB needed (venvs + model cache)"
fi
echo "    disk: ${FREE_GB}GB free"

echo "==> Cloning Seed-VC @ ${SEED_VC_PIN:0:8}"
if [[ -d seed-vc/.git ]]; then
    echo "    already present, skipping clone"
else
    rm -rf seed-vc
    git clone https://github.com/Plachtaa/seed-vc.git
    git -C seed-vc checkout "$SEED_VC_PIN"
fi

echo "==> Creating seed-vc venv (torch 2.4.0 + cu124)"
[[ -d seed-vc/.venv ]] || python3 -m venv seed-vc/.venv
seed-vc/.venv/bin/pip install --quiet --upgrade pip
seed-vc/.venv/bin/pip install --quiet \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url "$TORCH_INDEX"
seed-vc/.venv/bin/pip install --quiet \
    --extra-index-url "$TORCH_INDEX" -r requirements-seedvc.txt

echo "==> Creating chatterbox venv (chatterbox-tts + torch 2.6.0)"
[[ -d chatterbox/.venv ]] || python3 -m venv chatterbox/.venv
chatterbox/.venv/bin/pip install --quiet --upgrade pip
chatterbox/.venv/bin/pip install --quiet -r requirements-chatterbox.txt
# perth (audio watermarking used by Chatterbox) still imports pkg_resources,
# which was removed in setuptools >= 81 — pin it back down.
chatterbox/.venv/bin/pip install --quiet "setuptools<81"

echo "==> Verifying installs"
seed-vc/.venv/bin/python -c 'import torch; assert torch.__version__.startswith("2.4.0"); print("    seed-vc:", torch.__version__)'
chatterbox/.venv/bin/python - <<'EOF'
import warnings; warnings.filterwarnings("ignore")
from perth import PerthImplicitWatermarker
assert PerthImplicitWatermarker is not None
print("    chatterbox + perth watermarker: OK")
EOF

mkdir -p samples outputs checkpoints

cat <<'EOF'

Voice Lab is ready!

    ./voice-ui.sh        # start the web UI (prints the local URL)
    ./convert_voice.sh   # or use the CLI tools — see README.md

First conversion/generation triggers a one-time ~2GB model download.
EOF
