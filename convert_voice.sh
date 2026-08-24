#!/usr/bin/env bash
# Voice-to-voice conversion: make SOURCE audio sound like the TARGET speaker.
#
# Usage:
#   ./convert_voice.sh <source_audio> <target_reference_audio> [output_dir]
#
# Example:
#   ./convert_voice.sh samples/person_b_note.wav samples/person_a_ref.wav outputs/
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <source_audio> <target_reference_audio> [output_dir]" >&2
    exit 1
fi

SOURCE="$1"
TARGET="$2"
OUT="${3:-outputs}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT"

exec "$ROOT/seed-vc/.venv/bin/python" "$ROOT/seed-vc/inference.py" \
    --source "$SOURCE" \
    --target "$TARGET" \
    --output "$OUT" \
    --diffusion-steps 25 \
    --fp16 True
