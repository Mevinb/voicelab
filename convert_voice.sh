#!/usr/bin/env bash
# Voice-to-voice conversion: make SOURCE audio sound like the TARGET speaker.
#
# Usage:
#   ./convert_voice.sh <source_audio> <target_reference_audio> [output_dir] [vc_worker options]
#
# Examples:
#   ./convert_voice.sh samples/person_b_note.opus samples/person_a_ref.wav outputs/
#   ./convert_voice.sh note.opus ref.wav outputs/ --engine v2 --style
#
# Options are passed to `vc_worker.py convert` (see --help): --engine v1|v1f0|v2,
# --steps N, --speed X, --similarity X, --style, --pitch-shift N, --seed N.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <source_audio> <target_reference_audio> [output_dir] [options]" >&2
    exit 1
fi

SOURCE="$(realpath "$1")"
TARGET="$(realpath "$2")"
shift 2
OUT="outputs"
if [[ $# -gt 0 && "$1" != -* ]]; then OUT="$1"; shift; fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT"
OUT_FILE="$(realpath "$OUT")/vc_$(basename "${SOURCE%.*}")_to_$(basename "${TARGET%.*}")_$(date +%Y%m%d-%H%M%S).wav"

exec "$ROOT/seed-vc/.venv/bin/python" "$ROOT/vc_worker.py" convert "$SOURCE" "$TARGET" -o "$OUT_FILE" "$@"
