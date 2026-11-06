#!/usr/bin/env bash
# Voice Lab launcher — starts the Seed-VC worker + the web UI.
#
#   ./voice-ui.sh          then open the printed http://127.0.0.1:<port>
#   Ctrl-C                 stops both
#
# Env overrides: VC_PRELOAD (engines loaded at start, default "v1"; "" = lazy),
#                VC_WORKER_PORT (default 8677)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/outputs/ui_logs"
mkdir -p "$LOG_DIR"
export VC_WORKER_PORT="${VC_WORKER_PORT:-8677}"

# First run on a fresh clone? Bootstrap everything automatically.
if [[ ! -x "$ROOT/seed-vc/.venv/bin/python" || ! -x "$ROOT/chatterbox/.venv/bin/python" ]]; then
    echo "[voice-ui] first run detected — running setup.sh (installs deps, ~10 min)..."
    "$ROOT/setup.sh"
fi

# Stop leftover instances for a clean restart
pkill -f "$ROOT/vc_worker.py" 2>/dev/null || true
pkill -f "$ROOT/app.py" 2>/dev/null || true
sleep 1

echo "[voice-ui] starting Seed-VC worker (first run downloads ~2 GB of models)..."
cd "$ROOT"
export PYTHONUNBUFFERED=1
# Progress bars in the worker refresh every 2 s so the mirrored log stays readable.
TQDM_MININTERVAL=2 "$ROOT/seed-vc/.venv/bin/python" "$ROOT/vc_worker.py" serve --preload "${VC_PRELOAD-v1}" \
    >"$LOG_DIR/vc_worker.log" 2>&1 &
WORKER_PID=$!
# Mirror the worker log live into this terminal (also kept in the log file);
# progress-bar redraws (\r) become separate lines.
tail -n +1 -F --pid="$WORKER_PID" "$LOG_DIR/vc_worker.log" 2>/dev/null | "$ROOT/chatterbox/.venv/bin/python" -c '
import os, re, sys
buf = b""
while chunk := os.read(0, 4096):
    *lines, buf = re.split(rb"[\r\n]+", buf + chunk)
    for line in filter(bytes.strip, lines):
        sys.stdout.buffer.write(b"[vc] " + line + b"\n")
    sys.stdout.flush()
' &
TAIL_PID=$!
trap 'kill "$WORKER_PID" "$TAIL_PID" 2>/dev/null || true' EXIT

health() {
    "$ROOT/chatterbox/.venv/bin/python" -c \
        "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$VC_WORKER_PORT/health', timeout=2)" \
        >/dev/null 2>&1
}

# Wait until the worker answers /health (up to 10 min for the first model download)
for _ in $(seq 1 300); do
    health && break
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then
        echo "[voice-ui] worker died — last log lines:" >&2
        tail -20 "$LOG_DIR/vc_worker.log" >&2
        exit 1
    fi
    sleep 2
done
health || { echo "[voice-ui] worker did not become ready; see $LOG_DIR/vc_worker.log" >&2; exit 1; }
echo "[voice-ui] worker ready."

# Pick a free UI port starting at 7861 (7860 is often taken, e.g. by other web UIs)
UI_PORT=""
for p in $(seq 7861 7899); do
    if ! (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null; then
        UI_PORT=$p
        break
    fi
done
if [[ -z "$UI_PORT" ]]; then
    echo "[voice-ui] no free port in 7861-7899" >&2
    exit 1
fi

echo "[voice-ui] opening UI at http://127.0.0.1:$UI_PORT  (Ctrl-C to stop)"
GRADIO_SERVER_PORT="$UI_PORT" "$ROOT/chatterbox/.venv/bin/python" "$ROOT/app.py"
