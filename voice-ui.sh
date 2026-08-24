#!/usr/bin/env bash
# Voice Lab launcher — starts the Seed-VC worker + the web UI.
#
#   ./voice-ui.sh          then open http://127.0.0.1:7860
#   Ctrl-C                 stops both
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/outputs/ui_logs"
mkdir -p "$LOG_DIR"

# First run on a fresh clone? Bootstrap everything automatically.
if [[ ! -x "$ROOT/seed-vc/.venv/bin/python" || ! -x "$ROOT/chatterbox/.venv/bin/python" ]]; then
    echo "[voice-ui] first run detected — running setup.sh (installs deps, ~10 min)..."
    "$ROOT/setup.sh"
fi

# Stop leftover instances for a clean restart
pkill -f "$ROOT/vc_worker.py" 2>/dev/null || true
pkill -f "$ROOT/app.py" 2>/dev/null || true
sleep 1

echo "[voice-ui] starting Seed-VC worker (loads model, ~15s)..."
cd "$ROOT"
nohup "$ROOT/seed-vc/.venv/bin/python" "$ROOT/vc_worker.py" \
    >"$LOG_DIR/vc_worker.log" 2>&1 &
WORKER_PID=$!

# Wait until the worker answers /health (first run also downloads ~1GB of models)
for _ in $(seq 1 90); do
    if "$ROOT/chatterbox/.venv/bin/python" - <<'EOF' >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:8677/health", timeout=2)
EOF
    then
        break
    fi
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then
        echo "[voice-ui] worker died — last log lines:" >&2
        tail -20 "$LOG_DIR/vc_worker.log" >&2
        exit 1
    fi
    sleep 2
done

if ! kill -0 "$WORKER_PID" 2>/dev/null; then
    echo "[voice-ui] worker failed to start" >&2
    exit 1
fi
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

trap 'kill "$WORKER_PID" 2>/dev/null || true' EXIT

echo "[voice-ui] opening UI at http://127.0.0.1:$UI_PORT  (Ctrl-C to stop)"
GRADIO_SERVER_PORT="$UI_PORT" "$ROOT/chatterbox/.venv/bin/python" "$ROOT/app.py"
