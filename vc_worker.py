"""Always-warm Seed-VC worker service.

Runs inside the seed-vc venv (which cannot share a process with Chatterbox
due to conflicting transformers pins). Exposes a tiny local HTTP API that
the Gradio UI calls:

    GET  /health   -> {"status": "ok"}
    POST /convert  -> {"audio_b64": ..., "filename": ...}
                      body: {"source_b64", "target_b64",
                             "diffusion_steps", "length_adjust"}

Start via ./voice-ui.sh (do not run by hand unless you know why).
"""
import base64
import json
import shutil
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

VOICE_ROOT = Path(__file__).parent.resolve()
SEED_VC_DIR = VOICE_ROOT / "seed-vc"
PORT = 8677
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # generous cap for long voice notes

# seed-vc/inference.py resolves its HF cache relative to CWD at import time,
# so we must run from the project root where checkpoints/ was created.
import os

os.chdir(VOICE_ROOT)
sys.path.insert(0, str(SEED_VC_DIR))

import inference as seed_vc  # noqa: E402

_job_lock = threading.Lock()


def load_seed_vc():
    args = SimpleNamespace(
        checkpoint=None,
        config=None,
        fp16=True,
        f0_condition=False,
    )
    t0 = time.time()
    seed_vc.load_models(args)
    print(f"[vc_worker] Seed-VC models loaded in {time.time() - t0:.1f}s", flush=True)


def convert(source_bytes: bytes, target_bytes: bytes, steps: int, length_adjust: float) -> bytes:
    with _job_lock:
        workdir = Path(tempfile.mkdtemp(prefix="vcjob_", dir=str(VOICE_ROOT / "outputs")))
        try:
            src_path = workdir / "source.wav"
            tgt_path = workdir / "target.wav"
            out_dir = workdir / "out"
            out_dir.mkdir()
            src_path.write_bytes(source_bytes)
            tgt_path.write_bytes(target_bytes)

            args = SimpleNamespace(
                source=str(src_path),
                target=str(tgt_path),
                output=str(out_dir),
                diffusion_steps=int(steps),
                length_adjust=float(length_adjust),
                inference_cfg_rate=0.7,
                f0_condition=False,
                auto_f0_adjust=True,
                semi_tone_shift=0,
                checkpoint=None,
                config=None,
                fp16=True,
            )
            seed_vc.main(args)

            wavs = sorted(out_dir.glob("*.wav"))
            if not wavs:
                raise RuntimeError("conversion produced no output file")
            return wavs[0].read_bytes()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if self.path != "/convert":
            self._send_json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > MAX_UPLOAD_BYTES:
                raise ValueError(f"invalid request size: {length}")
            req = json.loads(self.rfile.read(length))
            t0 = time.time()
            audio = convert(
                base64.b64decode(req["source_b64"]),
                base64.b64decode(req["target_b64"]),
                int(req.get("diffusion_steps", 25)),
                float(req.get("length_adjust", 1.0)),
            )
            print(f"[vc_worker] converted in {time.time() - t0:.1f}s", flush=True)
            self._send_json({
                "audio_b64": base64.b64encode(audio).decode(),
                "filename": f"converted_{int(time.time())}.wav",
            })
        except Exception as exc:  # surface the error to the UI
            print(f"[vc_worker] ERROR: {exc}", flush=True)
            self._send_json({"error": str(exc)}, 500)

    def log_message(self, fmt: str, *args) -> None:  # quiet default access log
        pass


if __name__ == "__main__":
    load_seed_vc()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[vc_worker] listening on 127.0.0.1:{PORT}", flush=True)
    server.serve_forever()
