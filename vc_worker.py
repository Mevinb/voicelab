"""Always-warm Seed-VC worker (voice conversion engine).

Runs inside the seed-vc venv, which cannot share a process with Chatterbox due
to conflicting transformers pins. Three Seed-VC engines are available:

    v1    timbre conversion (DiT + Whisper-small, 22 kHz, fastest)
    v1f0  pitch-aware timbre conversion (DiT + Whisper-base + RMVPE, 44 kHz);
          auto-matches the source pitch to the reference, avoids robotic/buzzy output
    v2    timbre + optional accent/emotion conversion (CFM + AR)

Models are loaded once and kept resident, so repeat conversions only pay for
inference. Usage:

    vc_worker.py serve [--preload v1]          # HTTP API for the web UI
    vc_worker.py convert SRC REF [-o OUT.wav]  # one-off CLI conversion

HTTP API (127.0.0.1 only; JSON bodies must use Content-Type: application/json):

    GET  /health   -> {"status": "ok", "loaded": [...], "device": ...}
    POST /convert  -> {"output_path", "seed", "seconds", "duration", ...}
                      body: {"source": path, "reference": path, "engine", ...}
    POST /unload   -> frees GPU memory held by the engines
"""
import argparse
import gc
import json
import os
import random
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

VOICE_ROOT = Path(__file__).parent.resolve()
SEED_VC_DIR = VOICE_ROOT / "seed-vc"
OUTPUT_DIR = VOICE_ROOT / "outputs"
PORT = int(os.environ.get("VC_WORKER_PORT", "8677"))
ENGINE_SR = {"v1": 22050, "v1f0": 44100, "v2": 22050}  # native sample rate of each model
ENGINE_VRAM_GB = {"v1": 2.0, "v1f0": 3.0, "v2": 3.0}   # approx. weights + working memory
REF_MAX_S = 25      # every engine only uses the first 25 s of the reference

# seed-vc resolves its ./checkpoints cache relative to CWD at import time.
os.chdir(VOICE_ROOT)
sys.path.insert(0, str(SEED_VC_DIR))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

import audio_utils  # noqa: E402
import inference as seed_v1  # noqa: E402

DEVICE = seed_v1.device
_upstream_load_v1 = seed_v1.load_models
_job_lock = threading.Lock()
_engines: dict = {}


class SeedVCv1:
    f0 = False

    def __init__(self):
        self.models = _upstream_load_v1(SimpleNamespace(checkpoint=None, config=None, fp16=True,
                                                        f0_condition=self.f0))

    def convert(self, src, ref, workdir, *, diffusion_steps, length_adjust, cfg_rate, pitch_shift=0, **_):
        out_dir = Path(workdir) / "v1_out"
        # Upstream main() calls load_models() on *every* conversion; hand it this
        # engine's resident models instead of re-reading ~2 GB of weights each time.
        # Swapped per call (under _job_lock) so v1 and v1f0 can both stay loaded.
        seed_v1.load_models = lambda _args: self.models
        try:
            seed_v1.main(SimpleNamespace(
                source=src, target=ref, output=str(out_dir),
                diffusion_steps=diffusion_steps, length_adjust=length_adjust, inference_cfg_rate=cfg_rate,
                f0_condition=self.f0, auto_f0_adjust=self.f0, semi_tone_shift=int(pitch_shift) if self.f0 else 0,
                checkpoint=None, config=None, fp16=True,
            ))
        finally:
            seed_v1.load_models = _upstream_load_v1
        wav, sr = sf.read(next(out_dir.glob("*.wav")), dtype="float32")
        return wav, sr

    def close(self):
        self.models = None


class SeedVCv1F0(SeedVCv1):
    f0 = True


class SeedVCv2:
    def __init__(self):
        import yaml
        from hydra.utils import instantiate
        from omegaconf import DictConfig

        cfg = DictConfig(yaml.safe_load((SEED_VC_DIR / "configs/v2/vc_wrapper.yaml").read_text()))
        self.model = instantiate(cfg)
        self.model.load_checkpoints()
        self.model.to(DEVICE).eval()
        self.model.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=torch.float16, device=DEVICE)

    def convert(self, src, ref, workdir, *, diffusion_steps, length_adjust, cfg_rate,
                similarity_cfg_rate, convert_style, temperature, top_p, **_):
        full = None
        # stream_output=False is broken upstream (returns nothing); drain the stream instead.
        for _, full in self.model.convert_voice_with_streaming(
            source_audio_path=src, target_audio_path=ref,
            diffusion_steps=diffusion_steps, length_adjust=length_adjust,
            intelligebility_cfg_rate=cfg_rate, similarity_cfg_rate=similarity_cfg_rate,
            top_p=top_p, temperature=temperature, repetition_penalty=1.0,
            convert_style=convert_style, anonymization_only=False,
            device=DEVICE, dtype=torch.float16, stream_output=True,
        ):
            pass
        if full is None:
            raise RuntimeError("conversion produced no audio")
        sr, wav = full
        return wav, sr

    def close(self):
        pass


ENGINES = {"v1": SeedVCv1, "v1f0": SeedVCv1F0, "v2": SeedVCv2}


def get_engine(name: str):
    if name not in ENGINES:
        raise ValueError(f"unknown engine {name!r} (choose from {', '.join(ENGINES)})")
    if name not in _engines:
        # Small GPUs can't hold every engine at once: drop the resident ones if needed.
        if _engines and torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_gb = torch.cuda.mem_get_info()[0] / 2**30
            if free_gb < ENGINE_VRAM_GB[name] + 0.5:
                print(f"[vc_worker] {free_gb:.1f} GB VRAM free: unloading {', '.join(_engines)} to load {name}",
                      flush=True)
                unload_all()
        t0 = time.time()
        _engines[name] = ENGINES[name]()
        print(f"[vc_worker] Seed-VC {name} loaded in {time.time() - t0:.1f}s", flush=True)
    return _engines[name]


def unload_all() -> None:
    for engine in _engines.values():
        engine.close()
    _engines.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def convert(source: str, reference: str, engine: str = "v1", diffusion_steps: int = 30,
            length_adjust: float = 1.0, cfg_rate: float = 0.7, similarity_cfg_rate: float = 0.7,
            convert_style: bool = False, temperature: float = 1.0, top_p: float = 0.9,
            pitch_shift: int = 0, seed: int = -1, output: str | None = None) -> dict:
    for label, path in (("source", source), ("reference", reference)):
        if not Path(path).is_file():
            raise ValueError(f"{label} file not found: {path}")
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r} (choose from {', '.join(ENGINES)})")
    sr_in = ENGINE_SR[engine]
    with _job_lock:
        seed = int(seed) if int(seed) >= 0 else random.randrange(2**31)
        print(f"[vc_worker] job: {engine} steps={diffusion_steps} speed={length_adjust} seed={seed} "
              f"pitch={pitch_shift} source={Path(source).name} reference={Path(reference).name}", flush=True)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        OUTPUT_DIR.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="vcjob_", dir=OUTPUT_DIR) as workdir:
            src = Path(workdir) / "source.wav"
            ref = Path(workdir) / "reference.wav"
            sf.write(src, audio_utils.normalize(audio_utils.decode(source, sr_in), sr_in,
                                                audio_utils.REFERENCE_LUFS), sr_in)
            ref_seconds = audio_utils.prepare_reference(reference, ref, sr_in, max_seconds=REF_MAX_S)
            if ref_seconds < 1.0:
                raise ValueError("reference voice has under 1 s of audible speech")

            vc = get_engine(engine)
            t0 = time.time()
            wav, sr = vc.convert(
                str(src), str(ref), workdir,
                diffusion_steps=int(diffusion_steps), length_adjust=float(length_adjust),
                cfg_rate=float(cfg_rate), similarity_cfg_rate=float(similarity_cfg_rate),
                convert_style=bool(convert_style), temperature=float(temperature), top_p=float(top_p),
                pitch_shift=int(pitch_shift),
            )
            elapsed = time.time() - t0

        out = Path(output) if output else OUTPUT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_vc-{engine}_{uuid.uuid4().hex[:6]}.wav"
        duration = audio_utils.save_output(wav, sr, out)
        print(f"[vc_worker] {engine}: {duration:.1f}s of audio in {elapsed:.1f}s (RTF {elapsed / duration:.2f})", flush=True)
        return {"output_path": str(out), "engine": engine, "seed": seed, "seconds": round(elapsed, 2),
                "duration": round(duration, 2), "reference_seconds": round(ref_seconds, 1)}


# Only these request fields are accepted over HTTP (no arbitrary output paths).
HTTP_FIELDS = {"source", "reference", "engine", "diffusion_steps", "length_adjust", "cfg_rate",
               "similarity_cfg_rate", "convert_style", "temperature", "top_p", "pitch_shift", "seed"}


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
            self._send_json({"status": "ok", "loaded": sorted(_engines), "device": str(DEVICE)})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        # Requiring a JSON content type forces a CORS preflight, so web pages
        # in your browser cannot drive this worker cross-origin.
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self._send_json({"error": "Content-Type must be application/json"}, 415)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}") if 0 <= length <= 1 << 20 else None
            if not isinstance(req, dict):
                raise ValueError("invalid request body")
            if self.path == "/convert":
                self._send_json(convert(**{k: v for k, v in req.items() if k in HTTP_FIELDS}))
            elif self.path == "/unload":
                with _job_lock:
                    unload_all()
                self._send_json({"status": "ok"})
            else:
                self._send_json({"error": "not found"}, 404)
        except Exception as exc:  # surface the error to the UI
            print(f"[vc_worker] ERROR: {exc!r}", flush=True)
            self._send_json({"error": str(exc)}, 500)

    def log_message(self, fmt: str, *args) -> None:  # quiet default access log
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="run the HTTP worker for the web UI")
    serve.add_argument("--preload", default="v1", help="comma-separated engines to load at start ('' for none)")
    cli = sub.add_parser("convert", help="convert one file and exit")
    cli.add_argument("source", help="recording whose words/delivery to keep")
    cli.add_argument("reference", help="clip of the target voice (5-25 s)")
    cli.add_argument("-o", "--output", help="output .wav (default: outputs/<timestamp>.wav)")
    cli.add_argument("--engine", choices=sorted(ENGINES), default="v1")
    cli.add_argument("--steps", type=int, default=30, dest="diffusion_steps")
    cli.add_argument("--speed", type=float, default=1.0, help="length adjust: <1 faster, >1 slower")
    cli.add_argument("--cfg-rate", type=float, default=0.7)
    cli.add_argument("--similarity", type=float, default=0.7, help="v2: voice-similarity CFG rate")
    cli.add_argument("--style", action="store_true", help="v2: also convert accent & emotion")
    cli.add_argument("--pitch-shift", type=int, default=0, help="v1f0: semitones on top of auto pitch matching")
    cli.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args()

    if args.cmd == "convert":
        result = convert(args.source, args.reference, engine=args.engine, diffusion_steps=args.diffusion_steps,
                         length_adjust=args.speed, cfg_rate=args.cfg_rate, similarity_cfg_rate=args.similarity,
                         convert_style=args.style, pitch_shift=args.pitch_shift, seed=args.seed,
                         output=args.output)
        print(json.dumps(result, indent=2))
        return

    for name in filter(None, args.preload.split(",")):
        get_engine(name.strip())
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[vc_worker] listening on 127.0.0.1:{PORT} (device: {DEVICE})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
