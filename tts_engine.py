"""Chatterbox text-to-speech engine shared by the web UI and speak.py.

Runs inside the chatterbox venv. Adds on top of the raw Chatterbox API:
  * four selectable models, one resident on the GPU at a time
  * long-text support (sentence chunking; each model call is capped at ~40 s)
  * cached voice conditioning per reference clip (no re-embedding every run)
  * reference cleanup, reproducible seeds, loudness-normalized output
"""
import gc
import hashlib
import os
import random
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

VOICE_ROOT = Path(__file__).parent.resolve()
# Keep every model inside the project (./checkpoints) like the Seed-VC worker does.
os.environ.setdefault("HF_HUB_CACHE", str(VOICE_ROOT / "checkpoints" / "hf_cache"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import audio_utils  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
SR = 24000  # Chatterbox output / reference sample rate


@dataclass(frozen=True)
class ModelSpec:
    label: str
    expressive: bool    # supports exaggeration + cfg_weight (pace)
    min_ref_s: float    # shortest usable reference clip
    max_chars: int      # characters per generation chunk (~20-25 s of speech)


MODELS = {
    "original": ModelSpec("Chatterbox — English, expressive", True, 1.0, 300),
    "turbo": ModelSpec("Chatterbox Turbo — English, fast, [laugh]-style tags", False, 5.0, 300),
    "nano": ModelSpec("Chatterbox Nano — English, tiny, runs on CPU", False, 5.0, 300),
    "multilingual": ModelSpec("Chatterbox Multilingual V3 — 23 languages", True, 1.0, 200),
}
TURBO_TAGS = ["[laugh]", "[chuckle]", "[sigh]", "[gasp]", "[cough]", "[clear throat]", "[sniff]", "[groan]",
              "[shush]", "[whispering]", "[happy]", "[sarcastic]", "[angry]", "[surprised]", "[fear]",
              "[crying]", "[dramatic]", "[narration]"]


def _download(repo: str, patterns: list[str]) -> str:
    """snapshot_download that reuses identical weight files already cached for another repo.

    Turbo and Nano ship byte-identical decoders (~1 GB); hardlinking the cached
    blob means the second model only downloads what is actually new.
    """
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    cache = Path(HF_HUB_CACHE)
    try:
        siblings = HfApi().model_info(repo, files_metadata=True).siblings or []
    except Exception:  # offline: fall back to whatever is cached
        siblings = []
    blobs = cache / f"models--{repo.replace('/', '--')}" / "blobs"
    for sib in siblings:
        if not sib.lfs or not any(fnmatch(sib.rfilename, p) for p in patterns):
            continue
        target = blobs / sib.lfs.sha256
        if not target.exists() and (twin := next(cache.glob(f"models--*/blobs/{sib.lfs.sha256}"), None)):
            blobs.mkdir(parents=True, exist_ok=True)
            os.link(twin, target)
    return snapshot_download(repo, allow_patterns=patterns)


def _load(key: str):
    if key == "original":
        from chatterbox.tts import ChatterboxTTS
        ckpt = _download("ResembleAI/chatterbox", ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors",
                                                   "tokenizer.json", "conds.pt"])
        return ChatterboxTTS.from_local(ckpt, DEVICE)
    if key in ("turbo", "nano"):
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        nano = key == "nano"
        # Only the files the model actually loads (upstream also pulls an unused 1 GB s3gen).
        ckpt = _download("ResembleAI/chatterbox-nano" if nano else "ResembleAI/chatterbox-turbo",
                         ["ve.safetensors", "s3gen_meanflow.safetensors", "t3_*_v1.safetensors",
                          "*.json", "*.txt", "conds.pt"])
        return ChatterboxTurboTTS.from_local(ckpt, DEVICE, nano=nano)
    if key == "multilingual":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        return ChatterboxMultilingualTTS.from_pretrained(device=DEVICE, t3_model="v3")
    raise ValueError(f"unknown model {key!r}")


def languages() -> dict[str, str]:
    from chatterbox.mtl_tts import SUPPORTED_LANGUAGES
    return dict(SUPPORTED_LANGUAGES)


_SENTENCE_END = re.compile(r"(?:(?<=[.!?…])|(?<=[.!?…][\"'”’)\]]))\s+|(?<=[。！？])|\n+")
_SOFT_BREAK = re.compile(r"(?<=[,;:，；、])\s*")


def split_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of whole sentences, each at most ~max_chars long."""
    pieces: list[str] = []
    for sentence in filter(None, (s.strip() for s in _SENTENCE_END.split(text))):
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        # Over-long sentence: break at commas, then at word boundaries ([tags] stay whole).
        for clause in filter(None, (c.strip() for c in _SOFT_BREAK.split(sentence))):
            words, line = re.findall(r"\[[^\]]*\]|\S{1,%d}" % max_chars, clause), ""
            for word in words:
                if line and len(line) + 1 + len(word) > max_chars:
                    pieces.append(line)
                    line = word
                else:
                    line = f"{line} {word}".strip()
            if line:
                pieces.append(line)

    chunks: list[str] = []
    for piece in pieces:
        if chunks and len(chunks[-1]) + 1 + len(piece) <= max_chars:
            chunks[-1] += " " + piece
        else:
            chunks.append(piece)
    return chunks


class TTSEngine:
    def __init__(self):
        self.key: str | None = None
        self.model = None
        self._builtin_conds = None
        self._conds: dict[tuple[str, str], object] = {}
        self._lock = threading.RLock()

    def unload(self) -> None:
        """Free the resident model's GPU memory (it reloads on next use)."""
        with self._lock:
            if self.key:
                print(f"[tts] unloading {self.key} to free GPU memory", flush=True)
            self.model = self._builtin_conds = self.key = None
            self._conds.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _ensure(self, key: str):
        if key != self.key:
            self.unload()
            print(f"[tts] loading {MODELS[key].label} on {DEVICE} (first use downloads it)...", flush=True)
            t0 = time.time()
            self.model, self.key = _load(key), key
            print(f"[tts] {key} ready in {time.time() - t0:.1f}s", flush=True)
            self._builtin_conds = self.model.conds
        return self.model

    def _condition(self, key: str, reference: str | None, exaggeration: float) -> None:
        """Point the model at the reference voice, reusing cached embeddings."""
        model = self.model
        if not reference:
            if self._builtin_conds is None:
                raise ValueError("this model has no built-in voice; please provide a reference clip")
            model.conds = self._builtin_conds
            return
        digest = hashlib.sha1(Path(reference).read_bytes()).hexdigest()
        if (key, digest) not in self._conds:
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                seconds = audio_utils.prepare_reference(reference, tmp.name, SR, max_seconds=30)
                if seconds < MODELS[key].min_ref_s:
                    raise ValueError(f"{MODELS[key].label.split(' —')[0]} needs at least "
                                     f"{MODELS[key].min_ref_s:.0f} s of speech in the reference "
                                     f"(got {seconds:.1f} s after trimming silence)")
                model.prepare_conditionals(tmp.name, exaggeration=exaggeration)
            self._conds[(key, digest)] = model.conds
        model.conds = self._conds[(key, digest)]

    def generate(self, text: str, model_key: str = "original", reference: str | None = None,
                 exaggeration: float = 0.5, cfg_weight: float = 0.5, temperature: float = 0.8,
                 language: str = "en", seed: int = -1, pause_s: float = 0.2, progress=None):
        """Synthesize `text`. Returns (wav float32 array, sample_rate, info dict)."""
        text = text.strip()
        if not text:
            raise ValueError("no text to speak")
        spec = MODELS[model_key]
        with self._lock:
            model = self._ensure(model_key)
            self._condition(model_key, reference, exaggeration)

            seed = int(seed) if int(seed) >= 0 else random.randrange(2**31)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            kwargs = {"temperature": float(temperature)}
            if spec.expressive:
                kwargs.update(exaggeration=float(exaggeration), cfg_weight=float(cfg_weight))
            else:  # Turbo/Nano ignore these and warn if they are non-zero
                kwargs.update(exaggeration=0.0, cfg_weight=0.0, min_p=0.0)
            if model_key == "multilingual":
                kwargs["language_id"] = language

            chunks = split_text(text, spec.max_chars)
            print(f"[tts] job: {model_key} seed={seed} {len(text)} chars -> {len(chunks)} chunk(s)"
                  f"{'' if reference else ' (built-in voice)'}", flush=True)
            silence = np.zeros(int(model.sr * pause_s), dtype=np.float32)
            parts, t_job = [], time.time()
            for i, chunk in enumerate(chunks):
                if progress:
                    progress(i, len(chunks), chunk)
                t0 = time.time()
                wav = model.generate(chunk, **kwargs).squeeze(0).cpu().numpy().astype(np.float32)
                print(f"[tts]   chunk {i + 1}/{len(chunks)}: {len(wav) / model.sr:.1f}s audio in "
                      f"{time.time() - t0:.1f}s  {chunk[:60]!r}", flush=True)
                parts += [wav, silence]
            audio = np.concatenate(parts[:-1])
            print(f"[tts] done: {len(audio) / model.sr:.1f}s audio in {time.time() - t_job:.1f}s", flush=True)
        return audio, model.sr, {"model": model_key, "seed": seed, "chunks": len(chunks)}

    def save(self, audio: np.ndarray, sr: int, path: str | Path) -> float:
        return audio_utils.save_output(audio, sr, path)
