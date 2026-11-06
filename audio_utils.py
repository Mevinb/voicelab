"""Audio helpers shared by the UI (chatterbox venv) and the Seed-VC worker.

Only depends on numpy, scipy, soundfile and pyloudnorm (present in both venvs)
plus the ffmpeg binary, so any format ffmpeg understands (opus voice notes,
m4a, webm mic recordings, mp3, ...) is accepted everywhere.
"""
import subprocess
from pathlib import Path

import numpy as np
import pyloudnorm
import soundfile as sf
from scipy.ndimage import uniform_filter
from scipy.signal import istft, stft

REFERENCE_LUFS = -20.0
OUTPUT_LUFS = -18.0
PEAK_CEILING = 10 ** (-1.0 / 20)  # -1 dBFS
RUMBLE_FILTER = "highpass=f=80"  # applied by ffmpeg before denoising


def decode(path: str | Path, sr: int, denoise: bool = False) -> np.ndarray:
    """Decode any audio file to mono float32 at `sr` using ffmpeg (optionally denoised)."""
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), *(["-af", RUMBLE_FILTER] if denoise else []),
         "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise ValueError(f"could not decode {Path(path).name}: {proc.stderr.decode().strip()[-300:]}")
    wav = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if wav.size == 0:
        raise ValueError(f"{Path(path).name} contains no audio")
    return spectral_gate(wav, sr) if denoise else wav


def spectral_gate(wav: np.ndarray, sr: int, reduction_db: float = 18.0, margin_db: float = 6.0) -> np.ndarray:
    """Stationary-noise reduction for phone / voice-note recordings.

    The noise spectrum is estimated from the clip's quietest 10% of frames
    (pauses between words); time-frequency bins that are not clearly above it
    are attenuated by `reduction_db`. The gain mask is smoothed to avoid the
    "musical noise" / underwater artefacts of hard gating.
    """
    n_fft = 1 << int(np.ceil(np.log2(sr * 0.032)))  # ~32 ms window
    if len(wav) < 4 * n_fft:
        return wav
    _, _, spec = stft(wav, sr, nperseg=n_fft, noverlap=n_fft * 3 // 4)
    mag = np.abs(spec)
    energy = mag.sum(axis=0)
    noise = mag[:, energy <= np.percentile(energy, 10)].mean(axis=1, keepdims=True)
    floor = 10 ** (-reduction_db / 20)
    gain = np.where(mag > noise * 10 ** (margin_db / 20), 1.0, floor)
    gain = np.clip(uniform_filter(gain, size=(5, 7), mode="nearest"), floor, 1.0)
    _, out = istft(spec * gain, sr, nperseg=n_fft, noverlap=n_fft * 3 // 4)
    return out[: len(wav)].astype(np.float32)


def _frame_db(wav: np.ndarray, sr: int) -> tuple[int, np.ndarray]:
    """Per-20 ms frame level in dB relative to the clip's loudest frame."""
    frame = max(1, int(sr * 0.02))
    n = len(wav) // frame
    rms = np.sqrt(np.mean(wav[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    return frame, 20 * np.log10(rms / (rms.max(initial=0) + 1e-12))


def trim_silence(wav: np.ndarray, sr: int, threshold_db: float = -45.0, pad_s: float = 0.15) -> np.ndarray:
    """Cut leading/trailing silence (relative to the clip's peak), keeping a little padding."""
    frame, db = _frame_db(wav, sr)
    loud = np.flatnonzero(db > threshold_db)
    if loud.size == 0:
        return wav
    pad = int(sr * pad_s)
    return wav[max(0, loud[0] * frame - pad): min(len(wav), (loud[-1] + 1) * frame + pad)]


def remove_pauses(wav: np.ndarray, sr: int, max_pause_s: float = 0.4, threshold_db: float = -40.0) -> np.ndarray:
    """Shorten internal pauses longer than `max_pause_s` so more real speech fits in a reference."""
    frame, db = _frame_db(wav, sr)
    n, max_frames = len(db), max(1, int(max_pause_s * sr / frame))
    keep = np.ones(n, dtype=bool)
    quiet = np.concatenate([[False], db <= threshold_db, [False]])
    edges = np.flatnonzero(np.diff(quiet.astype(np.int8)))
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start > max_frames:
            keep[start + max_frames // 2: end - (max_frames - max_frames // 2)] = False
    return np.concatenate([wav[: n * frame][np.repeat(keep, frame)], wav[n * frame:]])


def _fade(wav: np.ndarray, sr: int, fade_s: float = 0.01) -> np.ndarray:
    k = min(len(wav) // 2, int(sr * fade_s))
    if k:
        ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
        wav = wav.copy()
        wav[:k] *= ramp
        wav[-k:] *= ramp[::-1]
    return wav


def normalize(wav: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """Loudness-normalize to `target_lufs`, then hard-limit peaks to -1 dBFS."""
    try:
        loudness = pyloudnorm.Meter(sr).integrated_loudness(wav)
        if np.isfinite(loudness):
            wav = wav * 10 ** ((target_lufs - loudness) / 20)
    except ValueError:  # clip shorter than the 400 ms gating block
        pass
    peak = np.abs(wav).max()
    if peak > PEAK_CEILING:
        wav = wav * (PEAK_CEILING / peak)
    return wav.astype(np.float32)


def prepare_reference(src: str | Path, dst: str | Path, sr: int, max_seconds: float | None = None,
                      denoise: bool = False) -> float:
    """Clean a reference voice clip (mono, trimmed, loudness-normalized) and save as WAV.

    Returns the cleaned duration in seconds.
    """
    wav = trim_silence(decode(src, sr, denoise=denoise), sr)
    if denoise:
        wav = remove_pauses(wav, sr)
    if max_seconds:
        wav = wav[: int(sr * max_seconds)]
    sf.write(str(dst), normalize(wav, sr, REFERENCE_LUFS), sr)
    return len(wav) / sr


def build_reference(clips: list[str | Path], dst: str | Path, sr: int, max_seconds: float = 25.0,
                    denoise: bool = True, gap_s: float = 0.25, min_clip_s: float = 0.5) -> dict:
    """Merge several short clips of one speaker into a single clean reference WAV.

    Each clip is (optionally) denoised, trimmed, has long pauses shortened and is
    loudness-matched, then the clips are joined in order with short gaps and
    capped at `max_seconds` (Seed-VC only reads the first 25 s of a reference).
    Returns {"seconds", "clips": [(name, seconds_used)], "skipped": [(name, reason)], "truncated"}.
    """
    gap = np.zeros(int(sr * gap_s), dtype=np.float32)
    parts, used, skipped, total, limit, truncated = [], [], [], 0, int(sr * max_seconds), False
    for clip in clips:
        name = Path(clip).name
        try:
            wav = remove_pauses(trim_silence(decode(clip, sr, denoise=denoise), sr), sr)
        except ValueError as exc:
            skipped.append((name, str(exc)))
            continue
        if len(wav) < sr * min_clip_s:
            skipped.append((name, f"under {min_clip_s:g} s of speech"))
            continue
        start = total + (len(gap) if parts else 0)
        room = limit - start
        if room < sr * min_clip_s:
            skipped.append((name, f"{max_seconds:g} s limit reached"))
            truncated = True
            continue
        truncated |= len(wav) > room
        wav = _fade(normalize(wav, sr, REFERENCE_LUFS)[:room], sr)
        parts += [gap, wav] if parts else [wav]
        total = start + len(wav)
        used.append((name, round(len(wav) / sr, 1)))
    if not parts:
        raise ValueError("none of the clips contained usable speech")
    wav = np.concatenate(parts)
    sf.write(str(dst), normalize(wav, sr, REFERENCE_LUFS), sr)
    return {"seconds": round(len(wav) / sr, 1), "clips": used, "skipped": skipped, "truncated": truncated}


def save_output(wav: np.ndarray, sr: int, dst: str | Path) -> float:
    """Normalize generated audio to a consistent listening level and save. Returns seconds."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    sf.write(str(dst), normalize(wav, sr, OUTPUT_LUFS), sr)
    return len(wav) / sr
