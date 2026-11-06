"""Voice Lab — unified web UI for Seed-VC (voice conversion) and Chatterbox (TTS).

Runs inside the chatterbox venv:
    chatterbox/.venv/bin/python app.py

Voice conversion jobs are dispatched to the always-warm Seed-VC worker
(vc_worker.py, started by ./voice-ui.sh); TTS runs in-process here.
"""
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path

import gradio as gr
import requests
import soundfile as sf
import torch

import audio_utils
from tts_engine import DEVICE, MODELS, TURBO_TAGS, TTSEngine, languages

VOICE_ROOT = Path(__file__).parent.resolve()
OUTPUT_DIR = VOICE_ROOT / "outputs"
VOICES_DIR = VOICE_ROOT / "voices"
HISTORY_FILE = OUTPUT_DIR / "history.jsonl"
WORKER_URL = f"http://127.0.0.1:{os.environ.get('VC_WORKER_PORT', '8677')}"
NO_VOICE = "— none / use upload —"
# Approximate VRAM per model (weights + working memory), used to decide when to
# evict models from the other process on small GPUs.
VRAM_GB = {"v1": 2.0, "v1f0": 3.0, "v2": 3.0, "original": 4.0, "multilingual": 4.0, "turbo": 3.5, "nano": 2.5}
HEADROOM_GB = 1.0
# Library voices are stored at 44.1 kHz so the 44 kHz pitch-aware engine gets the
# full bandwidth; every engine resamples references to its own rate anyway.
VOICE_SR = 44100
REFERENCE_MAX_S = 25

VC_ENGINES = {
    "V1 · best voice similarity, fastest (recommended)": "v1",
    "V1-F0 · pitch-aware 44 kHz (use if V1 sounds robotic / buzzy)": "v1f0",
    "V2 · also converts accent & emotion (slower, lower similarity)": "v2",
}
TTS_CHOICES = {spec.label: key for key, spec in MODELS.items()}
LANGUAGE_CHOICES = {f"{name} ({code})": code for code, name in sorted(languages().items(), key=lambda kv: kv[1])}

tts = TTSEngine()


# ---------------------------------------------------------------- helpers
def new_output_path(kind: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{kind}_{uuid.uuid4().hex[:6]}.wav"


def log_history(**entry) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    with HISTORY_FILE.open("a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **entry}) + "\n")


def voice_names() -> list[str]:
    VOICES_DIR.mkdir(exist_ok=True)
    return sorted(p.stem for p in VOICES_DIR.glob("*.wav"))


def voice_path(name: str | None) -> str | None:
    return str(VOICES_DIR / f"{name}.wav") if name and name != NO_VOICE and name in voice_names() else None


def voice_dropdown_update(value=NO_VOICE):
    return gr.update(choices=[NO_VOICE, *voice_names()], value=value)


def gpu_free_gb() -> float:
    """Free VRAM across all processes (nvidia-smi avoids creating a CUDA context here)."""
    if DEVICE != "cuda":
        return float("inf")
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return int(out.split()[0]) / 1024
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return float("inf")


def worker_loaded() -> list[str]:
    try:
        return requests.get(f"{WORKER_URL}/health", timeout=2).json()["loaded"]
    except requests.RequestException:
        return []


def make_room(task: str, resident: bool) -> None:
    """Small GPUs can't hold every model at once: free the *other* process's models if needed."""
    need = HEADROOM_GB if resident else VRAM_GB[task]
    free = gpu_free_gb()
    if free >= need:
        return
    if task in MODELS and worker_loaded():
        print(f"[ui] {free:.1f} GB VRAM free, {task} needs ~{need:.1f} GB: unloading Seed-VC models", flush=True)
        requests.post(f"{WORKER_URL}/unload", json={}, timeout=60)
    elif task in VC_ENGINES.values() and tts.key:
        print(f"[ui] {free:.1f} GB VRAM free, {task} needs ~{need:.1f} GB: unloading TTS model", flush=True)
        tts.unload()


def worker_status() -> str:
    try:
        info = requests.get(f"{WORKER_URL}/health", timeout=2).json()
        loaded = ", ".join(info["loaded"]) or "none yet"
        return f"Seed-VC worker: ✅ {info['device']} (loaded: {loaded})"
    except requests.RequestException:
        return "Seed-VC worker: ❌ not running — start everything with `./voice-ui.sh`"


# ---------------------------------------------------------------- voice conversion
def run_conversion(source, reference, engine_label, steps, speed, cfg_rate, similarity, convert_style,
                   pitch_shift, seed):
    if not source or not reference:
        raise gr.Error("Please provide BOTH the source audio (person B) and a reference voice (person A).")
    engine = VC_ENGINES[engine_label]
    print(f"[ui] conversion request: {engine}, {int(steps)} steps (progress in [vc] lines)", flush=True)
    try:
        make_room(engine, resident=engine in worker_loaded())
        resp = requests.post(f"{WORKER_URL}/convert", timeout=3600, json={
            "source": source, "reference": reference, "engine": engine,
            "diffusion_steps": int(steps), "length_adjust": float(speed), "cfg_rate": float(cfg_rate),
            "similarity_cfg_rate": float(similarity), "convert_style": bool(convert_style),
            "pitch_shift": int(pitch_shift), "seed": int(seed),
        })
    except requests.ConnectionError:
        raise gr.Error("Seed-VC worker is not running. Start everything with ./voice-ui.sh")
    data = resp.json()
    if resp.status_code != 200:
        print(f"[ui] conversion failed: {data.get('error')}", flush=True)
        raise gr.Error(f"Conversion failed: {data.get('error', resp.status_code)}")
    print(f"[ui] conversion saved: {data['output_path']}", flush=True)

    log_history(kind="conversion", file=data["output_path"], engine=engine, seed=data["seed"],
                duration=data["duration"], details=f"{Path(source).name} → {Path(reference).name}")
    note = (f"**Done** in {data['seconds']:.1f}s · {data['duration']:.1f}s of audio · engine {engine} · "
            f"seed `{data['seed']}` · reference used: {data['reference_seconds']:.1f}s")
    return data["output_path"], note


# ---------------------------------------------------------------- text to speech
def run_tts(text, model_label, language_label, reference, exaggeration, cfg_weight, temperature, seed,
            progress=gr.Progress()):
    if not text.strip():
        raise gr.Error("Please enter some text to speak.")
    key = TTS_CHOICES[model_label]
    if key != tts.key:
        progress(0, desc=f"Loading {model_label.split(' —')[0]} (first use downloads 2-4 GB)…")
        tts.unload()  # free the previous model before measuring headroom
    make_room(key, resident=key == tts.key)
    t0 = time.time()
    try:
        audio, sr, info = tts.generate(
            text, model_key=key, reference=reference, exaggeration=exaggeration, cfg_weight=cfg_weight,
            temperature=temperature, language=LANGUAGE_CHOICES.get(language_label, "en"), seed=seed,
            progress=lambda i, n, chunk: progress(i / n, desc=f"Sentence group {i + 1}/{n}: {chunk[:50]}…"),
        )
    except ValueError as exc:
        print(f"[ui] TTS failed: {exc}", flush=True)
        raise gr.Error(str(exc))
    elapsed = time.time() - t0
    out = new_output_path(f"tts-{key}")
    duration = tts.save(audio, sr, out)
    print(f"[ui] TTS saved: {out}", flush=True)
    log_history(kind="tts", file=str(out), engine=key, seed=info["seed"], duration=round(duration, 2),
                details=text[:120])
    note = (f"**Done** in {elapsed:.1f}s · {duration:.1f}s of audio · {info['chunks']} chunk(s) · "
            f"seed `{info['seed']}`")
    return str(out), note


def tts_model_changed(model_label):
    spec = MODELS[TTS_CHOICES[model_label]]
    key = TTS_CHOICES[model_label]
    return (gr.update(visible=key == "multilingual"), gr.update(visible=spec.expressive),
            gr.update(visible=spec.expressive), gr.update(visible=key in ("turbo", "nano")))


# ---------------------------------------------------------------- voice library
def clean_voice_name(name) -> str:
    name = re.sub(r"[^\w \-]", "", (name or "")).strip()[:60]
    if not name:
        raise gr.Error("Give the voice a name (letters, numbers, spaces, - and _).")
    return name


def saved_voice_updates(name, msg):
    return msg, voice_table(), voice_dropdown_update(), voice_dropdown_update(), voice_dropdown_update(name)


def length_warning(seconds: float) -> str:
    if seconds < 5:
        return " ⚠️ Under 5 s: similarity will be weak, and Turbo/Nano need ≥5 s. Add more clips if you can."
    if seconds < 10:
        return " ⚠️ Under 10 s: similarity improves a lot with 10–25 s of speech."
    return ""


def save_voice(name, audio, denoise):
    name = clean_voice_name(name)
    if not audio:
        raise gr.Error("Upload or record a reference clip first.")
    VOICES_DIR.mkdir(exist_ok=True)
    try:
        seconds = audio_utils.prepare_reference(audio, VOICES_DIR / f"{name}.wav", VOICE_SR,
                                                max_seconds=30, denoise=denoise)
    except ValueError as exc:
        raise gr.Error(str(exc))
    return saved_voice_updates(name, f"Saved **{name}** ({seconds:.1f}s after cleanup).{length_warning(seconds)}")


def build_report(report: dict) -> str:
    lines = [f"**{report['seconds']:.1f} s** of speech from {len(report['clips'])} clip(s)."
             + length_warning(report["seconds"])]
    lines += [f"- ✅ `{name}` — {seconds:.1f} s used" for name, seconds in report["clips"]]
    lines += [f"- ⏭️ `{name}` — skipped: {why}" for name, why in report["skipped"]]
    if report["truncated"]:
        lines.append(f"\nCapped at {REFERENCE_MAX_S} s (the models ignore anything longer) — "
                     "upload your cleanest clips first.")
    return "\n".join(lines)


def build_voice(files, denoise, dst: Path) -> str:
    if not files:
        raise gr.Error("Upload one or more short clips of the same person first.")
    try:
        report = audio_utils.build_reference(files, dst, VOICE_SR, max_seconds=REFERENCE_MAX_S, denoise=denoise)
    except ValueError as exc:
        raise gr.Error(str(exc))
    print(f"[ui] built reference {dst.name}: {report['seconds']}s from {len(report['clips'])} clip(s)", flush=True)
    return build_report(report)


def preview_built_voice(files, denoise):
    OUTPUT_DIR.mkdir(exist_ok=True)
    for old in OUTPUT_DIR.glob("reference-preview_*.wav"):
        old.unlink(missing_ok=True)
    dst = OUTPUT_DIR / f"reference-preview_{uuid.uuid4().hex[:6]}.wav"
    return str(dst), build_voice(files, denoise, dst)


def save_built_voice(name, files, denoise):
    name = clean_voice_name(name)
    VOICES_DIR.mkdir(exist_ok=True)
    report = build_voice(files, denoise, VOICES_DIR / f"{name}.wav")
    return saved_voice_updates(name, f"Saved **{name}**.\n\n{report}")


def delete_voice(name):
    path = voice_path(name)
    if not path:
        raise gr.Error("Pick a saved voice to delete.")
    Path(path).unlink()
    return (f"Deleted **{name}**.", voice_table(), voice_dropdown_update(), voice_dropdown_update(),
            voice_dropdown_update(), None)


def voice_table():
    return [[name, round(sf.info(str(VOICES_DIR / f"{name}.wav")).duration, 1)] for name in voice_names()]


# ---------------------------------------------------------------- history
def history_rows(limit: int = 50):
    if not HISTORY_FILE.exists():
        return []
    entries = [json.loads(line) for line in HISTORY_FILE.read_text().splitlines() if line.strip()]
    entries = [e for e in entries if Path(e["file"]).exists()][-limit:][::-1]
    return [[e["time"], e["kind"], e["engine"], e["seed"], e["duration"], e["details"], e["file"]] for e in entries]


def history_select(rows, evt: gr.SelectData):
    row = rows.iloc[evt.index[0]].tolist() if hasattr(rows, "iloc") else rows[evt.index[0]]
    return row[-1], row[3]


# ---------------------------------------------------------------- UI
def reference_picker(label):
    dropdown = gr.Dropdown([NO_VOICE, *voice_names()], value=NO_VOICE, label="Saved voice")
    audio = gr.Audio(label=label, type="filepath", sources=["upload", "microphone"])
    dropdown.change(voice_path, dropdown, audio)
    return dropdown, audio


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Voice Lab") as demo:
        gr.Markdown("# 🎙️ Voice Lab\nLocal voice cloning — nothing leaves your machine.")
        status = gr.Markdown(f"TTS device: **{DEVICE}** · {worker_status()}")

        with gr.Tab("🔁 Voice Conversion"):
            gr.Markdown(
                "**Make person B's recording sound like person A.**\n\n"
                "- *Source*: the voice note / recording whose words and delivery you want kept.\n"
                "- *Reference*: 10–25 s of clean speech of the target voice — pick a saved voice or upload one. "
                "Only have short voice notes? Merge them in **🗂️ Voice Library → Build a voice from several "
                "short clips**.\n"
                "- Output sounds robotic, buzzy or off-pitch? Try the **V1-F0** engine."
            )
            with gr.Row():
                source_in = gr.Audio(label="Source audio (person B)", type="filepath",
                                     sources=["upload", "microphone"])
                with gr.Column():
                    vc_voice, vc_ref = reference_picker("Reference voice (person A)")
            engine_in = gr.Radio(list(VC_ENGINES), value=next(iter(VC_ENGINES)), label="Engine")
            with gr.Row():
                steps_in = gr.Slider(4, 100, value=30, step=1,
                                     label="Diffusion steps (higher = cleaner, slower; 50 for best quality)")
                speed_in = gr.Slider(0.7, 1.5, value=1.0, step=0.05,
                                     label="Length adjust (<1 faster, >1 slower, 1.0 = original pacing)")
            with gr.Accordion("Advanced", open=False):
                with gr.Row():
                    cfg_in = gr.Slider(0.0, 1.0, value=0.7, step=0.05,
                                       label="CFG rate (V1) / intelligibility (V2)")
                    sim_in = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Voice similarity (V2)")
                with gr.Row():
                    style_in = gr.Checkbox(label="Also convert accent & emotion (V2, slower)")
                    pitch_in = gr.Slider(-12, 12, value=0, step=1,
                                         label="Pitch shift in semitones (V1-F0, on top of automatic pitch matching)",
                                         visible=VC_ENGINES[engine_in.value] == "v1f0")
                    vc_seed_in = gr.Number(-1, precision=0, label="Seed (-1 = random)")
            convert_btn = gr.Button("Convert voice", variant="primary")
            conv_out = gr.Audio(label="Converted output", type="filepath")
            conv_info = gr.Markdown()
            engine_in.change(lambda label: (VC_ENGINES[label] == "v2", gr.update(visible=VC_ENGINES[label] == "v1f0")),
                             engine_in, [style_in, pitch_in])
            convert_btn.click(run_conversion,
                              [source_in, vc_ref, engine_in, steps_in, speed_in, cfg_in, sim_in, style_in, pitch_in,
                               vc_seed_in],
                              [conv_out, conv_info])

        with gr.Tab("💬 Text to Speech"):
            gr.Markdown("**Type text and speak it in a cloned voice.** Any length works — long text is split "
                        "into sentences. Leave the reference empty to use the model's built-in voice.")
            with gr.Row():
                model_in = gr.Dropdown(list(TTS_CHOICES), value=next(iter(TTS_CHOICES)), label="Model")
                lang_in = gr.Dropdown(list(LANGUAGE_CHOICES), value="English (en)", label="Language",
                                      visible=False)
            text_in = gr.Textbox(lines=6, label="Text to speak")
            tags_md = gr.Markdown("Expression tags you can type inline: " + " ".join(f"`{t}`" for t in TURBO_TAGS),
                                  visible=False)
            tts_voice, tts_ref = reference_picker("Reference voice (optional)")
            with gr.Row():
                exag_in = gr.Slider(0.25, 2.0, value=0.5, step=0.05, label="Exaggeration (emotion intensity)")
                pace_in = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                    label="Pace / CFG (lower = slower & more deliberate)")
            with gr.Row():
                temp_in = gr.Slider(0.05, 1.5, value=0.8, step=0.05, label="Temperature (variation)")
                tts_seed_in = gr.Number(-1, precision=0, label="Seed (-1 = random)")
            tts_btn = gr.Button("Generate speech", variant="primary")
            tts_out = gr.Audio(label="Generated audio", type="filepath")
            tts_info = gr.Markdown()
            model_in.change(tts_model_changed, model_in, [lang_in, exag_in, pace_in, tags_md])
            tts_btn.click(run_tts,
                          [text_in, model_in, lang_in, tts_ref, exag_in, pace_in, temp_in, tts_seed_in],
                          [tts_out, tts_info])

        with gr.Tab("🗂️ Voice Library"):
            gr.Markdown("Save clean reference clips once and reuse them in both tabs. Clips are trimmed, "
                        "loudness-normalized and stored privately in `voices/` (git-ignored).")
            with gr.Row():
                with gr.Column():
                    new_name = gr.Textbox(label="Voice name", placeholder="e.g. Alice calm")
                    new_audio = gr.Audio(label="Reference clip (10–25 s, single speaker, no music)",
                                         type="filepath", sources=["upload", "microphone"])
                    new_denoise = gr.Checkbox(True, label="Clean up (denoise + shorten pauses; for phone recordings)")
                    save_btn = gr.Button("Save voice", variant="primary")
                with gr.Column():
                    voices_df = gr.Dataframe(voice_table(), headers=["Voice", "Seconds"], interactive=False)
                    lib_voice = gr.Dropdown([NO_VOICE, *voice_names()], value=NO_VOICE, label="Selected voice")
                    lib_preview = gr.Audio(label="Preview", type="filepath", interactive=False)
                    delete_btn = gr.Button("Delete selected voice", variant="stop")
            lib_msg = gr.Markdown()

            gr.Markdown("### 🧩 Build a voice from several short clips\n"
                        "No single clean 10–25 s recording? Upload several short clips of the **same person** "
                        "(WhatsApp notes, phone recordings…). Each one is denoised, trimmed, has long pauses "
                        "shortened and is level-matched, then they are joined in upload order (max 25 s). "
                        "Only include clips where that person alone is speaking.")
            with gr.Row():
                with gr.Column():
                    build_files = gr.File(label="Short clips (any format)", file_count="multiple",
                                          file_types=["audio", ".opus", ".ogg", ".oga", ".m4a", ".aac", ".amr",
                                                      ".webm", ".mp4"], type="filepath")
                    build_denoise = gr.Checkbox(True, label="Denoise (recommended for phone / WhatsApp audio)")
                    build_name = gr.Textbox(label="Voice name", placeholder="e.g. Alice merged")
                    with gr.Row():
                        build_preview_btn = gr.Button("Preview merged reference")
                        build_save_btn = gr.Button("Save as voice", variant="primary")
                with gr.Column():
                    build_preview = gr.Audio(label="Merged reference (listen before saving)", type="filepath",
                                             interactive=False)
                    build_msg = gr.Markdown()

            lib_voice.change(voice_path, lib_voice, lib_preview)
            save_btn.click(save_voice, [new_name, new_audio, new_denoise],
                           [lib_msg, voices_df, vc_voice, tts_voice, lib_voice])
            build_preview_btn.click(preview_built_voice, [build_files, build_denoise], [build_preview, build_msg])
            build_save_btn.click(save_built_voice, [build_name, build_files, build_denoise],
                                 [build_msg, voices_df, vc_voice, tts_voice, lib_voice])
            delete_btn.click(delete_voice, lib_voice,
                             [lib_msg, voices_df, vc_voice, tts_voice, lib_voice, lib_preview])

        with gr.Tab("🕘 History") as history_tab:
            gr.Markdown("Recent outputs (newest first). Click a row to play it; the seed lets you reproduce it.")
            history_df = gr.Dataframe(history_rows(), interactive=False,
                                      headers=["Time", "Type", "Model", "Seed", "Seconds", "Details", "File"])
            with gr.Row():
                history_audio = gr.Audio(label="Selected output", type="filepath", interactive=False)
                history_seed = gr.Number(label="Seed of selected output", interactive=False, precision=0)
            refresh_btn = gr.Button("Refresh")
            history_tab.select(history_rows, None, history_df)
            refresh_btn.click(history_rows, None, history_df)
            history_df.select(history_select, history_df, [history_audio, history_seed])

        demo.load(lambda: f"TTS device: **{DEVICE}** · {worker_status()}", None, status)
    return demo


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    VOICES_DIR.mkdir(exist_ok=True)
    # Port comes from voice-ui.sh (first free port ≥7861); fall back to Gradio's
    # auto-scan so `chatterbox/.venv/bin/python app.py` also works standalone.
    build_ui().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "0")) or None,
        allowed_paths=[str(OUTPUT_DIR), str(VOICES_DIR)],
    )
