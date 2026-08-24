"""Voice Lab — unified web UI for Seed-VC (voice conversion) and Chatterbox (TTS).

Runs inside the chatterbox venv:
    chatterbox/.venv/bin/python app.py

Voice conversion jobs are dispatched to the always-warm Seed-VC worker
(vc_worker.py, started by ./voice-ui.sh); TTS runs in-process here.
"""
import base64
import time
from pathlib import Path

import gradio as gr
import requests
import torch
import torchaudio

VOICE_ROOT = Path(__file__).parent.resolve()
OUTPUT_DIR = VOICE_ROOT / "outputs"
WORKER_URL = "http://127.0.0.1:8677"

_tts_model = None


def get_tts_model():
    global _tts_model
    if _tts_model is None:
        from chatterbox.tts import ChatterboxTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ui] loading Chatterbox on {device}...", flush=True)
        _tts_model = ChatterboxTTS.from_pretrained(device=device)
    return _tts_model


def normalize_to_wav(audio_path: str) -> bytes:
    """Re-encode any supported input (mp3/m4a/ogg/webm/mic) as WAV bytes."""
    wav, sr = torchaudio.load(audio_path)
    import tempfile

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    torchaudio.save(tmp.name, wav, sr)
    return Path(tmp.name).read_bytes()


def run_conversion(source, target, steps, length_adjust):
    if not source or not target:
        raise gr.Error("Please provide BOTH the source audio (person B) and the reference voice (person A).")

    src_b64 = base64.b64encode(normalize_to_wav(source)).decode()
    tgt_b64 = base64.b64encode(normalize_to_wav(target)).decode()

    try:
        resp = requests.post(
            f"{WORKER_URL}/convert",
            json={
                "source_b64": src_b64,
                "target_b64": tgt_b64,
                "diffusion_steps": int(steps),
                "length_adjust": float(length_adjust),
            },
            timeout=1800,
        )
    except requests.ConnectionError:
        raise gr.Error("Seed-VC worker is not running. Start everything with ./voice-ui.sh")

    if resp.status_code != 200:
        raise gr.Error(f"Conversion failed: {resp.json().get('error', resp.status_code)}")

    data = resp.json()
    out_path = OUTPUT_DIR / f"converted_{int(time.time())}.wav"
    out_path.write_bytes(base64.b64decode(data["audio_b64"]))
    return str(out_path)


def run_tts(text, reference, exaggeration, cfg_weight):
    if not text.strip():
        raise gr.Error("Please enter some text to speak.")

    model = get_tts_model()
    kwargs = {"exaggeration": float(exaggeration), "cfg_weight": float(cfg_weight)}
    if reference:
        kwargs["audio_prompt_path"] = str(reference)

    wav = model.generate(text, **kwargs)
    out_path = OUTPUT_DIR / f"tts_{int(time.time())}.wav"
    torchaudio.save(str(out_path), wav, model.sr)
    return str(out_path)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Voice Lab") as demo:
        gr.Markdown("# 🎙️ Voice Lab\nLocal voice cloning — nothing leaves your machine.")

        with gr.Tab("🔁 Voice Conversion"):
            gr.Markdown(
                "**Make person B's recording sound like person A.**\n\n"
                "- *Source*: the voice note / recording whose words you want kept.\n"
                "- *Reference*: a short clean clip (5–15s) of the target voice."
            )
            with gr.Row():
                source_in = gr.Audio(label="Source audio (person B)", type="filepath",
                                     sources=["upload", "microphone"])
                ref_in = gr.Audio(label="Reference voice (person A)", type="filepath",
                                  sources=["upload", "microphone"])
            with gr.Row():
                steps_in = gr.Slider(10, 100, value=25, step=5,
                                     label="Diffusion steps (higher = slightly better, slower)")
                speed_in = gr.Slider(0.8, 1.3, value=1.0, step=0.05,
                                     label="Speed adjust (1.0 = keep original pacing)")
            convert_btn = gr.Button("Convert voice", variant="primary")
            conv_out = gr.Audio(label="Converted output", type="filepath")
            convert_btn.click(run_conversion, [source_in, ref_in, steps_in, speed_in], conv_out)

        with gr.Tab("💬 Text to Speech"):
            gr.Markdown("**Type text and speak it in a cloned voice.** Leave reference "
                        "empty to use the built-in default voice.")
            text_in = gr.Textbox(lines=4, label="Text to speak")
            ref2_in = gr.Audio(label="Reference voice (optional)", type="filepath",
                               sources=["upload", "microphone"])
            with gr.Row():
                exag_in = gr.Slider(0.25, 1.0, value=0.5, step=0.05,
                                    label="Exaggeration (emotion intensity)")
                cfg_in = gr.Slider(0.2, 0.9, value=0.5, step=0.05,
                                   label="Pace (lower = slower & more deliberate)")
            tts_btn = gr.Button("Generate speech", variant="primary")
            tts_out = gr.Audio(label="Generated audio", type="filepath")
            tts_btn.click(run_tts, [text_in, ref2_in, exag_in, cfg_in], tts_out)

    return demo


if __name__ == "__main__":
    import os

    OUTPUT_DIR.mkdir(exist_ok=True)
    # Port comes from voice-ui.sh (first free port ≥7861); fall back to Gradio's
    # auto-scan so `chatterbox/.venv/bin/python app.py` also works standalone.
    build_ui().launch(server_name="127.0.0.1", server_port=int(os.environ.get("GRADIO_SERVER_PORT", "0")) or None)
