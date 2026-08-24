# 🎙️ Voice Lab

Local, private voice cloning workbench with a web UI. Two engines behind one
interface:

| Tab | Engine | What it does |
|---|---|---|
| 🔁 **Voice Conversion** | [Seed-VC](https://github.com/Plachtaa/seed-vc) | Take someone's recording and re-render it **in another person's voice** — words and delivery preserved, zero-shot from a 1–30s reference clip |
| 💬 **Text to Speech** | [Chatterbox](https://github.com/resemble-ai/chatterbox) | Type text and have it spoken in a cloned voice (MIT-licensed, ElevenLabs-competitive quality) |

Everything runs **on your machine** — no audio ever leaves it.

> Verified on an RTX 4050 Laptop (6 GB VRAM): both models resident at once use
> ~4.3 GB, voice conversion runs faster than real time (RTF ≈ 0.45).

## How it works

```
                ┌────────────────────────────┐
   browser ────▶│  Gradio UI (app.py)        │
                │  port auto-picked ≥ 7861   │
                └──────┬─────────────┬───────┘
                       │             │
        in-process     │             │  HTTP (127.0.0.1:8677)
                       ▼             ▼
              ┌────────────┐  ┌──────────────────┐
              │ Chatterbox │  │ vc_worker.py     │
              │ TTS        │  │ warm Seed-VC     │
              └────────────┘  └──────────────────┘
```

The two engines pin conflicting `transformers` versions (4.x vs 5.x), so they
run as separate processes. Seed-VC stays loaded as a tiny local HTTP worker so
repeat conversions take seconds instead of re-loading ~2 GB of weights each time.

## Quick start

Requirements: Linux/macOS, Python 3.10–3.12, git, ffmpeg, NVIDIA GPU
(CPU works but is slow), ~20 GB free disk.

```bash
./setup.sh        # creates venvs, installs pinned deps, clones Seed-VC @ pinned commit
./voice-ui.sh     # starts everything; opens http://127.0.0.1:<port>
```

First conversion/generation triggers a one-time ~2 GB model download from
Hugging Face into `./checkpoints/`.

## CLI equivalents

```bash
# Voice-to-voice conversion (source audio -> target speaker's voice)
./convert_voice.sh <source.(wav|mp3|opus|m4a)> <target_reference.wav> outputs/

# Text-to-speech in a cloned voice
chatterbox/.venv/bin/python speak.py "Hello world" --reference ref.wav -o out.wav
```

WhatsApp voice notes (`.opus`) work directly via ffmpeg.

## Getting good clones

- The **reference clip decides everything**: clean, single speaker, no music,
  5–15 seconds. Garbage in, garbage out.
- Conversion preserves the *source* speaker's pacing and emotion; use the
  speed slider if you need to stretch/compress timing.
- Chatterbox: lower `Pace` (~0.3) for deliberate delivery; `Exaggeration`
  adds emotion.

## Troubleshooting

| Symptom | Fix |
|---|---|
| UI doesn't open on 7860 | Intentional — ports ≥7861 are used to avoid clashes (e.g. SD WebUI). Check the URL printed by `voice-ui.sh`. |
| "Seed-VC worker is not running" | Start via `./voice-ui.sh`, not `python app.py` alone. |
| First run seems stuck | It's downloading ~2 GB of model weights; see `outputs/ui_logs/vc_worker.log`. |
| `perth ... NoneType not callable` | Run `chatterbox/.venv/bin/pip install "setuptools<81"` (`setup.sh` does this for you). |
| CUDA out of memory | Close other GPU apps; each engine alone needs ≤3 GB. |

## Privacy & ethics ⚠️

- `samples/`, `outputs/` and `checkpoints/` are git-ignored by design — your
  recordings never belong in a repository.
- Only clone voices of people who **consented**. Impersonating real people
  without permission is illegal in many jurisdictions and always a bad look.
- Generated audio carries Resemble AI's PerTh watermark and remains detectable
  by forensic tools. Label synthetic media when you share it.

## License & credits

- This wrapper/UI code: MIT (see `LICENSE`)
- [Seed-VC](https://github.com/Plachtaa/seed-vc) (GPL-3.0) is cloned at install
  time to a pinned commit and not redistributed here.
- [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) (MIT) via
  `chatterbox-tts`.
# voicelab
