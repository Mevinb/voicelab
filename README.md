# 🎙️ Voice Lab

Local, private voice cloning workbench with a web UI. Two engines behind one
interface:

| Tab | Engine | What it does |
|---|---|---|
| 🔁 **Voice Conversion** | [Seed-VC](https://github.com/Plachtaa/seed-vc) V1 / V1-F0 / V2 | Take someone's recording and re-render it **in another person's voice** — words and delivery preserved, zero-shot from a 1–25s reference clip. V1-F0 is pitch-aware at 44 kHz; V2 can also transfer accent & emotion. |
| 💬 **Text to Speech** | [Chatterbox](https://github.com/resemble-ai/chatterbox) Original / Turbo / Nano / Multilingual V3 | Type text of any length and have it spoken in a cloned voice; Turbo supports `[laugh]`-style tags, Multilingual covers 23 languages |
| 🗂️ **Voice Library** | — | Save cleaned reference voices once, pick them from a dropdown in both tabs. **Build a voice from several short clips** (e.g. WhatsApp notes): denoised, trimmed, level-matched and merged into one ≤25 s reference |
| 🕘 **History** | — | Replay recent outputs and recover their seeds |

Everything runs **on your machine** — no audio ever leaves it.

> Measured on an RTX 4050 Laptop (6 GB VRAM): warm V1 conversion RTF ≈ 0.33,
> V2 ≈ 0.85. On GPUs too small to hold everything, the UI automatically evicts
> the other process's models (they reload in ~10 s).

## How it works

```
                ┌────────────────────────────┐
   browser ────▶│  Gradio UI (app.py)        │
                │  port auto-picked ≥ 7861   │
                └──────┬─────────────┬───────┘
                       │             │
        in-process     │             │  HTTP (127.0.0.1:8677), file paths only
                       ▼             ▼
              ┌────────────────┐  ┌──────────────────┐
              │ tts_engine.py  │  │ vc_worker.py     │
              │ Chatterbox ×4  │  │ warm Seed-VC     │
              │                │  │ v1 / v1f0 / v2   │
              └────────────────┘  └──────────────────┘
```

The two engines pin conflicting `transformers` versions (4.x vs 5.x), so they
run as separate processes. Seed-VC stays loaded in a tiny local HTTP worker so
repeat conversions only pay for inference. Both venvs use the same torch build,
so uv hardlinks the CUDA libraries once (~7 GB for both venvs total).

All inputs go through ffmpeg (opus/ogg voice notes, m4a, mp3, webm mic
recordings). References are trimmed of silence and loudness-normalized;
outputs are normalized to a consistent level.

## Quick start

Requirements: Linux/macOS, [uv](https://docs.astral.sh/uv/) (recommended; it
fetches Python 3.11 for you) or Python 3.10–3.12, git, ffmpeg, NVIDIA GPU (CPU
works but is slow), ~10 GB free disk plus 2–4 GB per model you use.

```bash
./setup.sh        # creates venvs, installs pinned deps, clones Seed-VC @ pinned commit
./voice-ui.sh     # starts everything; prints http://127.0.0.1:<port>
```

Each model downloads once from Hugging Face into `./checkpoints/` the first
time you use it. `voice-ui.sh` streams the worker's log live into the terminal
(lines prefixed `[vc]`); TTS progress is printed as `[tts]` lines.

## CLI equivalents

```bash
# Voice-to-voice conversion (source audio -> target speaker's voice)
./convert_voice.sh <source.(wav|mp3|opus|m4a)> <target_reference> outputs/
./convert_voice.sh note.opus ref.wav outputs/ --engine v2 --style --seed 42
./convert_voice.sh note.opus ref.wav outputs/ --engine v1f0 --steps 50   # pitch-aware, 44 kHz

# Text-to-speech in a cloned voice (any length; --file for long text)
chatterbox/.venv/bin/python speak.py "Hello world" --reference ref.wav -o out.wav
chatterbox/.venv/bin/python speak.py "Hi [chuckle] there" --model turbo --reference ref.wav
chatterbox/.venv/bin/python speak.py "Bonjour à tous" --model multilingual --language fr
```

WhatsApp voice notes (`.opus`) work directly via ffmpeg.

## Getting good clones

- The **reference clip decides everything**: clean, single speaker, no music,
  10–25 seconds of actual speech. Garbage in, garbage out.
- Only have short or noisy voice notes? Use **Voice Library → Build a voice
  from several short clips**: upload 3–10 notes of the same person, preview
  the merged result, save it. Put the cleanest clips first (only the first
  25 s are used) and drop any clip where someone else talks.
- **V1** gives the highest speaker similarity (Resemblyzer score vs the
  target: V1 ≈ 0.93, V1-F0 ≈ 0.91, V2 ≈ 0.87–0.89 in our tests) and is the
  fastest. Prefer it over V2 unless you need accent/emotion transfer.
- **V1-F0** extracts pitch with RMVPE and matches the source's pitch to the
  reference. Try it when V1 sounds buzzy, robotic or off-pitch. It is slower,
  needs ~3 GB VRAM and downloads ~1.5 GB on first use. Only use the
  pitch-shift slider if the result is consistently too high or too low.
- Raise diffusion steps to ~50 for the cleanest result.
- Conversion preserves the *source* speaker's pacing and emotion; use the
  length slider to stretch/compress timing, or V2 with *accent & emotion* to
  adopt the reference speaker's style.
- Chatterbox: lower `Pace` (~0.3) for deliberate delivery; `Exaggeration`
  adds emotion.

## Troubleshooting

| Symptom | Fix |
|---|---|
| UI doesn't open on 7860 | Intentional — ports ≥7861 are used to avoid clashes (e.g. SD WebUI). Check the URL printed by `voice-ui.sh`. |
| "Seed-VC worker is not running" | Start via `./voice-ui.sh`, not `python app.py` alone. |
| First run seems stuck | It's downloading model weights; watch the `[vc]`/`[tts]` lines in the terminal (also `outputs/ui_logs/vc_worker.log`). |
| Turbo/Nano: "needs at least 5 s" | Those models require a ≥5 s reference after silence trimming. |
| Output doesn't sound like the person | Use V1 (not V2), build the reference from several clips (10–25 s total, denoise on), and try 50 diffusion steps. |
| Output sounds robotic / buzzy | Switch to the V1-F0 engine. |
| Merged reference sounds "underwater" | Untick *Denoise* in the clip builder — the clips were cleaner than expected. |
| `perth ... NoneType not callable` | Run `chatterbox/.venv/bin/pip install "setuptools<81"` (`setup.sh` does this for you). |
| CUDA out of memory | Close other GPU apps; the UI evicts idle models automatically, but each model alone needs 2–4 GB. |

## Privacy & ethics ⚠️

- `samples/`, `voices/`, `outputs/` and `checkpoints/` are git-ignored by design — your
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
