#!/usr/bin/env python
"""Text-to-speech in a cloned voice using Chatterbox.

Usage:
    ./chatterbox/.venv/bin/python speak.py "Hello world" [--reference ref.wav] [-o out.wav]

If --reference is omitted, uses Chatterbox's default voice.
"""
import argparse

import torch
import torchaudio
from chatterbox.tts import ChatterboxTTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate speech in a cloned voice")
    parser.add_argument("text", help="Text to speak")
    parser.add_argument(
        "--reference",
        help="WAV clip of the target voice (5-10s clean speech recommended)",
    )
    parser.add_argument(
        "-o", "--output", default="outputs/tts_output.wav", help="Output WAV path"
    )
    parser.add_argument(
        "--exaggeration", type=float, default=0.5, help="Emotion exaggeration 0-1"
    )
    parser.add_argument(
        "--cfg-weight", type=float, default=0.5,
        help="Pace control: lower = slower/more deliberate",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTTS.from_pretrained(device=device)

    kwargs = {
        "exaggeration": args.exaggeration,
        "cfg_weight": args.cfg_weight,
    }
    if args.reference:
        kwargs["audio_prompt_path"] = args.reference

    wav = model.generate(args.text, **kwargs)
    torchaudio.save(args.output, wav, model.sr)
    print(f"Wrote {args.output} ({device})")


if __name__ == "__main__":
    main()
