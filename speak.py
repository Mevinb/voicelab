#!/usr/bin/env python
"""Text-to-speech in a cloned voice using Chatterbox.

Usage:
    ./chatterbox/.venv/bin/python speak.py "Hello world" [--reference ref.wav] [-o out.wav]
    ./chatterbox/.venv/bin/python speak.py --file chapter.txt --model turbo --reference ref.wav
    ./chatterbox/.venv/bin/python speak.py "Bonjour" --model multilingual --language fr

If --reference is omitted, uses the model's built-in default voice. Text of any
length works (it is split into sentence groups).
"""
import argparse
from pathlib import Path

from tts_engine import MODELS, TTSEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate speech in a cloned voice")
    parser.add_argument("text", nargs="?", help="Text to speak")
    parser.add_argument("--file", help="Read the text from this file instead")
    parser.add_argument("--reference", help="Clip of the target voice (5-15 s of clean speech recommended)")
    parser.add_argument("-o", "--output", default="outputs/tts_output.wav", help="Output WAV path")
    parser.add_argument("--model", choices=list(MODELS), default="original")
    parser.add_argument("--language", default="en", help="Language code for --model multilingual")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Emotion exaggeration 0.25-2")
    parser.add_argument("--cfg-weight", type=float, default=0.5,
                        help="Pace control: lower = slower/more deliberate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args()

    text = Path(args.file).read_text() if args.file else args.text
    if not text:
        parser.error("provide text or --file")

    engine = TTSEngine()
    audio, sr, info = engine.generate(
        text, model_key=args.model, reference=args.reference, exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight, temperature=args.temperature, language=args.language, seed=args.seed,
        progress=lambda i, n, _chunk: print(f"  chunk {i + 1}/{n}", flush=True),
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    seconds = engine.save(audio, sr, args.output)
    print(f"Wrote {args.output} ({seconds:.1f}s, model {info['model']}, seed {info['seed']})")


if __name__ == "__main__":
    main()
