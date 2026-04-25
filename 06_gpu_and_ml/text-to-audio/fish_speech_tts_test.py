from __future__ import annotations

import argparse
import base64
from pathlib import Path

import requests


DEFAULT_BASE_URL = "https://flowerwine8023--fish-speech-s2-pro-demo-serve.modal.run"
DEFAULT_REFERENCE_DIR = Path("tmp/references/audio")

REFERENCE_PRESETS = {
    "mizuki-1": {
        "audio": DEFAULT_REFERENCE_DIR / "1.mp3",
        "text": "昨晚的梦境…味道很美妙，感谢你与我分享。",
    },
    "mizuki-2": {
        "audio": DEFAULT_REFERENCE_DIR / "2.mp3",
        "text": "如果心里郁闷的话，就先放下手头的事，出去呼吸一下新鲜空气吧",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Call the deployed Fish Speech TTS service with an optional reference voice."
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Deployed Fish Speech service base URL.",
    )
    parser.add_argument(
        "--prompt",
        default="阴翳的天气，容易勾起人心深处的负面情绪…",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--output",
        default="tmp/fish-speech-test.wav",
        help="Where to save the generated WAV file.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(REFERENCE_PRESETS),
        default="mizuki-2",
        help="Built-in reference preset.",
    )
    parser.add_argument(
        "--reference-audio",
        default="",
        help="Override reference audio path.",
    )
    parser.add_argument(
        "--reference-text",
        default="",
        help="Override reference transcript.",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Disable reference voice cloning and use random voice.",
    )
    return parser


def build_reference(args: argparse.Namespace) -> list[dict]:
    if args.no_reference:
        return []

    preset = REFERENCE_PRESETS[args.preset]
    audio_path = Path(args.reference_audio) if args.reference_audio else preset["audio"]
    reference_text = args.reference_text or preset["text"]

    if not audio_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {audio_path}")
    if not reference_text:
        raise ValueError("Reference text must not be empty when reference audio is used.")

    return [
        {
            "audio": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
            "text": reference_text,
        }
    ]


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "text": args.prompt,
        "format": "wav",
        "latency": "normal",
        "streaming": False,
        "normalize": True,
        "references": build_reference(args),
    }

    response = requests.post(
        args.base_url.rstrip("/") + "/v1/tts",
        json=payload,
        timeout=600,
    )
    response.raise_for_status()
    output_path.write_bytes(response.content)

    print(f"Saved audio to {output_path}")


if __name__ == "__main__":
    main()
