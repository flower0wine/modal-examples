"""
Deploy Fish Speech on Modal for demo testing.

Usage:

1. Deploy the API server:
   modal deploy 06_gpu_and_ml/text-to-audio/fish_speech_tts.py

2. Run a smoke test and save audio locally:
   modal run 06_gpu_and_ml/text-to-audio/fish_speech_tts.py --prompt "Hello from Fish Speech on Modal."

Notes:
- By default this uses Fish Audio's current flagship open-weights model: `fishaudio/s2-pro`.
- The first run downloads model weights into a Modal Volume and can take a while.
- Fish Audio's official docs recommend at least 24 GB of VRAM for S2 inference.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import base64
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import modal

MINUTES = 60
API_PORT = 8080
GPU_TYPE = "L40S"
MODEL_REPO = os.environ.get("FISH_SPEECH_MODEL_REPO", "fishaudio/s2-pro")
MODEL_SLUG = MODEL_REPO.split("/")[-1]
MODEL_DIR = f"/models/{MODEL_SLUG}"
APP_NAME = f"fish-speech-{MODEL_SLUG}-demo"

app = modal.App(APP_NAME)

model_volume = modal.Volume.from_name("fish-speech-models", create_if_missing=True)

image = (
    modal.Image.from_registry("fishaudio/fish-speech:latest", add_python="3.12")
    .entrypoint([])
    .pip_install("huggingface_hub[hf_transfer]")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
)


@app.function(
    image=image,
    volumes={"/models": model_volume},
    timeout=60 * MINUTES,
)
def download_model(force: bool = False) -> str:
    from huggingface_hub import snapshot_download

    destination = Path(MODEL_DIR)
    ready_marker = destination / "config.json"

    if force and destination.exists():
        import shutil

        shutil.rmtree(destination)

    if ready_marker.exists():
        return f"Model already cached at {destination}"

    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=str(destination),
    )
    model_volume.commit()
    return f"Downloaded {MODEL_REPO} to {destination}"


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/models": model_volume},
    scaledown_window=15 * MINUTES,
    timeout=45 * MINUTES,
)
@modal.web_server(port=API_PORT, startup_timeout=30 * MINUTES)
def serve():
    model_path = Path(MODEL_DIR)
    if not (model_path / "config.json").exists():
        raise RuntimeError(
            f"Model weights are missing under {model_path}. "
            "Run download_model first."
        )

    cmd = [
        "/bin/bash",
        "-lc",
        (
            "cd /app && "
            "uv run tools/api_server.py "
            f"--listen 0.0.0.0:{API_PORT} "
            f"--llama-checkpoint-path {MODEL_DIR} "
            f"--decoder-checkpoint-path {MODEL_DIR}/codec.pth "
            "--decoder-config-name modded_dac_vq"
        ),
    ]

    print(f"Starting Fish Speech server for {MODEL_REPO} on port {API_PORT}")
    subprocess.Popen(cmd)


def _post_json_bytes(url: str, payload: dict, retries: int = 20, delay_s: int = 15) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=300) as response:
                return response.read()
        except HTTPError as exc:
            last_error = exc
            if exc.code == 503 and attempt < retries:
                time.sleep(delay_s)
                continue
            raise
        except URLError as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(delay_s)
                continue
            raise

    if last_error:
        raise last_error
    raise RuntimeError("Request failed without a captured exception")


def _reference_text_from_transcript_json(transcript_path: Path, max_chars: int = 140) -> str:
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    lines: list[str] = []
    total = 0
    for item in transcript:
        line = item["text"].splitlines()[0].strip()
        if not line:
            continue
        total += len(line)
        lines.append(line)
        if total >= max_chars:
            break
    return "".join(lines)


@app.local_entrypoint()
def test(
    prompt: str = "Hello from Fish Speech S2 running on Modal.",
    output_path: str = "/tmp/fish-speech/output.wav",
    ensure_download: bool = True,
    reference_audio_path: str = "",
    reference_text: str = "",
    reference_transcript_json: str = "",
):
    if ensure_download:
        print(download_model.remote())

    url = serve.get_web_url().rstrip("/")
    print(f"Fish Speech server URL: {url}")

    references = []
    if reference_audio_path:
        audio_path = Path(reference_audio_path)
        resolved_reference_text = reference_text
        if not resolved_reference_text and reference_transcript_json:
            resolved_reference_text = _reference_text_from_transcript_json(
                Path(reference_transcript_json)
            )
        if not resolved_reference_text:
            raise ValueError(
                "Provide reference_text or reference_transcript_json when using reference_audio_path."
            )
        references.append(
            {
                "audio": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
                "text": resolved_reference_text,
            }
        )

    audio_bytes = _post_json_bytes(
        f"{url}/v1/tts",
        {
            "text": prompt,
            "format": "wav",
            "latency": "normal",
            "streaming": False,
            "normalize": True,
            "references": references,
        },
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(audio_bytes)
    print(f"Saved audio to {output_file}")
