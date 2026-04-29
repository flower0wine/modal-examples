"""
Deploy CosyVoice2-0.5B on Modal for zero-shot text-to-speech testing.

Usage:

1. Deploy the API server:
   modal deploy 06_gpu_and_ml/text-to-audio/cosyvoice2_tts.py

2. Run a smoke test and save audio locally:
   modal run 06_gpu_and_ml/text-to-audio/cosyvoice2_tts.py --prompt "你好，欢迎使用 Modal 上的 CosyVoice2。"

Notes:
- The first run downloads model weights into a Modal Volume and can take a while.
- CosyVoice2-0.5B is primarily a zero-shot voice cloning model, so reference audio
  and matching text usually produce better and more predictable results.
- If no reference is provided, the example uses CosyVoice's bundled prompt audio.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import modal

MINUTES = 60
MODEL_REPO = os.environ.get("COSYVOICE_MODEL_REPO", "FunAudioLLM/CosyVoice2-0.5B")
MODEL_SLUG = MODEL_REPO.split("/")[-1]
MODEL_DIR = f"/models/{MODEL_SLUG}"
COSYVOICE_ROOT = "/workspace/CosyVoice"
DEFAULT_PROMPT_WAV = f"{COSYVOICE_ROOT}/asset/zero_shot_prompt.wav"
DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"
GPU_TYPE = os.environ.get("COSYVOICE_GPU", "L40S")

app = modal.App(f"cosyvoice2-{MODEL_SLUG.lower()}-demo")

model_volume = modal.Volume.from_name("cosyvoice-models", create_if_missing=True)

image = (
    modal.Image.micromamba(python_version="3.10")
    .apt_install("build-essential", "ffmpeg", "git", "git-lfs", "sox", "libsox-dev")
    .micromamba_install("pynini==2.1.5", channels=["conda-forge"])
    .run_commands(
        "git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "
        f"{COSYVOICE_ROOT}",
        "python -m pip install 'setuptools<81' wheel",
        "python -m pip install --no-build-isolation openai-whisper==20231117",
        "cd /workspace/CosyVoice && python -m pip install -r requirements.txt",
        "python -m pip uninstall -y deepspeed",
        "python -m pip install 'huggingface_hub[hf_transfer]'",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONPATH": f"{COSYVOICE_ROOT}:{COSYVOICE_ROOT}/third_party/Matcha-TTS",
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
    ready_marker = destination / "cosyvoice2.yaml"

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


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/models": model_volume},
    scaledown_window=15 * MINUTES,
    timeout=45 * MINUTES,
)
@modal.concurrent(max_inputs=4)
class CosyVoice2:
    @modal.enter()
    def load(self):
        import torch
        from cosyvoice.cli.cosyvoice import AutoModel

        model_path = Path(MODEL_DIR)
        if not (model_path / "cosyvoice2.yaml").exists():
            raise RuntimeError(
                f"Model weights are missing under {model_path}. "
                "Run download_model first."
            )

        print(f"Loading {MODEL_REPO} from {model_path}")
        self.model = AutoModel(model_dir=str(model_path))
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CosyVoice2 sample rate: {self.model.sample_rate}")

    def _load_prompt_speech(self, reference_audio: str):
        import io

        from cosyvoice.utils.file_utils import load_wav

        if reference_audio:
            return load_wav(io.BytesIO(base64.b64decode(reference_audio)), 16000)
        return load_wav(DEFAULT_PROMPT_WAV, 16000)

    @modal.method()
    def generate(
        self,
        text: str,
        reference_audio: str = "",
        reference_text: str = "",
        stream: bool = False,
        text_frontend: bool = True,
    ) -> bytes:
        import io

        import torch
        import torchaudio

        if not text.strip():
            raise ValueError("text must not be empty")

        prompt_text = reference_text or DEFAULT_PROMPT_TEXT
        prompt_speech_16k = self._load_prompt_speech(reference_audio)

        chunks = []
        with torch.inference_mode():
            for result in self.model.inference_zero_shot(
                text,
                prompt_text,
                prompt_speech_16k,
                stream=stream,
                text_frontend=text_frontend,
            ):
                chunks.append(result["tts_speech"].cpu())

        if not chunks:
            raise RuntimeError("CosyVoice2 returned no audio")

        wav = torch.cat(chunks, dim=-1)
        buffer = io.BytesIO()
        torchaudio.save(buffer, wav, self.model.sample_rate, format="wav")
        buffer.seek(0)
        return buffer.read()

    @modal.asgi_app()
    def api(self):
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import Response

        web_app = FastAPI(title="CosyVoice2 TTS API")

        @web_app.get("/health")
        async def health():
            return {"status": "ok", "model": MODEL_REPO}

        @web_app.post("/v1/tts")
        async def tts(payload: dict):
            output_format = payload.get("format", "wav")
            if output_format != "wav":
                raise HTTPException(status_code=400, detail="Only wav output is supported")

            reference_audio = ""
            reference_text = ""
            references = payload.get("references") or []
            if references:
                reference_audio = references[0].get("audio", "")
                reference_text = references[0].get("text", "")

            text = payload.get("text", "")
            streaming = bool(payload.get("streaming", False))
            normalize = bool(payload.get("normalize", True))
            text_frontend = bool(payload.get("text_frontend", True))

            try:
                audio_bytes = self.generate.local(
                    text=text,
                    reference_audio=reference_audio,
                    reference_text=reference_text,
                    stream=streaming,
                    text_frontend=text_frontend and normalize,
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            return Response(content=audio_bytes, media_type="audio/wav")

        return web_app


def _post_json_bytes(url: str, payload: dict, retries: int = 20, delay_s: int = 15) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=600) as response:
                return response.read()
        except HTTPError as exc:
            last_error = exc
            error_body = exc.read().decode("utf-8", errors="replace")
            if error_body:
                print(f"HTTP {exc.code} response body: {error_body}")
            if exc.code in {502, 503, 504} and attempt < retries:
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


@app.local_entrypoint()
def test(
    prompt: str = "你好，欢迎使用 Modal 上的 CosyVoice2 语音合成服务。",
    output_path: str = "/tmp/cosyvoice2/output.wav",
    ensure_download: bool = True,
    reference_audio_path: str = "",
    reference_text: str = "",
    text_frontend: bool = True,
):
    if ensure_download:
        print(download_model.remote())

    url = CosyVoice2().api.get_web_url().rstrip("/")
    print(f"CosyVoice2 server URL: {url}")

    references = []
    if reference_audio_path:
        audio_path = Path(reference_audio_path)
        if not reference_text:
            raise ValueError("Provide reference_text when using reference_audio_path.")
        references.append(
            {
                "audio": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
                "text": reference_text,
            }
        )

    audio_bytes = _post_json_bytes(
        f"{url}/v1/tts",
        {
            "text": prompt,
            "format": "wav",
            "streaming": False,
            "normalize": True,
            "text_frontend": text_frontend,
            "references": references,
        },
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(audio_bytes)
    print(f"Saved audio to {output_file}")
