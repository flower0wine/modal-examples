# Fish Speech on Modal Notes

This document records the practical setup, implementation, testing workflow, and pitfalls encountered while deploying Fish Speech on Modal in this repo.

## Goal

Deploy a usable Fish Speech TTS demo on Modal, then test it locally with reference voice cloning and keep the process reproducible.

## Current Implementation

Relevant files:

- `06_gpu_and_ml/text-to-audio/fish_speech_tts.py`
- `06_gpu_and_ml/text-to-audio/fish_speech_tts_test.py`

Current deployed service:

- `https://flowerwine8023--fish-speech-s2-pro-demo-serve.modal.run`

Current model choice:

- `fishaudio/s2-pro`

Why this model:

- Fish Audio's current official inference docs center on `s2-pro`.
- Official docs recommend at least `24 GB` VRAM for S2 inference.
- In practice, this deployment used about `22-24 GB` GPU memory after warmup.

GPU used:

- `L40S`

## What Worked

### 1. Modal authentication

Authentication was completed successfully before deployment.

Useful check:

```powershell
.\.venv\Scripts\python.exe -m modal profile current
```

### 2. Deployment approach

The successful path was:

- Use `modal.Image.from_registry("fishaudio/fish-speech:latest", add_python="3.12")`
- Add `huggingface_hub` explicitly
- Cache model weights in a Modal `Volume`
- Start the official Fish Speech API server via `tools/api_server.py`

### 3. Weight caching

Model weights are downloaded once into:

- Modal volume name: `fish-speech-models`
- In-container path: `/models/s2-pro`

This avoids redownloading the model on every request.

### 4. End-to-end synthesis

Both of these worked:

- No-reference generation
- Reference-audio generation using uploaded/base64 audio

Generated test artifacts in this workspace:

- `tmp/fish-speech-demo.wav`
- `tmp/fish-speech-mizuki-ref-1.wav`
- `tmp/fish-speech-mizuki-ref-2.wav`
- `tmp/fish-speech-mizuki-orig-1.wav`
- `tmp/fish-speech-mizuki-orig-2.wav`
- `tmp/fish-speech-test-script.wav`

## Commands That Worked

### Deploy

```powershell
.\.venv\Scripts\python.exe -m modal deploy 06_gpu_and_ml\text-to-audio\fish_speech_tts.py
```

### Local smoke test through Modal

```powershell
.\.venv\Scripts\python.exe -m modal run 06_gpu_and_ml\text-to-audio\fish_speech_tts.py --prompt "你好，欢迎来到 Modal 上的 Fish Speech 演示。" --output-path "tmp\fish-speech-demo.wav"
```

### Call deployed service directly

```powershell
curl -X POST "https://flowerwine8023--fish-speech-s2-pro-demo-serve.modal.run/v1/tts" `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"你好，欢迎来到 Modal 上的 Fish Speech 演示。\",\"format\":\"wav\",\"latency\":\"normal\",\"streaming\":false,\"normalize\":true}" `
  --output tmp\direct-call.wav
```

### Use the local helper test script

```powershell
.\.venv\Scripts\python.exe 06_gpu_and_ml\text-to-audio\fish_speech_tts_test.py
.\.venv\Scripts\python.exe 06_gpu_and_ml\text-to-audio\fish_speech_tts_test.py --preset mizuki-1
.\.venv\Scripts\python.exe 06_gpu_and_ml\text-to-audio\fish_speech_tts_test.py --prompt "今天感觉怎么样？"
.\.venv\Scripts\python.exe 06_gpu_and_ml\text-to-audio\fish_speech_tts_test.py --no-reference
```

## Reference Audio Workflow

### Best practice

Fish Speech works better when reference input contains:

- Clean single-speaker audio
- No music or SFX
- Accurate matching transcript
- Around `5-30` seconds of speech

### Why the first Chinese output sounded foreign

The initial test used no reference audio. Fish Speech then selected a random voice style, which can produce Chinese with a non-native accent.

Using reference audio + matching text improved this significantly.

### Current local reference assets

User-provided reference files:

- `tmp/references/audio/1.mp3`
- `tmp/references/audio/2.mp3`

Their matching texts:

- `1.mp3`: `昨晚的梦境…味道很美妙，感谢你与我分享。`
- `2.mp3`: `如果心里郁闷的话，就先放下手头的事，出去呼吸一下新鲜空气吧`

Built-in presets in `06_gpu_and_ml/text-to-audio/fish_speech_tts_test.py`:

- `mizuki-1`
- `mizuki-2`

### Public source investigation

Public sources were investigated for Yumemizuki Mizuki voice lines.

Useful pages:

- [Yumemizuki Mizuki Chinese voice-over page](https://genshin-impact.fandom.com/wiki/Yumemizuki_Mizuki/Voice-Overs/Chinese)
- [Yumemizuki Mizuki Chinese voice-over category](https://genshin-impact.fandom.com/wiki/Category%3AYumemizuki_Mizuki_Chinese_Voice-Overs)

Important limitation:

- Direct automated downloading from Fandom file pages repeatedly hit Cloudflare/anti-bot protection.
- Because of that, manually acquired audio files were more reliable than automated scraping from Fandom.

### Temporary fallback source used during testing

These public YouTube sources were used for intermediate testing:

- [Version 5.4 "Moonlight Amidst Dreams" Trailer (CN Voiced)](https://www.youtube.com/watch?v=hhlUok5azfc)
- [Character Teaser - "Yumemizuki Mizuki: Dining on a Dish of Dreams"](https://www.youtube.com/watch?v=eP9NZFD86b4)

These are acceptable for experiments but not ideal for final cloning because:

- they may contain music
- they may contain multiple speakers
- subtitles may not align perfectly with clipped audio

## Problems Encountered and Fixes

### Problem 1: `modal` command not found

Cause:

- `modal` was not installed globally.

Fix:

- Use the project virtualenv.
- Run through `python -m modal` or the virtualenv executable.

### Problem 2: Heavy custom CUDA image build was too slow

Cause:

- Building Fish Speech from raw CUDA base image in Modal pulled too many system packages and took too long.

Fix:

- Switched to the official image: `fishaudio/fish-speech:latest`

### Problem 3: Modal could not determine Python version in the registry image

Observed error:

- Modal failed to determine the Python version in the supplied image.

Fix:

- Use:

```python
modal.Image.from_registry("fishaudio/fish-speech:latest", add_python="3.12")
```

### Problem 4: `huggingface_hub` missing in the official image

Observed error:

- `ModuleNotFoundError: No module named 'huggingface_hub'`

Fix:

- Explicitly install it:

```python
.pip_install("huggingface_hub[hf_transfer]")
```

### Problem 5: First-run startup is slow

Cause:

- Initial image hydration
- Weight download
- Server startup
- Warmup inference

Observed behavior:

- First request can take minutes.
- Later requests are much faster.

Mitigation:

- Keep the service deployed.
- Reuse the same model volume.
- Avoid repeated `modal run` for normal use when a deployed URL already exists.

### Problem 6: Fandom voice file automation failed

Cause:

- Cloudflare blocks or alters bot traffic on file pages.

Mitigation:

- Download audio manually in a browser when needed.
- Store the clean audio under `tmp/references/audio`.

### Problem 7: Reference text/audio mismatch hurts quality

Cause:

- A long transcript was once paired with a shorter clipped audio segment.

Effect:

- Voice cloning quality degrades.

Rule:

- Always use transcript text that matches the exact reference audio span.

### Problem 8: Network/TLS interruption during direct HTTPS requests

Observed error:

- `SSLError: UNEXPECTED_EOF_WHILE_READING`

Fix:

- Retry the request rather than assuming a model failure.
- This was a network-layer issue, not an inference issue.

## Current API Shape

The deployed service accepts:

- `text`
- `format`
- `latency`
- `streaming`
- `normalize`
- `references`

Reference shape:

```json
{
  "audio": "<base64-audio>",
  "text": "matching transcript"
}
```

## Recommended Future Workflow

When trying again later, use this sequence:

1. Confirm Modal auth works.
2. Confirm deployed URL still exists.
3. Prefer direct calls to deployed URL instead of `modal run`.
4. Use a clean single-speaker reference audio file.
5. Ensure transcript matches the exact audio span.
6. Start with a short Chinese prompt.
7. Compare `mizuki-1` and `mizuki-2`.
8. Only re-deploy if the implementation changed.

## Recommended Improvements

Potential next steps:

- Add batch prompt testing to `fish_speech_tts_test.py`
- Add a preset output naming convention
- Add automatic retries around direct HTTP calls
- Add a dedicated "Mizuki preset" wrapper endpoint
- Add support for reading prompts from a `.txt` file

## Notes About Cost and Performance

- `s2-pro` is heavy for demo use.
- `L40S` is workable.
- For cheaper experiments, a smaller model such as `s1-mini` may be worth testing later.
- For best voice quality, `s2-pro` remains the safer choice.
