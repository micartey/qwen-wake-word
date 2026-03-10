# qwen-wake-word

Wake word detection using [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) GGUF models via [py-qwen3-asr-cpp](https://github.com/femelo/py-qwen3-asr-cpp). Designed to run on a Raspberry Pi.

Listens to the microphone, transcribes audio in real-time, and prints `Hello World` when the wake word is detected. Uses Levenshtein distance for fuzzy matching to handle slight transcription errors.

## Requirements

- NixOS / Nix with flakes enabled
- A microphone

## Quick Start

```bash
git clone https://github.com/micartey/qwen-wake-word.git
cd qwen-wake-word
nix develop --command python app.py
```

The first run will create a virtualenv, install dependencies, and download the model (~3.2 GB for the default 1.7B Q8 model).

## Configuration

Edit the constants at the top of `app.py`:

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `qwen3-asr-1.7b-q8-0` | Model to use (see table below) |
| `WAKE_WORD` | `sarah` | Word to listen for |
| `MAX_DISTANCE` | `2` | Max Levenshtein distance for fuzzy matching |
| `CHUNK_SECONDS` | `1.5` | Audio chunk length in seconds |
| `OVERLAP_SECONDS` | `0.75` | Overlap between chunks |
| `SILENCE_THRESHOLD` | `0.005` | RMS energy below which chunks are skipped |
| `N_THREADS` | `4` | CPU threads for inference |

## Available Models

These are built into `py-qwen3-asr-cpp` and download automatically by name.

| Model ID | Size | RAM | Target |
|---|---|---|---|
| `qwen3-asr-0.6b-q4-k-m` | 685 MB | ~1 GB | Pi 4 (4 GB) |
| `qwen3-asr-0.6b-q5-k-m` | 800 MB | ~1.2 GB | Pi 4 (4 GB) |
| `qwen3-asr-0.6b-q8-0` | 1.35 GB | ~1.8 GB | Pi 4 (4 GB) |
| `qwen3-asr-0.6b-f16` | 1.88 GB | ~2.4 GB | Pi 4 (4 GB) |


## How It Works

1. Captures audio from the microphone in 1.5-second overlapping chunks
2. Skips silent chunks based on RMS energy
3. Transcribes each chunk using the Qwen3-ASR GGUF model (C++ backend, no PyTorch)
4. Compares each word in the transcription against the wake word using Levenshtein distance
5. Triggers when any word is within `MAX_DISTANCE` edits of the wake word

## Notes

- The C++ backend does not support forcing a language — it always auto-detects. The fuzzy matching approach works regardless of which language the model picks.
- On a Pi 4, expect ~1-2s latency per chunk with the Q4 0.6B model. 
- Models are cached locally after first download (`~/.local/share/py_qwen3_asr_cpp/models/`).

## License

Apache 2.0 (same as Qwen3-ASR)
