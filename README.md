# qwen-wake-word

Wake word detection using a finetuned variant of [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) via [py-qwen3-asr-cpp](https://github.com/femelo/py-qwen3-asr-cpp) and [vad](https://github.com/snakers4/silero-vad) for speech detection.

It is designed to run on a Raspberry Pi 5 with near real time performance.
Meaning it is able to check for a wake word in 1.5 seconds which feels quite natural in a real world example.

## Requirements

- NixOS / Nix with flakes enabled (alternatively run `./setup.sh` on a Pi 5)
- Prefarable a GPU
- A microphone

## Quick Start

```bash
git clone https://github.com/micartey/qwen-wake-word.git
cd qwen-wake-word
nix develop --command python wake-word.py
```

## Configuration

Edit the constants at the top of `wake_word.py`:

| Variable          | Default     | Description                                 |
| ----------------- | ----------- | ------------------------------------------- |
| `WAKE_WORD`       | `hey sarah` | Wake word to listen for                     |
| `MAX_DISTANCE`    | `4`         | Max Levenshtein distance for fuzzy matching |
| `CHUNK_SECONDS`   | `1.5`       | Audio chunk length in seconds               |
| `OVERLAP_SECONDS` | `0.75`      | Overlap between chunks                      |
| `SAMPLE_RATE`     | `16000`     | Sample rate of a mic                        |
| `N_THREADS`       | `4`         | CPU threads for inference                   |

If you mic does not support a sample rate of 16000, you should create a `asoundrc` config.

```bash
cat << 'EOF' > ~/.asoundrc
pcm.!default {
    type asym
    capture.pcm "mic_plug"
}

pcm.mic_plug {
    type plug
    slave.pcm "hw:2,0"
}
EOF
```

_(Run `python -m sounddevice` and replace `hw:2,0` with whatevery your mic is listed as)_

## Download Model

The model would theoretically downloaded automatical from huggingface.
You can also pre-download the finetuned model if the huggingface download doesn't work for you.

```bash
wget https://cdn.micartey.dev/api/v1/download/blob/qwen3-asr-0.6b-finetuned-q5_k.gguf
```

Simply place it in the root of the project next to the wake word python script.

## License

Apache 2.0 (same as Qwen3-ASR)
