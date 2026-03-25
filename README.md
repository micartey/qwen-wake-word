# qwen-wake-word

<div align="center">
    <img src="./.files/qwen-wake-word-banner.png" />
</div>

<br />

<div align="center">
    <img
        src="https://img.shields.io/badge/Written%20in-python-%23F2B655?style=for-the-badge"
        height="30"
    />
    <img
        src="https://img.shields.io/badge/Finetuned%20Model-qwen3%20asr%200.6b%20english%20v2-8379ec?style=for-the-badge"
        height="30"
    />
</div>

<br />

<div align="center">
    <a href="https://www.buymeacoffee.com/micartey" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;">
    </a>
    <br />
    <i>(Finetuning costs money 🫠)</i>
</div>

<br />

<p align="center">
  <a href="#-introduction">Introduction</a> •
  <a href="#-getting-started">Getting started</a> •
  <a href="https://huggingface.co/micartey/qwen3-asr-0.6b-english-v2/tree/main">Huggingface</a>
</p>

## 📚 Introduction

Wake word detection using a finetuned variant of [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) via [py-qwen3-asr-cpp](https://github.com/femelo/py-qwen3-asr-cpp) and [vad](https://github.com/snakers4/silero-vad) for speech detection.
The qwen asr model has been force aligned to detect only english.
For other languages, please take a look at the `finetuning` directory and create a custom finetuned model.

This project designed to run on a Raspberry Pi 5 with near real time performance.
Meaning it is able to check for a wake word in 1.5 seconds which feels quite natural in a real world example.

### Motivation

Wake word detections require custom models and retraining for different wake words.
I wanted a more generic approach by using a foundation model which can be finetuned, but generally works out of the box and generically.
Changing the wake word is mainly a matter of configuration (depending on the name / phrase).

I also wanted to have an alternative to picovoice as they are a very questionable company with [bad reputation](https://www.reddit.com/r/cscareerquestionsCAD/comments/qee7zp/picovoice_vancouver_interview_dlsde_roles/) that does not follow their own ToS and abuses their power on the selection of their users - e.g. by geo-locking accounts that are perfectly ToS complient.

### TODO

- [x] Create automation for generating the asoundrc config (see below)
- [x] API Interface to trigger action, script, service, ...
- [ ] Improve timeout detection by forcefully kill the ASR thread
- [ ] Create v3 of finetuned model using 3:1 or higher mix of librispeech and common voice

## 🚀 Getting Started

> [!WARNING]  
> A Pi 4 does not have sufficient computation power to run this model according to my own testing.
> This might be due to a flawed setup, but most likely due to the difference in computational power and the use of a more powerful onboard GPU.
> It might be possible to improve speed even further by using a HALO accelerator - although that is unlikly as my 4080 super isn't that much faster.

- Nix flakes (alternatively run `./setup.sh` on a Pi 5)
- A microphone

Active cooling is recommended on a pi 5 or similar single board computers / embedded devices.
Especially to counter possible thermal throttling or damaging of components.

```bash
git clone https://github.com/micartey/qwen-wake-word.git
cd qwen-wake-word
nix develop --command python wake_word.py -w "hey sarah" "hi sarah" "hello sarah" -c "echo 'Hello World!'"
```

Alternativly run:

```bash
./setup.sh
source .venv/bin/activate
python wake_word.py -w "hey sarah" "hi sarah" "hello sarah" -c "echo 'Hello World!'"
```

When you just want to checkout the model or real-time transcription, run:

```bash
python transcribe.py
```

### Configuration

There is a lot you can configure without the need to touch the source code.
You can specify your own wake words / phrases, configure the chunk length or max levenshtein distance using the following CLI flags when running the script.

Sometimes, you also need additional settings for your microphone, depending on the supported sample rate.

#### CLI Flags

| Flag                    | Default                                    | Description                                                         |
| ----------------------- | ------------------------------------------ | ------------------------------------------------------------------- |
| `-w` / `--wake-words`   | `"hey sarah"` `"hi sarah"` `"hello sarah"` | Wake word phrases to listen for                                     |
| `-c` / `--command`      | _none_                                     | Shell command to execute (same thread) when a wake word is detected |
| `-d` / `--max-distance` | `2`                                        | Max Levenshtein distance for fuzzy matching                         |
| `--chunk-seconds`       | `1.5`                                      | Audio chunk length in seconds                                       |
| `--overlap-seconds`     | `0.75`                                     | Overlap between chunks in seconds                                   |

#### Microphone

If your mic does not support a sample rate of 16000 out of the box, you should create a `asoundrc` config.
The included script will list all capture devices and generate `~/.asoundrc` for the selected device:

```bash
./generate-asoundrc.sh
```

Or create the config manually:

```zsh
cat << 'EOF' > ~/.asoundrc
pcm.!default {
    type asym
    capture.pcm "mic_plug"
}

pcm.mic_plug {
    type plug
    slave.pcm "hw:0,0"
}
EOF
```

_(You can use `python -m sounddevice` to list your devices and extract the correct `hw:` value)_

### Download Model

The model will be downloaded automatical from huggingface.
You can also pre-download the finetuned model if the huggingface download doesn't work for you.

```bash
wget https://cdn.micartey.dev/api/v1/download/blob/qwen3-asr-0.6b-finetuned-q5_k.gguf
```

Simply place it in the root of the project next to the wake word python script.
