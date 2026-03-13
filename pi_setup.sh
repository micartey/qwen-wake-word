sudo apt update

sudo apt install -y vim kitty btop

sudo apt install -y \
    python3-pip python3-venv python3-dev pipx \
    libsndfile1 ffmpeg make git \
    portaudio19-dev libasound2-plugins

pipx install uv
export PATH=/home/pi/.local/bin:$PATH # To make uv available in current bash

uv venv .venv
source .venv/bin/activate

uv pip install sounddevice numpy huggingface-hub torch torchaudio
uv pip install "py-qwen3-asr-cpp @ git+https://github.com/femelo/py-qwen3-asr-cpp"

# CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_ARM_DOTPROD=OFF -DGGML_ARM_I8MM=OFF" uv pip install --reinstall --no-cache "py-qwen3-asr-cpp @ git+https://github.com/femelo/py-qwen3-asr-cpp"

# arecord -l
# python -m sounddevice
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
