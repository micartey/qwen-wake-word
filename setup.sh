#!/usr/bin/env bash
set -euo pipefail

echo "=== Qwen3-ASR Finetuning Setup (Ubuntu + A100 80GB) ==="

apt-get update
apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git git-lfs cmake gcc g++ make \
    libsndfile1 ffmpeg \
    wget curl

git lfs install

if ! command -v nvidia-smi &>/dev/null; then
    echo "WARNING: nvidia-smi not found. CUDA drivers may not be installed."
    echo "Most cloud GPU instances come with drivers pre-installed."
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

python3 -m venv .venv-train
source .venv-train/bin/activate

pip install --upgrade pip wheel setuptools

pip install "torch>=2.6" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install \
    qwen-asr \
    datasets \
    peft \
    librosa \
    soundfile \
    jiwer \
    gguf \
    safetensors \
    tqdm

pip uninstall -y torchcodec 2>/dev/null || true

MAX_JOBS=4 pip install flash-attn --no-build-isolation || echo "flash-attn build failed (optional, training will still work)"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Activate the environment:"
echo "  source .venv-train/bin/activate"
echo ""
echo "Then run:"
echo "  # 1. Prepare data"
echo "  python finetuning/prepare_data.py --output_dir ./data"
echo ""
echo "  # 2. Train (full SFT on A100)"
echo "  python finetuning/train_full.py \\"
echo "    --train_file ./data/train.jsonl \\"
echo "    --eval_file ./data/eval.jsonl \\"
echo "    --output_dir ./output"
echo ""
echo "  # 3. Evaluate"
echo "  python finetuning/evaluate.py \\"
echo "    --model_path ./output/final_model \\"
echo "    --eval_file ./data/eval.jsonl"
echo ""
echo "  # 4. Convert to GGUF"
echo "  python finetuning/convert_to_gguf.py \\"
echo "    -i ./output/final_model \\"
echo "    -o ./qwen3-asr-0.6b-finetuned-f16.gguf -t f16"
