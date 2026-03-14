# Finetuning

1. Rent a server at [runpod.io](https://runpod.io?ref=ihq2yc86) of kind Ubuntu
2. Clone the repository or copy it via scp
3. Run:

```bash
./setup.py
source .venv-train/bin/activate
```

## Generate Data

You can download the dataset using wget:

```bash
wget https://cdn.micartey.dev/api/v1/download/blob/qwen3-dataset.tar.gz
tar -xzf qwen3-dataset.tar.gz

# Replace absolute paths
sed -i 's|/home/daniel/Workspace/Python|/root|g' ./data/train.jsonl
sed -i 's|/home/daniel/Workspace/Python|/root|g' ./data/eval.jsonl
```

Or generate one from scratch:

```bash
python finetuning/prepare_data.py \
    --librispeech_hours 360 \
    --librispeech_split train.clean.360 \
    --output_dir ./data
```

## Finetune

```bash
python finetuning/train_full.py \
  --train_file ./data/train.jsonl \
  --eval_file ./data/eval.jsonl \
  --output_dir ./output
```

## Evaluate

```bash
python finetuning/evaluate.py \
  --model_path ./output/final_model \
  --eval_file ./data/eval.jsonl
```

The output will look as follows:

```
=================================================
Results (968 samples):
  WER: 0.0361 (3.61%)
  CER: 0.0135 (1.35%)
==================================================
  English WER: 0.0361 (3.61%) [968 samples]
```

For reference: Whisper large-v3 has a score of `~3.0%` and base qwen3 has a score of `~6%` so this is almost as good as the large whisper model and twice as good as the base qwen3 model.

However, these values are bollocks as the data, as well as testing data, have been generated with the same tool.

## Convert to GGUF

```bash
python finetuning/convert_to_gguf.py \
  -i ./output/final_model \
  -o ./qwen3-asr-0.6b-finetuned-f16.gguf -t f16 # q8_0
```

## Quantize

```bash
python finetuning/quantize_gguf.py \
  ./qwen3-asr-0.6b-finetuned-f16.gguf \
  ./qwen3-asr-0.6b-finetuned-q5_k.gguf -t q5_k
```
