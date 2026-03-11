# Qwen3-ASR Finetuning for English Precision

## Architecture
- Encoder-decoder: Whisper-style audio encoder (18 layers/0.6B) + Qwen3 causal LM decoder (28 layers)
- Model class: `Qwen3ASRForConditionalGeneration`, model_type: `qwen3_asr`
- Requires `qwen-asr` package (not standard `AutoModel`), needs `transformers>=4.57.6`
- Audio: 128 mel-bin spectrograms at 16kHz, vocab=151,936

## Finetuning Approach: LoRA
- Official SFT script: https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning/
- LoRA reference: https://github.com/ysys12138/Qwen3-ASR-Lora-finetune
- Config: rank=16, alpha=32, dropout=0.05, target="all-linear", lr=2e-4
- ~4GB VRAM for 0.6B with LoRA + gradient checkpointing

## Data Format (JSONL)
```json
{"audio": "/path/to/file.wav", "text": "language English<asr_text>The actual transcript."}
```
- Language prefix: `language English<asr_text>`, `language German<asr_text>`, `language None<asr_text>`

## Datasets
- LibriSpeech train-clean-100 (100h, clean English)
- Common Voice EN (diverse accents)
- Optional: Common Voice DE for German

## Key Gotchas
- Must patch `forward()` on outer model to delegate to `model.thinker`
- Must patch `get_input_embeddings` / `set_input_embeddings` for PEFT compatibility
- Checkpoint callback copies HF config files so each checkpoint is self-contained

## GGUF Conversion
- **IS possible** via `predict-woo/qwen3-asr.cpp` converter format
- Single GGUF file contains BOTH audio encoder + text decoder
- Converter: `finetuning/convert_to_gguf.py` (adapted from qwen3-asr.cpp's `scripts/convert_hf_to_gguf.py`)
- Supports f16, f32, q8_0 output types
- For further quantization (q4_k_m, q5_k_m), use llama.cpp's `llama-quantize` tool on the f16 GGUF
- Existing GGUF models on HF: `FlippyDora/qwen3-asr-0.6b-GGUF`, `OpenVoiceOS/qwen3-asr-0.6b-q5-k-m`
- Alternative split approach: `HaujetZhao/Qwen3-ASR-GGUF` (ONNX encoder + GGUF decoder separately)

## Deployment After Finetuning
1. Merge LoRA: `python finetuning/merge_lora.py`
2. Convert to GGUF: `python finetuning/convert_to_gguf.py -i ./merged_model -o model.gguf -t f16`
3. Quantize (optional): `llama-quantize model.gguf model-q5_k_m.gguf q5_k_m`
4. Use with `py-qwen3-asr-cpp` on Pi (same as current app.py, just point to local GGUF path)
