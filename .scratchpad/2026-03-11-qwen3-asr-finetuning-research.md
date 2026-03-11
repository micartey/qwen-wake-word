# Qwen3-ASR Finetuning Research

Date: 2026-03-11

## 1. Model Architecture

### Overview
Qwen3-ASR is an **encoder-decoder** style multimodal ASR model. It uses a "thinker" architecture with two main components:

1. **Audio Encoder** (`qwen3_asr_audio_encoder`): Whisper-style encoder
2. **Text Decoder** (`qwen3` type): A Qwen3 causal language model

The model processes audio through the encoder, projects it into the LLM's embedding space, then autoregressively decodes text tokens.

### 0.6B Model Config
- **Audio Encoder**: 18 layers, 14 attention heads, d_model=896, FFN dim=3584, 128 mel bins, output_dim=1024, downsample_hidden_size=480
- **Text Decoder (Qwen3)**: 28 layers, 16 attention heads, 8 KV heads, hidden_size=1024, intermediate=3072, head_dim=128, vocab=151936
- **RoPE**: mrope (multimodal RoPE) with sections [24, 20, 20], theta=1M
- **Total size**: ~1.88 GB safetensors (bfloat16)

### 1.7B Model Config
- **Audio Encoder**: 24 layers, 16 attention heads, d_model=1024, FFN dim=4096, 128 mel bins, output_dim=2048, downsample_hidden_size=480
- **Text Decoder (Qwen3)**: 28 layers, 16 attention heads, 8 KV heads, hidden_size=2048, intermediate=6144, head_dim=128, vocab=151936
- **RoPE**: same mrope config as 0.6B
- **Total size**: ~2B params

### Key Architecture Details
- Architecture class: `Qwen3ASRForConditionalGeneration`
- model_type: `qwen3_asr`
- The model wraps in a "thinker" pattern: `model.thinker.forward()` handles the actual forward pass
- Audio is processed as 128 mel-bin spectrograms with chunked windowing (n_window=50, n_window_infer=800)
- Special tokens: audio_start=151669, audio_end=151670, audio_token=151676
- Supports streaming inference via chunked audio input
- Foundation: derived from Qwen3-Omni's audio understanding capabilities

## 2. HuggingFace Transformers Support

- **Supported**: Yes, via the `qwen-asr` Python package which wraps transformers
- The model uses `transformers>=4.57.6`
- Model class is registered as `qwen3_asr` model_type
- The `qwen-asr` package provides `Qwen3ASRModel.from_pretrained()` which loads it via transformers
- vLLM also has day-0 native support
- The model is NOT a standard HF model you load with `AutoModelForSpeechSeq2Seq` - it uses the custom `qwen-asr` package

## 3. Official Finetuning Scripts

### Location
`https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning/`

### Files
- `qwen3_asr_sft.py` - Official SFT (supervised fine-tuning) script
- `README.md` - Finetuning documentation

### Official Approach: Full SFT
The official script does **full parameter fine-tuning** (not LoRA) using HuggingFace `Trainer`.

**Key details:**
- Uses `qwen-asr` + `datasets` packages
- Requires FlashAttention 2 for efficiency
- Training is done on the `.thinker` submodule with a patched forward
- Data format: JSONL with `audio` (wav path) and `text` (transcript with language prefix)
- Text format: `language English<asr_text>The actual transcript here.`
- Uses standard HF TrainingArguments
- Default hyperparams: lr=2e-5, batch_size=32, grad_acc=4, linear LR schedule, warmup_ratio=0.02
- Supports multi-GPU via `torchrun`
- Saves checkpoints as HF-compatible model dirs (copies config files for inference)
- Checkpoints can be loaded directly with `Qwen3ASRModel.from_pretrained("checkpoint-dir")`

### Data Format (JSONL)
```jsonl
{"audio":"/data/wavs/utt0001.wav","text":"language English<asr_text>This is a test sentence."}
{"audio":"/data/wavs/utt0002.wav","text":"language German<asr_text>Das ist ein Test."}
{"audio":"/data/wavs/utt0003.wav","text":"language None<asr_text>Unknown language text."}
```

## 4. Community Finetuning Resources

### LoRA Finetuning
- **Repo**: `ysys12138/Qwen3-ASR-Lora-finetune` (GitHub)
  - LoRA via PEFT library
  - Params: lora_rank=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear"
  - LR=2e-4 for LoRA (10x higher than full SFT)
  - Requires `pip install peft`
  - Includes audio preprocessing utilities (resample, noise augmentation)

### Community Fine-tuned Models on HuggingFace
Based on Qwen3-ASR-1.7B:
1. **OzLabs/Caspi-1.7B** - Hebrew ASR (142 downloads, most popular finetune)
   - Full SFT on Hebrew datasets (ivrit-ai/crowd-transcribe-v5, ivrit-ai/crowd-recital-whisper-training)
   - Achieved 4.2% WER on Hebrew eval (vs 5.1% baseline)
2. **Kushtrim/Qwen3-ASR-1.7B-Albanian** - Albanian ASR
3. **Kushtrim/Qwen3-ASR-1.7B-Norwegian** - Norwegian ASR
4. **David-A-Amoo/Qwen3-ASR-1.7B-Yoruba** - Yoruba ASR

Based on Qwen3-ASR-0.6B:
- 8 finetune models, 1 adapter, 9 quantizations

### Other Notable Projects
- `jhqxxx/aha` - Rust inference library with Qwen3-ASR support
- `second-state/qwen3_asr_rs` - Rust implementation (184 stars)
- `moona3k/mlx-qwen3-asr` - Apple Silicon MLX port (22 stars)
- `andrewleech/qwen3-asr-onnx` - ONNX conversion

## 5. Datasets for English ASR Finetuning

### Tier 1 (Large, High-Quality)
| Dataset | Size | Description |
|---------|------|-------------|
| **LibriSpeech** | 960h | Read English audiobooks. Clean/other splits. Gold standard benchmark. |
| **GigaSpeech** | 10,000h | Multi-domain: YouTube, podcasts, audiobooks. Real-world audio. |
| **Common Voice** (en) | 3,000h+ | Crowdsourced read speech. Diverse accents. Mozilla project. |
| **MLS (Multilingual LibriSpeech)** | 44,000h (en) | Derived from LibriVox audiobooks. |
| **People's Speech** | 30,000h | Diverse open English ASR corpus. |
| **SPGISpeech** | 5,000h | Financial earnings calls. Professional domain. |

### Tier 2 (Specialized/Medium)
| Dataset | Size | Description |
|---------|------|-------------|
| **TED-LIUM 3** | 452h | TED talks transcriptions. |
| **VoxPopuli** | 400h (en) | European Parliament speeches. |
| **Fleurs** (en) | ~12h | Google's multilingual benchmark. Small but good for eval. |
| **WSJ** | 80h | Wall Street Journal read speech. Classic benchmark. |
| **AMI** | 100h | Meeting recordings. Multi-speaker. |

### For German
| Dataset | Size | Description |
|---------|------|-------------|
| **Common Voice** (de) | 1,500h+ | Crowdsourced German. |
| **MLS** (de) | 1,900h | German audiobooks. |
| **Fleurs** (de) | ~12h | Eval benchmark. |
| **VoxPopuli** (de) | 200h+ | EU Parliament, German. |
| **CoVoST 2** | - | Speech translation (can extract German ASR pairs). |

### Recommended for This Project
For English wake-word / ASR improvement:
- **LibriSpeech** (clean + other) for baseline quality
- **Common Voice English** for accent diversity
- **GigaSpeech** for real-world robustness
- For domain-specific: collect your own data in the target domain

## 6. Recommended Finetuning Approach

### For Small Datasets (<100h)
**Use LoRA** (Parameter-Efficient Fine-Tuning):
- Much less GPU memory needed
- Less risk of catastrophic forgetting
- Faster training
- Settings from community: rank=16, alpha=32, target="all-linear", lr=2e-4
- Can focus on just the text decoder, or both encoder and decoder

### For Medium Datasets (100-1000h)
**Full SFT** using the official script:
- Official settings: lr=2e-5, batch_size=32, grad_acc=4
- 1 epoch is usually sufficient
- Consider freezing the audio encoder and only training the decoder

### For Language Adaptation
The proven approach (from Caspi-1.7B, Albanian, Norwegian models):
1. Collect language-specific audio-transcript pairs
2. Format as JSONL with language prefix
3. Run full SFT with official script
4. 1 epoch, lr=2e-5

### General Best Practices
1. **Audio preprocessing**: Resample to 16kHz mono WAV
2. **Data augmentation**: Add noise, speed perturbation, volume changes
3. **Segment length**: Keep utterances reasonable (5-30 seconds optimal)
4. **Text normalization**: Consistent casing, punctuation, number formatting
5. **Evaluation**: Use WER on a held-out set; use greedy decoding for consistency
6. **Mixed training**: Include some original-language data to prevent forgetting

## 7. GGUF Conversion

### Current Status
**There is NO official GGUF support for Qwen3-ASR.**

The model architecture (`qwen3_asr`) is not a standard text-only LLM - it's a multimodal encoder-decoder model. GGUF/llama.cpp does not natively support this architecture.

### Alternative Quantization/Deployment Paths
1. **ONNX**: `andrewleech/qwen3-asr-onnx` has done ONNX conversion
2. **MLX**: `moona3k/mlx-qwen3-asr` for Apple Silicon
3. **Rust/C inference**: `second-state/qwen3_asr_rs` and `jhqxxx/aha` for native inference
4. **vLLM**: Official support, best for server deployment
5. **INT8 OpenVINO**: `dseditor/Qwen3-ASR-1.7B-INT8_OpenVINO`

### If You Need GGUF
The model has two components that would need separate handling:
- Audio encoder: Could potentially be converted separately
- Text decoder: Based on Qwen3 architecture which IS supported in llama.cpp
- The projection/adapter layers between them are the challenge

Practically, you would need to:
1. Export the text decoder weights separately
2. Convert audio encoder to a separate format (ONNX)
3. Build a custom inference pipeline that connects them

**This is non-trivial and not recommended unless you have specific edge deployment requirements.**

## 8. Summary & Recommendations

| Aspect | Recommendation |
|--------|---------------|
| **Model size** | 0.6B for edge/fast inference, 1.7B for best quality |
| **Finetuning method** | LoRA for <100h data, full SFT for >100h |
| **Framework** | Official `qwen-asr` + HF Trainer |
| **English datasets** | LibriSpeech + Common Voice + GigaSpeech |
| **German datasets** | Common Voice DE + MLS DE |
| **Deployment** | vLLM (server), ONNX (edge), MLX (Mac) |
| **GGUF** | Not supported; use ONNX or native inference instead |

## References
- Paper: arXiv:2601.21337
- Official repo: https://github.com/QwenLM/Qwen3-ASR
- Official finetune: https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning
- LoRA finetune: https://github.com/ysys12138/Qwen3-ASR-Lora-finetune
- Hebrew finetune example: https://huggingface.co/OzLabs/Caspi-1.7B
