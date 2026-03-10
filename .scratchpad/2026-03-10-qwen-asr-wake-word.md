# Qwen ASR Wake Word Detection

## Model Choice
- **Qwen3-ASR-0.6B Q4_K_M** (GGUF quantized) — 685MB on disk
- Uses `py-qwen3-asr-cpp` (C++ backend via pybind11)
- HuggingFace repo: `OpenVoiceOS/qwen3-asr-0.6b-q4-k-m`
- Model name for py-qwen3-asr-cpp: `qwen3-asr-0.6b-q4-k-m`
- API: `Qwen3ASRModel(asr_model=...)` → `.transcribe(np_array)`
- Accepts numpy float32 arrays directly
- Auto-downloads GGUF from HuggingFace on first use

## Why GGUF over PyTorch
- No PyTorch dependency (~2GB saved)
- Q4 quantization: ~685MB vs ~2.4GB float32
- C++ inference: ~2-3x faster than Python/PyTorch on ARM
- Expected: ~1-2s per 2s chunk on Pi 4

## Pi 4 Constraints
- 4GB RAM → Q4 model ~685MB, plenty of headroom
- No CUDA → CPU with 4 threads (Cortex-A72)
- Build requires cmake + gcc (for pybind11 C++ compilation)

## Architecture
- Streaming mic capture via `sounddevice` (portaudio backend)
- 2-second chunks with 0.5-second overlap
- Silence gating to skip empty audio (energy < 0.005)
- Simple string matching on transcription output for wake word
