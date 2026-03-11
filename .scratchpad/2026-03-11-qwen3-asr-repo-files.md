# Qwen3-ASR Repository Files

Fetched on 2026-03-11 from https://github.com/QwenLM/Qwen3-ASR

## Repo Structure

```
.github/
.gitignore
LICENSE
MANIFEST.in
README.md (60KB - main readme, very large)
assets/
docker/
  Dockerfile-qwen3-asr-cu128
examples/
  example_qwen3_asr_transformers.py
  example_qwen3_asr_vllm.py
  example_qwen3_asr_vllm_streaming.py
  example_qwen3_forced_aligner.py
finetuning/
  README.md
  qwen3_asr_sft.py
pyproject.toml
qwen_asr/
  __init__.py
  __main__.py
  cli/
    demo.py
    demo_streaming.py
    serve.py
  core/
    transformers_backend/
    vllm_backend/
  inference/
    assets/
    qwen3_asr.py
    qwen3_forced_aligner.py
    utils.py
```

No requirements.txt - uses pyproject.toml instead.
