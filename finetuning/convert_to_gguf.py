import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

import gguf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class Qwen3ASRToGGUF:
    TENSOR_MAP = {
        "thinker.audio_tower.conv2d1.weight": "audio.encoder.conv1.weight",
        "thinker.audio_tower.conv2d1.bias": "audio.encoder.conv1.bias",
        "thinker.audio_tower.conv2d2.weight": "audio.encoder.conv2.weight",
        "thinker.audio_tower.conv2d2.bias": "audio.encoder.conv2.bias",
        "thinker.audio_tower.conv2d3.weight": "audio.encoder.conv3.weight",
        "thinker.audio_tower.conv2d3.bias": "audio.encoder.conv3.bias",
        "thinker.audio_tower.conv_out.weight": "audio.encoder.conv_out.weight",
        "thinker.audio_tower.conv_out.bias": "audio.encoder.conv_out.bias",
        "thinker.audio_tower.layer_norm.weight": "audio.encoder.ln.weight",
        "thinker.audio_tower.layer_norm.bias": "audio.encoder.ln.bias",
        "thinker.audio_tower.ln_post.weight": "audio.encoder.ln_post.weight",
        "thinker.audio_tower.ln_post.bias": "audio.encoder.ln_post.bias",
        "thinker.audio_tower.embed_positions.weight": "audio.encoder.pos_embd.weight",
        "thinker.audio_tower.proj1.weight": "audio.encoder.proj1.weight",
        "thinker.audio_tower.proj1.bias": "audio.encoder.proj1.bias",
        "thinker.audio_tower.proj2.weight": "audio.encoder.proj2.weight",
        "thinker.audio_tower.proj2.bias": "audio.encoder.proj2.bias",
        "thinker.model.embed_tokens.weight": "token_embd.weight",
        "thinker.model.norm.weight": "output_norm.weight",
        "thinker.lm_head.weight": "output.weight",
    }

    AUDIO_LAYER_PATTERNS = [
        (
            r"thinker\.audio_tower\.layers\.(\d+)\.self_attn\.q_proj\.weight",
            "audio.encoder.blk.{}.attn_q.weight",
        ),
        (
            r"thinker\.audio_tower\.layers\.(\d+)\.self_attn\.k_proj\.weight",
            "audio.encoder.blk.{}.attn_k.weight",
        ),
        (
            r"thinker\.audio_tower\.layers\.(\d+)\.self_attn\.v_proj\.weight",
            "audio.encoder.blk.{}.attn_v.weight",
        ),
        (
            r"thinker\.audio_tower\.layers\.(\d+)\.self_attn\.out_proj\.weight",
            "audio.encoder.blk.{}.attn_out.weight",
        ),
        (
            r"thinker\.audio_tower\.layers\.(\d+)\.self_attn\.q_proj\.bias",
            "audio.encoder.blk.{}.attn_q.bias",
        ),
        (
            r"thinker\.audio_tower\.layers\.(\d+)\.self_attn\.k_proj\.bias",
            "audio.encoder.blk.{}.attn_k.bias",
        ),
        (
            r"thinker\.audio_tower\.layers\.(\d+)\.self_attn\.v_proj\.bias",
            "audio.encoder.blk.{}.attn_v.bias",
        ),
        (
            r"thinker\.audio_tower\.layers\.(\d+)\.self_attn\.out_proj\.bias",
            "audio.encoder.blk.{}.attn_out.bias",
        ),
        (
            r"thinker\.audio_tower\.layers\.(\d+)\.self_attn_layer_norm\.weight",
            "audio.encoder.blk.{}.attn_norm.weight",
        ),
        (
            r"thinker\.audio_tower\.layers\.(\d+)\.self_attn_layer_norm\.bias",
            "audio.encoder.blk.{}.attn_norm.bias",
        ),
        (
            r"thinker\.audio_tower\.layers\.(\d+)\.final_layer_norm\.weight",
            "audio.encoder.blk.{}.ffn_norm.weight",
        ),
        (
            r"thinker\.audio_tower\.layers\.(\d+)\.final_layer_norm\.bias",
            "audio.encoder.blk.{}.ffn_norm.bias",
        ),
        (
            r"thinker\.audio_tower\.layers\.(\d+)\.fc1\.weight",
            "audio.encoder.blk.{}.ffn_up.weight",
        ),
        (
            r"thinker\.audio_tower\.layers\.(\d+)\.fc1\.bias",
            "audio.encoder.blk.{}.ffn_up.bias",
        ),
        (
            r"thinker\.audio_tower\.layers\.(\d+)\.fc2\.weight",
            "audio.encoder.blk.{}.ffn_down.weight",
        ),
        (
            r"thinker\.audio_tower\.layers\.(\d+)\.fc2\.bias",
            "audio.encoder.blk.{}.ffn_down.bias",
        ),
    ]

    TEXT_LAYER_PATTERNS = [
        (
            r"thinker\.model\.layers\.(\d+)\.input_layernorm\.weight",
            "blk.{}.attn_norm.weight",
        ),
        (
            r"thinker\.model\.layers\.(\d+)\.self_attn\.q_proj\.weight",
            "blk.{}.attn_q.weight",
        ),
        (
            r"thinker\.model\.layers\.(\d+)\.self_attn\.k_proj\.weight",
            "blk.{}.attn_k.weight",
        ),
        (
            r"thinker\.model\.layers\.(\d+)\.self_attn\.v_proj\.weight",
            "blk.{}.attn_v.weight",
        ),
        (
            r"thinker\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight",
            "blk.{}.attn_output.weight",
        ),
        (
            r"thinker\.model\.layers\.(\d+)\.self_attn\.q_norm\.weight",
            "blk.{}.attn_q_norm.weight",
        ),
        (
            r"thinker\.model\.layers\.(\d+)\.self_attn\.k_norm\.weight",
            "blk.{}.attn_k_norm.weight",
        ),
        (
            r"thinker\.model\.layers\.(\d+)\.post_attention_layernorm\.weight",
            "blk.{}.ffn_norm.weight",
        ),
        (
            r"thinker\.model\.layers\.(\d+)\.mlp\.gate_proj\.weight",
            "blk.{}.ffn_gate.weight",
        ),
        (
            r"thinker\.model\.layers\.(\d+)\.mlp\.up_proj\.weight",
            "blk.{}.ffn_up.weight",
        ),
        (
            r"thinker\.model\.layers\.(\d+)\.mlp\.down_proj\.weight",
            "blk.{}.ffn_down.weight",
        ),
    ]

    def __init__(self, input_dir: Path, output_path: Path, output_type: str = "f16"):
        self.input_dir = input_dir
        self.output_path = output_path
        self.output_type = output_type
        self.config = self._load_config()
        self._extract_params()

    def _load_config(self) -> dict[str, Any]:
        config_path = self.input_dir / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_params(self) -> None:
        thinker_config = self.config.get("thinker_config", {})
        audio_config = thinker_config.get("audio_config", {})
        text_config = thinker_config.get("text_config", {})

        self.audio_encoder_layers = audio_config.get(
            "encoder_layers", audio_config.get("num_hidden_layers", 18)
        )
        self.audio_d_model = audio_config.get("d_model", 896)
        self.audio_attention_heads = audio_config.get("encoder_attention_heads", 14)
        self.audio_ffn_dim = audio_config.get("encoder_ffn_dim", 3584)
        self.audio_num_mel_bins = audio_config.get("num_mel_bins", 128)
        self.audio_downsample_hidden_size = audio_config.get(
            "downsample_hidden_size", 480
        )

        self.text_decoder_layers = text_config.get("num_hidden_layers", 28)
        self.text_hidden_size = text_config.get("hidden_size", 1024)
        self.text_attention_heads = text_config.get("num_attention_heads", 16)
        self.text_kv_heads = text_config.get("num_key_value_heads", 8)
        self.text_intermediate_size = text_config.get("intermediate_size", 3072)
        self.text_rope_theta = text_config.get("rope_theta", 1000000)
        self.text_rms_norm_eps = text_config.get("rms_norm_eps", 1e-6)
        self.text_head_dim = text_config.get("head_dim", 128)
        self.vocab_size = text_config.get("vocab_size", 151936)

        self.audio_start_token_id = thinker_config.get("audio_start_token_id", 151669)
        self.audio_end_token_id = thinker_config.get("audio_end_token_id", 151670)
        self.audio_pad_token_id = thinker_config.get("audio_token_id", 151676)

    def _map_tensor_name(self, hf_name: str) -> str | None:
        if hf_name in self.TENSOR_MAP:
            return self.TENSOR_MAP[hf_name]
        for pattern, template in self.AUDIO_LAYER_PATTERNS + self.TEXT_LAYER_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                return template.format(match.group(1))
        return None

    def _get_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        for sf_path in sorted(self.input_dir.glob("*.safetensors")):
            logger.info(f"Loading {sf_path.name}")
            with safe_open(sf_path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    yield name, f.get_tensor(name)

    def _should_quantize(self, tensor_name: str) -> bool:
        if any(x in tensor_name for x in ["token_embd", "output.weight", "pos_embd"]):
            return False
        if any(x in tensor_name for x in ["_norm", ".ln", "ln_post"]):
            return False
        if ".bias" in tensor_name:
            return False
        return True

    def _convert_dtype(
        self, tensor: torch.Tensor, tensor_name: str = ""
    ) -> tuple[np.ndarray, gguf.GGMLQuantizationType]:
        if tensor.dtype == torch.bfloat16:
            data = tensor.float().numpy()
        else:
            data = tensor.numpy()

        n_dims = len(data.shape)

        if n_dims == 4 and "conv" in tensor_name and "weight" in tensor_name:
            data = np.ascontiguousarray(data)

        if n_dims <= 1:
            return data.astype(np.float32), gguf.GGMLQuantizationType.F32

        if self.output_type == "f32":
            return data.astype(np.float32), gguf.GGMLQuantizationType.F32
        elif self.output_type == "f16":
            return data.astype(np.float16), gguf.GGMLQuantizationType.F16
        elif self.output_type == "q8_0":
            if not self._should_quantize(tensor_name):
                return data.astype(np.float16), gguf.GGMLQuantizationType.F16
            data = data.astype(np.float32)
            try:
                quantized = gguf.quants.quantize(data, gguf.GGMLQuantizationType.Q8_0)
                return quantized, gguf.GGMLQuantizationType.Q8_0
            except Exception as e:
                logger.warning(
                    f"Q8_0 failed for {tensor_name}: {e}, falling back to F16"
                )
                return data.astype(np.float16), gguf.GGMLQuantizationType.F16
        else:
            return data.astype(np.float16), gguf.GGMLQuantizationType.F16

    def _load_tokenizer(self) -> tuple[list[str], list[int], list[str]]:
        vocab_path = self.input_dir / "vocab.json"
        merges_path = self.input_dir / "merges.txt"

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)

        sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
        tokens = []
        toktypes = []

        for token, _ in sorted_vocab:
            tokens.append(token)
            if token.startswith("<|") and token.endswith("|>"):
                toktypes.append(gguf.TokenType.CONTROL)
            else:
                toktypes.append(gguf.TokenType.NORMAL)

        while len(tokens) < self.vocab_size:
            tokens.append(f"[PAD{len(tokens)}]")
            toktypes.append(gguf.TokenType.UNUSED)

        merges = []
        if merges_path.exists():
            with open(merges_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        merges.append(line)

        return tokens, toktypes, merges

    def convert(self) -> None:
        logger.info(
            f"Converting to GGUF: {self.input_dir} -> {self.output_path} ({self.output_type})"
        )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        arch = "qwen3-asr"
        writer = gguf.GGUFWriter(path=None, arch=arch)

        writer.add_name("Qwen3-ASR-0.6B-finetuned")
        writer.add_type(gguf.GGUFType.MODEL)

        ftype_map = {
            "f32": gguf.LlamaFileType.ALL_F32,
            "f16": gguf.LlamaFileType.MOSTLY_F16,
            "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        }
        writer.add_file_type(
            ftype_map.get(self.output_type, gguf.LlamaFileType.MOSTLY_F16)
        )
        writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

        writer.add_block_count(self.text_decoder_layers)
        writer.add_embedding_length(self.text_hidden_size)
        writer.add_feed_forward_length(self.text_intermediate_size)
        writer.add_head_count(self.text_attention_heads)
        writer.add_head_count_kv(self.text_kv_heads)
        writer.add_key_length(self.text_head_dim)
        writer.add_value_length(self.text_head_dim)
        writer.add_rope_freq_base(self.text_rope_theta)
        writer.add_layer_norm_rms_eps(self.text_rms_norm_eps)
        writer.add_vocab_size(self.vocab_size)

        writer.add_uint32(
            f"{arch}.audio.encoder.layer_count", self.audio_encoder_layers
        )
        writer.add_uint32(f"{arch}.audio.encoder.embedding_length", self.audio_d_model)
        writer.add_uint32(
            f"{arch}.audio.encoder.attention.head_count", self.audio_attention_heads
        )
        writer.add_uint32(
            f"{arch}.audio.encoder.feed_forward_length", self.audio_ffn_dim
        )
        writer.add_uint32(f"{arch}.audio.num_mel_bins", self.audio_num_mel_bins)
        writer.add_uint32(
            f"{arch}.audio.conv_channels", self.audio_downsample_hidden_size
        )
        writer.add_uint32(f"{arch}.audio.start_token_id", self.audio_start_token_id)
        writer.add_uint32(f"{arch}.audio.end_token_id", self.audio_end_token_id)
        writer.add_uint32(f"{arch}.audio.pad_token_id", self.audio_pad_token_id)

        tokens, toktypes, merges = self._load_tokenizer()
        writer.add_tokenizer_model("gpt2")
        writer.add_tokenizer_pre("qwen2")
        writer.add_token_list(tokens)
        writer.add_token_types(toktypes)
        if merges:
            writer.add_token_merges(merges)

        tokenizer_config_path = self.input_dir / "tokenizer_config.json"
        if tokenizer_config_path.exists():
            with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                tok_cfg = json.load(f)
            vocab_path = self.input_dir / "vocab.json"
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            eos_token = tok_cfg.get("eos_token")
            if isinstance(eos_token, dict):
                eos_token = eos_token.get("content")
            if eos_token and eos_token in vocab:
                writer.add_eos_token_id(vocab[eos_token])
            pad_token = tok_cfg.get("pad_token")
            if isinstance(pad_token, dict):
                pad_token = pad_token.get("content")
            if pad_token and pad_token in vocab:
                writer.add_pad_token_id(vocab[pad_token])

        tensor_count = 0
        skipped = 0
        for hf_name, tensor in tqdm(list(self._get_tensors()), desc="Converting"):
            ggml_name = self._map_tensor_name(hf_name)
            if ggml_name is None:
                logger.warning(f"Skipping: {hf_name}")
                skipped += 1
                continue
            data, dtype = self._convert_dtype(tensor, ggml_name)
            writer.add_tensor(ggml_name, data, raw_dtype=dtype)
            tensor_count += 1

        logger.info(f"Converted {tensor_count} tensors, skipped {skipped}")

        writer.write_header_to_file(path=self.output_path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
        logger.info(f"Done! GGUF saved to {self.output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert merged Qwen3-ASR HF model to GGUF"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to merged HF model directory (from merge_lora.py)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output GGUF file path"
    )
    parser.add_argument(
        "--type",
        "-t",
        choices=["f16", "f32", "q8_0"],
        default="f16",
        help="Output type (default: f16)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    converter = Qwen3ASRToGGUF(args.input, args.output, args.type)
    converter.convert()


if __name__ == "__main__":
    main()
