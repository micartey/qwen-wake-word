import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

import gguf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

QK_K = 256
K_SCALE_SIZE = 12

SKIP_QUANTIZE = [
    "token_embd",
    "output.weight",
    "pos_embd",
    "_norm",
    ".ln",
    "ln_post",
    ".bias",
]


def should_quantize(name: str) -> bool:
    return not any(s in name for s in SKIP_QUANTIZE)


def _make_qkx2_quants(data: np.ndarray, nmax: int):
    n_blocks = data.shape[0]

    xmin = data.min(axis=-1, keepdims=True)
    xmax = data.max(axis=-1, keepdims=True)

    xmin = np.minimum(xmin, 0.0)
    xmax = np.maximum(xmax, 0.0)

    diff = xmax - xmin
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_diff = np.where(diff == 0, 0.0, nmax / diff)

    quants = np.round((data - xmin) * inv_diff).astype(np.uint8)
    quants = np.clip(quants, 0, nmax)

    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(diff == 0, 0.0, diff / nmax)
    mn = xmin

    return scale.reshape(n_blocks), mn.reshape(n_blocks), quants


def _pack_scales_mins_q4k(scales: np.ndarray, mins: np.ndarray) -> np.ndarray:
    n_blocks = scales.shape[0]
    result = np.zeros((n_blocks, K_SCALE_SIZE), dtype=np.uint8)

    sc = np.clip(np.round(scales), 0, 63).astype(np.uint8)
    mn = np.clip(np.round(mins), 0, 63).astype(np.uint8)

    result[:, 0] = (sc[:, 0] & 0x3F) | ((sc[:, 4] & 0x30) << 2)
    result[:, 1] = (sc[:, 1] & 0x3F) | ((sc[:, 5] & 0x30) << 2)
    result[:, 2] = (sc[:, 2] & 0x3F) | ((sc[:, 6] & 0x30) << 2)
    result[:, 3] = (sc[:, 3] & 0x3F) | ((sc[:, 7] & 0x30) << 2)
    result[:, 4] = (mn[:, 0] & 0x3F) | ((mn[:, 4] & 0x30) << 2)
    result[:, 5] = (mn[:, 1] & 0x3F) | ((mn[:, 5] & 0x30) << 2)
    result[:, 6] = (mn[:, 2] & 0x3F) | ((mn[:, 6] & 0x30) << 2)
    result[:, 7] = (mn[:, 3] & 0x3F) | ((mn[:, 7] & 0x30) << 2)
    result[:, 8] = (sc[:, 4] & 0x0F) | ((mn[:, 4] & 0x0F) << 4)
    result[:, 9] = (sc[:, 5] & 0x0F) | ((mn[:, 5] & 0x0F) << 4)
    result[:, 10] = (sc[:, 6] & 0x0F) | ((mn[:, 6] & 0x0F) << 4)
    result[:, 11] = (sc[:, 7] & 0x0F) | ((mn[:, 7] & 0x0F) << 4)

    return result


def quantize_q4k_blocks(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    sub_blocks = blocks.reshape(n_blocks, 8, 32)

    all_scales = np.zeros((n_blocks, 8), dtype=np.float32)
    all_mins = np.zeros((n_blocks, 8), dtype=np.float32)
    all_quants = np.zeros((n_blocks, 8, 32), dtype=np.uint8)

    for j in range(8):
        sb = sub_blocks[:, j, :]
        scale, mn, quants = _make_qkx2_quants(sb, 15)
        all_scales[:, j] = scale
        all_mins[:, j] = -mn
        all_quants[:, j, :] = quants

    max_scale = all_scales.max(axis=-1, keepdims=True)
    max_min = all_mins.max(axis=-1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_max_scale = np.where(max_scale == 0, 0.0, 63.0 / max_scale)
        inv_max_min = np.where(max_min == 0, 0.0, 63.0 / max_min)

    d = max_scale / 63.0
    dmin = max_min / 63.0

    int_scales = np.round(all_scales * inv_max_scale).astype(np.uint8)
    int_mins = np.round(all_mins * inv_max_min).astype(np.uint8)

    d_f16 = d.reshape(n_blocks).astype(np.float16)
    dmin_f16 = dmin.reshape(n_blocks).astype(np.float16)
    eff_d = d_f16.astype(np.float32).reshape(n_blocks, 1) * int_scales.astype(
        np.float32
    )
    eff_dmin = dmin_f16.astype(np.float32).reshape(n_blocks, 1) * int_mins.astype(
        np.float32
    )

    for j in range(8):
        sc = eff_d[:, j : j + 1]
        mn = eff_dmin[:, j : j + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_sc = np.where(sc == 0, 0.0, 1.0 / sc)
        all_quants[:, j, :] = np.clip(
            np.round((sub_blocks[:, j, :] + mn) * inv_sc), 0, 15
        ).astype(np.uint8)

    packed_scales = _pack_scales_mins_q4k(int_scales, int_mins)

    qs_flat = np.zeros((n_blocks, QK_K // 2), dtype=np.uint8)
    for j in range(4):
        lo = all_quants[:, 2 * j, :] & 0x0F
        hi = all_quants[:, 2 * j + 1, :] & 0x0F
        qs_flat[:, j * 32 : (j + 1) * 32] = lo | (hi << 4)

    d_bytes = d_f16.view(np.uint8).reshape(n_blocks, 2)
    dmin_bytes = dmin_f16.view(np.uint8).reshape(n_blocks, 2)

    return np.concatenate([d_bytes, dmin_bytes, packed_scales, qs_flat], axis=-1)


def quantize_q5k_blocks(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    sub_blocks = blocks.reshape(n_blocks, 8, 32)

    all_scales = np.zeros((n_blocks, 8), dtype=np.float32)
    all_mins = np.zeros((n_blocks, 8), dtype=np.float32)
    all_quants = np.zeros((n_blocks, 8, 32), dtype=np.uint8)

    for j in range(8):
        sb = sub_blocks[:, j, :]
        scale, mn, quants = _make_qkx2_quants(sb, 31)
        all_scales[:, j] = scale
        all_mins[:, j] = -mn
        all_quants[:, j, :] = quants

    max_scale = all_scales.max(axis=-1, keepdims=True)
    max_min = all_mins.max(axis=-1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_max_scale = np.where(max_scale == 0, 0.0, 63.0 / max_scale)
        inv_max_min = np.where(max_min == 0, 0.0, 63.0 / max_min)

    d = max_scale / 63.0
    dmin = max_min / 63.0

    int_scales = np.round(all_scales * inv_max_scale).astype(np.uint8)
    int_mins = np.round(all_mins * inv_max_min).astype(np.uint8)

    d_f16 = d.reshape(n_blocks).astype(np.float16)
    dmin_f16 = dmin.reshape(n_blocks).astype(np.float16)
    eff_d = d_f16.astype(np.float32).reshape(n_blocks, 1) * int_scales.astype(
        np.float32
    )
    eff_dmin = dmin_f16.astype(np.float32).reshape(n_blocks, 1) * int_mins.astype(
        np.float32
    )

    for j in range(8):
        sc = eff_d[:, j : j + 1]
        mn = eff_dmin[:, j : j + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_sc = np.where(sc == 0, 0.0, 1.0 / sc)
        all_quants[:, j, :] = np.clip(
            np.round((sub_blocks[:, j, :] + mn) * inv_sc), 0, 31
        ).astype(np.uint8)

    packed_scales = _pack_scales_mins_q4k(int_scales, int_mins)

    ql = (all_quants & 0x0F).reshape(n_blocks, QK_K)
    qh_bits = ((all_quants >> 4) & 0x01).reshape(n_blocks, QK_K)

    qh = np.zeros((n_blocks, QK_K // 8), dtype=np.uint8)
    for i in range(8):
        qh |= (
            (qh_bits[:, i * 32 : (i + 1) * 32].reshape(n_blocks, 32) << i)
            .astype(np.uint8)
            .reshape(n_blocks, 32)
        )

    qs_packed = np.zeros((n_blocks, QK_K // 2), dtype=np.uint8)
    ql_8x32 = ql.reshape(n_blocks, 8, 32)
    for j in range(4):
        lo = ql_8x32[:, 2 * j, :] & 0x0F
        hi = ql_8x32[:, 2 * j + 1, :] & 0x0F
        qs_packed[:, j * 32 : (j + 1) * 32] = lo | (hi << 4)

    d_bytes = d_f16.view(np.uint8).reshape(n_blocks, 2)
    dmin_bytes = dmin_f16.view(np.uint8).reshape(n_blocks, 2)

    return np.concatenate([d_bytes, dmin_bytes, packed_scales, qh, qs_packed], axis=-1)


def quantize_tensor(
    float_data: np.ndarray, target_qtype
) -> tuple[np.ndarray, "gguf.GGMLQuantizationType"]:
    block_size = gguf.quants.GGML_QUANT_SIZES[target_qtype][0]

    if float_data.shape[-1] % block_size != 0:
        fallback_qtype = gguf.GGMLQuantizationType.Q8_0
        fallback_block = gguf.quants.GGML_QUANT_SIZES[fallback_qtype][0]
        if float_data.shape[-1] % fallback_block == 0:
            target_qtype = fallback_qtype
        else:
            return float_data, None

    if target_qtype in (gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q5_K):
        rows = float_data.reshape((-1, float_data.shape[-1]))
        n_blocks_per_row = rows.shape[-1] // QK_K
        all_blocks = rows.reshape(-1, QK_K)

        if target_qtype == gguf.GGMLQuantizationType.Q4_K:
            quantized_blocks = quantize_q4k_blocks(all_blocks)
        else:
            quantized_blocks = quantize_q5k_blocks(all_blocks)

        type_size = gguf.quants.GGML_QUANT_SIZES[target_qtype][1]
        out_shape = (*float_data.shape[:-1], n_blocks_per_row * type_size)
        return quantized_blocks.reshape(out_shape), target_qtype

    quantized = gguf.quants.quantize(float_data, target_qtype)
    return quantized, target_qtype


def main():
    parser = argparse.ArgumentParser(description="Quantize a Qwen3-ASR f16 GGUF file")
    parser.add_argument("input", type=Path, help="Input GGUF file (f16 or f32)")
    parser.add_argument("output", type=Path, help="Output quantized GGUF file")
    parser.add_argument(
        "--type",
        "-t",
        choices=["q8_0", "q5_k", "q4_k"],
        default="q5_k",
        help="Quantization type (default: q5_k)",
    )
    args = parser.parse_args()

    qtype_map = {
        "q8_0": gguf.GGMLQuantizationType.Q8_0,
        "q5_k": gguf.GGMLQuantizationType.Q5_K,
        "q4_k": gguf.GGMLQuantizationType.Q4_K,
    }
    target_qtype = qtype_map[args.type]

    logger.info(f"Reading {args.input}")
    reader = gguf.GGUFReader(str(args.input))

    writer = gguf.GGUFWriter(path=None, arch="qwen3-asr")

    logger.info("Copying metadata...")
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        if key == "general.file_type":
            continue

        types = field.types
        parts = field.parts

        if not types:
            continue

        vtype = types[0]

        if vtype == gguf.GGUFValueType.STRING:
            if len(types) == 1:
                val = str(bytes(parts[-1]), "utf-8")
                writer.add_key_value(key, val, gguf.GGUFValueType.STRING)
        elif vtype == gguf.GGUFValueType.UINT32:
            val = int(np.frombuffer(bytes(parts[-1]), dtype=np.uint32)[0])
            writer.add_key_value(key, val, gguf.GGUFValueType.UINT32)
        elif vtype == gguf.GGUFValueType.INT32:
            val = int(np.frombuffer(bytes(parts[-1]), dtype=np.int32)[0])
            writer.add_key_value(key, val, gguf.GGUFValueType.INT32)
        elif vtype == gguf.GGUFValueType.FLOAT32:
            val = float(np.frombuffer(bytes(parts[-1]), dtype=np.float32)[0])
            writer.add_key_value(key, val, gguf.GGUFValueType.FLOAT32)
        elif vtype == gguf.GGUFValueType.BOOL:
            val = bool(parts[-1][0])
            writer.add_key_value(key, val, gguf.GGUFValueType.BOOL)
        elif vtype == gguf.GGUFValueType.ARRAY:
            if key == "tokenizer.ggml.tokens":
                tokens = []
                for idx in field.data:
                    tokens.append(bytes(parts[idx]).decode("utf-8", errors="replace"))
                writer.add_token_list(tokens)
            elif key == "tokenizer.ggml.token_type":
                toktypes = [int(parts[idx][0]) for idx in field.data]
                writer.add_token_types(toktypes)
            elif key == "tokenizer.ggml.merges":
                merges = []
                for idx in field.data:
                    merges.append(bytes(parts[idx]).decode("utf-8", errors="replace"))
                writer.add_token_merges(merges)

    ftype_map = {
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "q5_k": gguf.LlamaFileType.MOSTLY_Q5_K_M,
        "q4_k": gguf.LlamaFileType.MOSTLY_Q4_K_M,
    }
    writer.add_file_type(ftype_map[args.type])

    logger.info(f"Quantizing tensors to {args.type}...")
    quantized_count = 0
    kept_count = 0
    fallback_count = 0

    for tensor in tqdm(reader.tensors, desc="Quantizing"):
        name = tensor.name
        data = tensor.data
        shape = tensor.shape
        n_dims = len(shape)
        src_type = tensor.tensor_type

        if n_dims <= 1 or not should_quantize(name):
            writer.add_tensor(name, data, raw_dtype=src_type)
            kept_count += 1
            continue

        if src_type in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
            float_data = data.astype(np.float32)
        else:
            writer.add_tensor(name, data, raw_dtype=src_type)
            kept_count += 1
            continue

        try:
            quantized, actual_qtype = quantize_tensor(float_data, target_qtype)
            if actual_qtype is None:
                writer.add_tensor(name, data, raw_dtype=src_type)
                kept_count += 1
            else:
                writer.add_tensor(name, quantized, raw_dtype=actual_qtype)
                if actual_qtype != target_qtype:
                    fallback_count += 1
                quantized_count += 1
        except Exception as e:
            logger.warning(
                f"Failed to quantize {name}: {type(e).__name__}: {e}, keeping original"
            )
            writer.add_tensor(name, data, raw_dtype=src_type)
            kept_count += 1

    logger.info(
        f"Quantized {quantized_count} tensors ({fallback_count} fell back to Q8_0), "
        f"kept {kept_count} as-is"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer.write_header_to_file(path=args.output)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    in_size = args.input.stat().st_size / (1024**3)
    out_size = args.output.stat().st_size / (1024**3)
    reduction = (1 - out_size / in_size) * 100
    logger.info(
        f"Done! {in_size:.2f} GB -> {out_size:.2f} GB ({reduction:.1f}% reduction)"
    )


if __name__ == "__main__":
    main()
