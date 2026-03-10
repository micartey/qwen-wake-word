import argparse
import json
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import load_dataset

SAMPLE_RATE = 16000


def process_librispeech(split: str, output_dir: Path, max_hours: float | None = None):
    jsonl_cache = output_dir / f".cache_librispeech_{split.replace('.', '_')}.jsonl"
    if jsonl_cache.exists():
        print(f"  LibriSpeech {split}: using cached {jsonl_cache}")
        with open(jsonl_cache) as f:
            return [json.loads(line) for line in f if line.strip()]

    ds = load_dataset("openslr/librispeech_asr", split=split)

    wav_dir = output_dir / "wavs" / "librispeech" / split
    wav_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    total_seconds = 0.0
    max_seconds = max_hours * 3600 if max_hours else float("inf")

    for i, row in enumerate(ds):
        if total_seconds >= max_seconds:
            break

        audio = row["audio"]
        arr = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]

        if sr != SAMPLE_RATE:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=SAMPLE_RATE)

        duration = len(arr) / SAMPLE_RATE
        if duration < 0.5 or duration > 30.0:
            continue

        wav_path = wav_dir / f"{i:08d}.wav"
        if not wav_path.exists():
            sf.write(str(wav_path), arr, SAMPLE_RATE)

        text = row["text"].strip()
        entries.append(
            {
                "audio": str(wav_path.resolve()),
                "text": f"language English<asr_text>{text}",
            }
        )
        total_seconds += duration

        if (i + 1) % 1000 == 0:
            print(f"  LibriSpeech {split}: {i + 1} files, {total_seconds / 3600:.1f}h")

    print(
        f"  LibriSpeech {split}: {len(entries)} files, {total_seconds / 3600:.1f}h total"
    )

    with open(jsonl_cache, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entries


def process_common_voice(
    lang: str, split: str, output_dir: Path, max_hours: float | None = None
):
    jsonl_cache = output_dir / f".cache_cv_{lang}_{split}.jsonl"
    if jsonl_cache.exists():
        print(f"  Common Voice {lang} {split}: using cached {jsonl_cache}")
        with open(jsonl_cache) as f:
            return [json.loads(line) for line in f if line.strip()]

    lang_map = {"en": "en", "de": "de"}
    cv_lang = lang_map.get(lang, lang)

    ds = load_dataset(
        "fsicoli/common_voice_16_0", cv_lang, split=split, trust_remote_code=True
    )

    wav_dir = output_dir / "wavs" / f"common_voice_{lang}" / split
    wav_dir.mkdir(parents=True, exist_ok=True)

    lang_name = {"en": "English", "de": "German"}.get(lang, lang)

    entries = []
    total_seconds = 0.0
    max_seconds = max_hours * 3600 if max_hours else float("inf")

    for i, row in enumerate(ds):
        if total_seconds >= max_seconds:
            break

        audio = row["audio"]
        arr = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]

        if sr != SAMPLE_RATE:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=SAMPLE_RATE)

        duration = len(arr) / SAMPLE_RATE
        if duration < 0.5 or duration > 30.0:
            continue

        wav_path = wav_dir / f"{i:08d}.wav"
        if not wav_path.exists():
            sf.write(str(wav_path), arr, SAMPLE_RATE)

        text = row["sentence"].strip()
        if not text:
            continue

        entries.append(
            {
                "audio": str(wav_path.resolve()),
                "text": f"language {lang_name}<asr_text>{text}",
            }
        )
        total_seconds += duration

        if (i + 1) % 1000 == 0:
            print(
                f"  Common Voice {lang} {split}: {i + 1} files, {total_seconds / 3600:.1f}h"
            )

    print(
        f"  Common Voice {lang} {split}: {len(entries)} files, {total_seconds / 3600:.1f}h total"
    )

    with open(jsonl_cache, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entries


def write_jsonl(entries: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {len(entries)} entries to {path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare ASR finetuning data")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for WAV files and JSONL",
    )
    parser.add_argument(
        "--librispeech_hours",
        type=float,
        default=100.0,
        help="Max hours of LibriSpeech to use (default: 100h from train-clean-100)",
    )
    parser.add_argument(
        "--cv_en_hours",
        type=float,
        default=50.0,
        help="Max hours of Common Voice English (default: 50h)",
    )
    parser.add_argument(
        "--cv_de_hours",
        type=float,
        default=0.0,
        help="Max hours of Common Voice German (default: 0, set >0 to include)",
    )
    parser.add_argument(
        "--eval_hours",
        type=float,
        default=2.0,
        help="Max hours for eval set per source (default: 2h)",
    )
    parser.add_argument(
        "--librispeech_split",
        type=str,
        default="train.clean.100",
        help="LibriSpeech train split (train.clean.100, train.clean.360, train.other.500)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_entries = []
    eval_entries = []

    print("=== Processing LibriSpeech (English) ===")
    train_entries.extend(
        process_librispeech(
            args.librispeech_split, output_dir, max_hours=args.librispeech_hours
        )
    )
    eval_entries.extend(
        process_librispeech("test.clean", output_dir, max_hours=args.eval_hours)
    )

    if args.cv_en_hours > 0:
        print("\n=== Processing Common Voice English ===")
        train_entries.extend(
            process_common_voice("en", "train", output_dir, max_hours=args.cv_en_hours)
        )
        eval_entries.extend(
            process_common_voice("en", "test", output_dir, max_hours=args.eval_hours)
        )

    if args.cv_de_hours > 0:
        print("\n=== Processing Common Voice German ===")
        train_entries.extend(
            process_common_voice("de", "train", output_dir, max_hours=args.cv_de_hours)
        )
        eval_entries.extend(
            process_common_voice("de", "test", output_dir, max_hours=args.eval_hours)
        )

    import random

    random.seed(42)
    random.shuffle(train_entries)
    random.shuffle(eval_entries)

    write_jsonl(train_entries, output_dir / "train.jsonl")
    write_jsonl(eval_entries, output_dir / "eval.jsonl")

    print(f"\nDone! Train: {len(train_entries)}, Eval: {len(eval_entries)}")
    print(f"Output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
