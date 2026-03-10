import argparse
import json
import re

import torch
from jiwer import wer, cer
from qwen_asr import Qwen3ASRModel


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_eval_data(path: str) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entries.append(entry)
    return entries


def extract_transcript(text_field: str) -> str:
    if "<asr_text>" in text_field:
        return text_field.split("<asr_text>", 1)[1]
    return text_field


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-ASR model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to finetuned model or base model (e.g. Qwen/Qwen3-ASR-0.6B)",
    )
    parser.add_argument(
        "--lora_path", type=str, default="", help="Path to LoRA adapter (if using LoRA)"
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        required=True,
        help="JSONL eval file with audio/text fields",
    )
    parser.add_argument(
        "--max_samples", type=int, default=0, help="Max samples to evaluate (0 = all)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="Write detailed results to this file",
    )
    args = parser.parse_args()

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"Loading model: {args.model_path}")
    model = Qwen3ASRModel.from_pretrained(
        args.model_path, dtype=dtype, device_map="auto"
    )

    if args.lora_path:
        from peft import PeftModel

        print(f"Loading LoRA adapter: {args.lora_path}")
        model.model = PeftModel.from_pretrained(model.model, args.lora_path)

    entries = load_eval_data(args.eval_file)
    if args.max_samples > 0:
        entries = entries[: args.max_samples]

    print(f"Evaluating {len(entries)} samples...")

    references = []
    hypotheses = []
    results = []

    for i, entry in enumerate(entries):
        audio_path = entry["audio"]
        ref_text = normalize_text(extract_transcript(entry["text"]))

        try:
            result = model.transcribe(audio=audio_path)
            hyp_text = normalize_text(result[0].text) if result else ""
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            hyp_text = ""

        references.append(ref_text)
        hypotheses.append(hyp_text)
        results.append(
            {
                "audio": audio_path,
                "reference": ref_text,
                "hypothesis": hyp_text,
            }
        )

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(entries)}")

    valid = [(r, h) for r, h in zip(references, hypotheses) if r]
    if not valid:
        print("No valid samples found!")
        return

    refs, hyps = zip(*valid)
    word_error = wer(list(refs), list(hyps))
    char_error = cer(list(refs), list(hyps))

    print(f"\n{'=' * 50}")
    print(f"Results ({len(valid)} samples):")
    print(f"  WER: {word_error:.4f} ({word_error * 100:.2f}%)")
    print(f"  CER: {char_error:.4f} ({char_error * 100:.2f}%)")
    print(f"{'=' * 50}")

    en_refs, en_hyps = [], []
    de_refs, de_hyps = [], []
    for entry, ref, hyp in zip(entries, references, hypotheses):
        if not ref:
            continue
        text_field = entry["text"]
        if "language English" in text_field:
            en_refs.append(ref)
            en_hyps.append(hyp)
        elif "language German" in text_field:
            de_refs.append(ref)
            de_hyps.append(hyp)

    if en_refs:
        en_wer = wer(en_refs, en_hyps)
        print(
            f"  English WER: {en_wer:.4f} ({en_wer * 100:.2f}%) [{len(en_refs)} samples]"
        )
    if de_refs:
        de_wer = wer(de_refs, de_hyps)
        print(
            f"  German WER:  {de_wer:.4f} ({de_wer * 100:.2f}%) [{len(de_refs)} samples]"
        )

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(
                {
                    "wer": word_error,
                    "cer": char_error,
                    "n_samples": len(valid),
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nDetailed results written to: {args.output_file}")


if __name__ == "__main__":
    main()
