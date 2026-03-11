import argparse

import torch
from peft import PeftModel
from qwen_asr import Qwen3ASRModel


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument(
        "--lora_path", type=str, required=True, help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for merged model",
    )
    args = parser.parse_args()

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"Loading base model: {args.model_path}")
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args.model_path, dtype=dtype, device_map="cpu"
    )

    print(f"Loading LoRA adapter: {args.lora_path}")
    asr_wrapper.model = PeftModel.from_pretrained(asr_wrapper.model, args.lora_path)

    print("Merging LoRA weights...")
    asr_wrapper.model = asr_wrapper.model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}")
    asr_wrapper.model.save_pretrained(args.output_dir)
    asr_wrapper.processor.save_pretrained(args.output_dir)

    print("Done! Merged model saved.")
    print(f"\nTo use: Qwen3ASRModel.from_pretrained('{args.output_dir}')")


if __name__ == "__main__":
    main()
