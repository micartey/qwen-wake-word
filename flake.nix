{
  description = "Qwen3-ASR wake word detection for Raspberry Pi";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python312;
        in
        {
          default = pkgs.mkShell {
            packages = [
              python
              pkgs.uv
              pkgs.cmake
              pkgs.gcc
              pkgs.gnumake
              pkgs.git
            ];

            buildInputs = [
              pkgs.portaudio # sudo apt install portaudio19-dev
              pkgs.ffmpeg
            ];

            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.portaudio
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
            ];

            shellHook = ''
              if [ ! -d .venv ]; then
                echo "Creating venv and installing dependencies..."
                uv venv .venv
                source .venv/bin/activate
                uv pip install sounddevice numpy huggingface-hub
                uv pip install "py-qwen3-asr-cpp @ git+https://github.com/femelo/py-qwen3-asr-cpp"
              else
                source .venv/bin/activate
              fi
              echo "Qwen3-ASR wake word detection environment"
              echo "Run: python wake_word.py"
            '';
          };

          train = pkgs.mkShell {
            packages = [
              python
              pkgs.uv
              pkgs.cmake
              pkgs.gcc
              pkgs.gnumake
              pkgs.git
            ];

            buildInputs = [
              pkgs.portaudio
              pkgs.ffmpeg
              pkgs.libsndfile
            ];

            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.portaudio
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
              pkgs.libsndfile
            ];

            shellHook = ''
              if [ ! -d .venv-train ]; then
                echo "Creating training venv..."
                uv venv .venv-train
                source .venv-train/bin/activate
                uv pip install wheel
                uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
                uv pip install qwen-asr datasets peft librosa soundfile jiwer gguf safetensors tqdm
                uv pip install flash-attn --no-build-isolation || echo "flash-attn build failed (optional)"
              else
                source .venv-train/bin/activate
              fi
              echo "Qwen3-ASR finetuning environment (GPU required)"
              echo ""
              echo "Steps:"
              echo "  1. Prepare data:   python finetuning/prepare_data.py --output_dir ./data"
              echo "  2a. Full SFT:      python finetuning/train_full.py --train_file ./data/train.jsonl --eval_file ./data/eval.jsonl"
              echo "  2b. LoRA (low GPU): python finetuning/train_lora.py --train_file ./data/train.jsonl --eval_file ./data/eval.jsonl"
              echo "  3. Evaluate:       python finetuning/evaluate.py --model_path ./output/final_model --eval_file ./data/eval.jsonl"
              echo "  4. Convert GGUF:   python finetuning/convert_to_gguf.py -i ./output/final_model -o ./qwen3-asr-0.6b-finetuned-f16.gguf -t f16"
            '';
          };
        }
      );
    };
}
