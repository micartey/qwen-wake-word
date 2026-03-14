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
                uv pip install sounddevice numpy huggingface-hub torch torchaudio
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
              pkgs.ffmpeg
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
              pkgs.ffmpeg
            ];

            shellHook = ''
              if [ ! -d .venv-train ]; then
                echo "Creating training venv..."
                uv venv .venv-train
                source .venv-train/bin/activate
                uv pip install wheel
                uv pip install torchcodec
                uv pip intsall "datasets<4.0.0"
                uv pip install "torch>=2.6" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
                uv pip install qwen-asr peft librosa soundfile jiwer gguf safetensors tqdm
                uv pip install flash-attn --no-build-isolation || echo "flash-attn build failed (optional)"
              else
                source .venv-train/bin/activate
              fi

              echo "Make sure to export your hugging face access token: HF_TOKEN"
            '';
          };
        }
      );
    };
}
