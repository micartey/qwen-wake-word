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
              pkgs.portaudio
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
              echo "Run: python app.py"
            '';
          };
        }
      );
    };
}
