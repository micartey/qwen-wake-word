import os
import queue
import sys

import numpy as np
import sounddevice as sd
from huggingface_hub import hf_hub_download
from py_qwen3_asr_cpp.model import Qwen3ASRModel

# Available GGUF models (name -> (HuggingFace repo, filename, size)):
#
# --- 0.6B models (built-in, auto-download via py-qwen3-asr-cpp) ---
# "qwen3-asr-0.6b-q4-k-m"   ~685 MB   (Pi 4 4GB OK)
# "qwen3-asr-0.6b-q5-k-m"   ~800 MB   (Pi 4 4GB OK)
# "qwen3-asr-0.6b-q8-0"     ~1.35 GB  (Pi 4 4GB OK)
# "qwen3-asr-0.6b-f16"      ~1.88 GB  (Pi 4 4GB tight)
#
# --- Finetuned (English-optimized, download from HuggingFace) ---
# "micartey/qwen3-asr-0.6b-english"  ~1.88 GB  (Pi 4 4GB tight)
#
# --- 1.7B models (manual download, pass local path) ---
# "qwen3-asr-1.7b-q8-0"     ~3.2 GB   (Pi 5 8GB OK)     FlippyDora/qwen3-asr-1.7b-GGUF
# "qwen3-asr-1.7b-f16"      ~4.71 GB  (Pi 5 8GB tight)  FlippyDora/qwen3-asr-1.7b-GGUF
HF_MODEL = "micartey/qwen3-asr-0.6b-english"
HF_FILENAME = (
    "./qwen3-asr-0.6b-finetuned-q5_k.gguf"  # "qwen3-asr-0.6b-finetuned-q8_0.gguf"
)

SAMPLE_RATE = 16000
CHUNK_SECONDS = 1.5
OVERLAP_SECONDS = 0.75
SILENCE_THRESHOLD = 0.005
N_THREADS = 4
WAKE_WORD = "sarah"
MAX_DISTANCE = 2

audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata[:, 0].copy())


def resolve_model():
    if os.path.isfile(HF_FILENAME):
        return HF_FILENAME

    local_path = hf_hub_download(repo_id=HF_MODEL, filename=HF_FILENAME)

    print(f"Model path: {local_path}")
    return local_path


def load_model():
    path = resolve_model()
    print(f"Loading {HF_MODEL}...")
    model = Qwen3ASRModel(
        asr_model=path,
        n_threads=N_THREADS,
    )
    print("Model loaded.")
    return model


def levenshtein(a, b):
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n
    prev = list(range(n + 1))
    for j in range(1, m + 1):
        curr = [j] + [0] * n
        for i in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[i] = min(curr[i - 1] + 1, prev[i] + 1, prev[i - 1] + cost)
        prev = curr
    return prev[n]


def check_wake_word(text):
    for word in text.lower().split():
        if levenshtein(word, WAKE_WORD) <= MAX_DISTANCE:
            return True
    return False


def main():
    model = load_model()

    chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)
    overlap_samples = int(OVERLAP_SECONDS * SAMPLE_RATE)
    buffer = np.zeros(0, dtype=np.float32)

    print(f"Listening for wake word: '{WAKE_WORD}'")
    print("Speak into the microphone... (Ctrl+C to quit)")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * 0.5),
        callback=audio_callback,
    ):
        try:
            while True:
                chunk = audio_queue.get()
                buffer = np.concatenate([buffer, chunk])

                if len(buffer) < chunk_samples:
                    continue

                audio_segment = buffer[:chunk_samples]
                buffer = buffer[chunk_samples - overlap_samples :]

                energy = np.sqrt(np.mean(audio_segment**2))
                print(energy)
                if energy < SILENCE_THRESHOLD:
                    continue

                result = model.transcribe(audio_segment)
                if not result:
                    continue

                text = result.text.strip()
                if not text:
                    continue

                print(f"  [{text}]")

                if check_wake_word(text):
                    print(">>> Hello World! <<<")

        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
