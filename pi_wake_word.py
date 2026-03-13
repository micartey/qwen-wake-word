import os

# These MUST remain at the very top to throttle backend C++ libraries before initialization
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import torch

torch.set_num_threads(2)

import queue
import sys

import numpy as np
import sounddevice as sd
from huggingface_hub import hf_hub_download
from py_qwen3_asr_cpp.model import Qwen3ASRModel

HF_MODEL = "micartey/qwen3-asr-0.6b-english"
HF_FILENAME = "./qwen3-asr-0.6b-finetuned-q5_k.gguf"

SAMPLE_RATE = 16000
CHUNK_SECONDS = 1.5
OVERLAP_SECONDS = 0.75
N_THREADS = 2  # Reduced to prevent 100% CPU lockups on Pi 5
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


def load_asr_model():
    path = resolve_model()
    print(f"Loading {HF_MODEL} on 2 threads...")
    model = Qwen3ASRModel(
        asr_model=path,
        n_threads=N_THREADS,
    )
    print("ASR Model loaded.")
    return model


def load_vad_model():
    print("Loading Silero VAD...")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    print("VAD Model loaded.")
    # utils[0] is the get_speech_timestamps function
    return model, utils[0]


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
    vad_model, get_speech_timestamps = load_vad_model()
    asr_model = load_asr_model()

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

                # 1. Gatekeeper: Check for actual human speech first
                audio_tensor = torch.from_numpy(audio_segment)

                speech_timestamps = get_speech_timestamps(
                    audio_tensor, vad_model, sampling_rate=SAMPLE_RATE
                )

                if not speech_timestamps:
                    continue

                print("Speech detected, running ASR...")

                # 2. Execution: Only runs if human speech is detected
                result = asr_model.transcribe(audio_segment)
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
