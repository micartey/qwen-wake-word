import os

# These MUST remain at the very top to throttle backend C++ libraries before initialization
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import torch

torch.set_num_threads(4)

import argparse
import queue
import signal
import string
import subprocess
import sys
import time

import numpy as np
import sounddevice as sd
from huggingface_hub import hf_hub_download
from py_qwen3_asr_cpp.model import Qwen3ASRModel

HF_MODEL = "micartey/qwen3-asr-0.6b-english-v2"
HF_FILENAME = "./qwen3-asr-0.6b-finetuned-q5_k.gguf"

# Defaults and constants
SAMPLE_RATE = 16000
CHUNK_SECONDS = 1.5
OVERLAP_SECONDS = 0.75
N_THREADS = 2
WAKE_WORDS = ["hey sarah", "hi sarah", "hello sarah"]
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


class ASRTimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise ASRTimeoutException()


def transcribe_with_timeout(asr_model, audio_segment, timeout=5):
    # Store the original handler and set the new one
    original_handler = signal.signal(signal.SIGALRM, _timeout_handler)

    # Schedule the alarm
    signal.alarm(timeout)

    try:
        result = asr_model.transcribe(audio_segment)
        return result
    except ASRTimeoutException:
        print(f"ASR timed out after {timeout}s, skipping result...")
        return None
    finally:
        # Cancel the alarm immediately after success or failure
        signal.alarm(0)
        # Safely restore the original signal handler
        signal.signal(signal.SIGALRM, original_handler)


def check_wake_word(text, wake_words, max_distance):
    transcribed_words = text.lower()
    clean_transcribed_words = transcribed_words.translate(
        str.maketrans("", "", string.punctuation)
    )

    for wake_word in wake_words:
        target = wake_word.lower()

        if levenshtein(target, clean_transcribed_words) < max_distance:
            return True

    return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wake word detection using Qwen ASR + Silero VAD",
    )
    parser.add_argument(
        "-w",
        "--wake-words",
        type=str,
        nargs="+",
        default=WAKE_WORDS,
        help="Wake word phrases to listen for (default: %(default)s)",
    )
    parser.add_argument(
        "-c",
        "--command",
        type=str,
        default=None,
        help="Shell command to execute (in the same thread) when a wake word is detected",
    )
    parser.add_argument(
        "-d",
        "--max-distance",
        type=int,
        default=MAX_DISTANCE,
        help="Max Levenshtein distance for fuzzy matching (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=CHUNK_SECONDS,
        help="Audio chunk length in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=OVERLAP_SECONDS,
        help="Overlap between chunks in seconds (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    wake_words = [w.lower() for w in args.wake_words]

    vad_model, get_speech_timestamps = load_vad_model()
    asr_model = load_asr_model()

    chunk_samples = int(args.chunk_seconds * SAMPLE_RATE)
    overlap_samples = int(args.overlap_seconds * SAMPLE_RATE)
    buffer = np.zeros(0, dtype=np.float32)

    print(f"Listening for wake words: {wake_words}")
    if args.command:
        print(f"On wake word, will run: {args.command}")
    print("Speak into the microphone... (Ctrl+C to quit)")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * 0.5),
        callback=audio_callback,
    ) as stream:
        try:
            while True:
                chunk = audio_queue.get()
                buffer = np.concatenate([buffer, chunk])

                if len(buffer) < chunk_samples:
                    continue

                audio_segment = buffer[:chunk_samples]

                audio_tensor = torch.from_numpy(audio_segment)

                speech_timestamps = get_speech_timestamps(
                    audio_tensor, vad_model, sampling_rate=SAMPLE_RATE
                )

                if not speech_timestamps:
                    buffer = buffer[chunk_samples - overlap_samples :]
                    continue

                print("Speech detected, running ASR...")

                stream.stop()

                start_time = time.time()
                result = transcribe_with_timeout(asr_model, audio_segment)
                elapsed_time = time.time() - start_time
                print(f"ASR took {elapsed_time:.2f} seconds.")

                while not audio_queue.empty():
                    audio_queue.get()
                buffer = np.zeros(0, dtype=np.float32)

                stream.start()

                if not result:
                    continue

                text = result.text.strip()
                if not text:
                    continue

                print(f"  [{text}]")

                if check_wake_word(text, wake_words, args.max_distance):
                    if args.command:
                        print(f">>> Wake word detected, running: {args.command}")
                        subprocess.run(args.command, shell=True)
                    else:
                        print(">>> Wake word detected! <<<")

        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
