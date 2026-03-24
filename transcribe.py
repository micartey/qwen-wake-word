import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import torch

torch.set_num_threads(4)

import multiprocessing
import queue
import sys
import time

import numpy as np
import sounddevice as sd

from wake_word import (
    SAMPLE_RATE,
    load_vad_model,
)

VAD_CHUNK_SAMPLES = 512

SPEECH_PAD_SAMPLES = int(SAMPLE_RATE * 0.3)
MIN_SPEECH_SAMPLES = int(SAMPLE_RATE * 0.4)
MAX_SPEECH_SAMPLES = int(SAMPLE_RATE * 10)
SILENCE_THRESHOLD_CHUNKS = 12

audio_queue: queue.Queue[np.ndarray] = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata[:, 0].copy())


def transcription_worker(task_queue, result_queue):
    from wake_word import load_asr_model

    asr_model = load_asr_model()
    result_queue.put("ready")

    while True:
        audio_segment = task_queue.get()
        if audio_segment is None:
            break

        start_time = time.time()
        result = asr_model.transcribe(audio_segment)
        elapsed = time.time() - start_time

        if result and result.text.strip():
            result_queue.put((result.text.strip(), elapsed))


def main():
    ctx = multiprocessing.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    worker = ctx.Process(
        target=transcription_worker,
        args=(task_queue, result_queue),
        daemon=True,
    )
    worker.start()

    vad_model, _ = load_vad_model()

    print("Waiting for ASR model in worker process...")
    result_queue.get()

    print("Streaming speech-to-text. Speak into the microphone... (Ctrl+C to quit)")

    in_speech = False
    silence_chunks = 0
    speech_buffer = np.zeros(0, dtype=np.float32)
    vad_buffer = np.zeros(0, dtype=np.float32)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * 0.1),
        callback=audio_callback,
    ):
        try:
            while True:
                try:
                    text, elapsed = result_queue.get_nowait()
                    print(f"  [{text}] ({elapsed:.2f}s)")
                except queue.Empty:
                    pass

                try:
                    chunk = audio_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                vad_buffer = np.concatenate([vad_buffer, chunk])

                while len(vad_buffer) >= VAD_CHUNK_SAMPLES:
                    vad_chunk = vad_buffer[:VAD_CHUNK_SAMPLES]
                    vad_buffer = vad_buffer[VAD_CHUNK_SAMPLES:]

                    tensor = torch.from_numpy(vad_chunk)
                    speech_prob = vad_model(tensor, SAMPLE_RATE).item()
                    is_speech = speech_prob > 0.5

                    if is_speech:
                        if not in_speech:
                            pad_start = max(0, len(speech_buffer) - SPEECH_PAD_SAMPLES)
                            speech_buffer = speech_buffer[pad_start:]
                            in_speech = True
                        silence_chunks = 0
                        speech_buffer = np.concatenate([speech_buffer, vad_chunk])
                    else:
                        if in_speech:
                            silence_chunks += 1
                            speech_buffer = np.concatenate([speech_buffer, vad_chunk])

                            if silence_chunks >= SILENCE_THRESHOLD_CHUNKS:
                                if len(speech_buffer) >= MIN_SPEECH_SAMPLES:
                                    task_queue.put(speech_buffer.copy())
                                speech_buffer = np.zeros(0, dtype=np.float32)
                                in_speech = False
                                silence_chunks = 0
                            elif len(speech_buffer) >= MAX_SPEECH_SAMPLES:
                                task_queue.put(speech_buffer.copy())
                                speech_buffer = np.zeros(0, dtype=np.float32)
                                in_speech = False
                                silence_chunks = 0
                        else:
                            speech_buffer = np.concatenate([speech_buffer, vad_chunk])
                            if len(speech_buffer) > SPEECH_PAD_SAMPLES * 2:
                                speech_buffer = speech_buffer[-SPEECH_PAD_SAMPLES:]

        except KeyboardInterrupt:
            print("\nStopping...")
            task_queue.put(None)
            worker.join(timeout=5)
            print("Stopped.")


if __name__ == "__main__":
    main()
