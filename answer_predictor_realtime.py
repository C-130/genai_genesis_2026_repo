"""
answer_predictor_realtime.py
────────────────────────────
Background audio → Whisper transcription → GPT-2 next-word prediction.

Usage (as a module):
    from answer_predictor_realtime import WordPredictor
    predictor = WordPredictor()
    predictor.start()                    # spawns background threads
    suggestions = predictor.suggestions  # list of up to 3 word strings
"""

import pyaudio
import wave
import tempfile
import threading
import queue
import os
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────
RATE           = 16000
CHUNK          = 1024
RECORD_SECONDS = 5
TOP_K          = 3
ALPHA          = 0.7   # blend: ALPHA * LM_prob + (1-ALPHA) * user_prob


class WordPredictor:
    """
    Runs mic capture + Whisper STT + GPT-2 prediction on background threads.
    Thread-safe: read `.suggestions` and `.last_transcript` from any thread.
    """

    def __init__(self):
        self._audio_queue     = queue.Queue()
        self._lock            = threading.Lock()
        self._suggestions     = []
        self._last_transcript = ""

        # Personalization counters
        self._word_counts = Counter()
        self._total_words = 0

        # OpenAI client
        self._client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))

        # Load GPT-2 once
        print("[WordPredictor] Loading GPT-2…")
        self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self._model     = AutoModelForCausalLM.from_pretrained("gpt2")
        self._model.eval()
        print("[WordPredictor] Model ready.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        """Start background recording and transcription threads."""
        threading.Thread(target=self._record_loop,     daemon=True).start()
        threading.Thread(target=self._transcribe_loop, daemon=True).start()

    @property
    def suggestions(self):
        """Latest top-3 predicted next words (list of str, thread-safe)."""
        with self._lock:
            return list(self._suggestions)

    @property
    def last_transcript(self):
        with self._lock:
            return self._last_transcript

    # ── Personalization ────────────────────────────────────────────────────────

    def _update_word_counts(self, text):
        for word in text.lower().split():
            self._word_counts[word] += 1
            self._total_words       += 1

    def _user_prob(self, word):
        if self._total_words == 0:
            return 0.0
        return self._word_counts.get(word.lower(), 0) / self._total_words

    # ── Prediction ─────────────────────────────────────────────────────────────

    def _predict(self, text):
        inputs = self._tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)
        probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
        top_probs, top_tokens = torch.topk(probs, 50)

        candidates = []
        for prob, token in zip(top_probs[0], top_tokens[0]):
            word = self._tokenizer.decode(token).strip()
            if not word or not word.isalpha():  # skip punctuation / whitespace
                continue
            score = ALPHA * prob.item() + (1 - ALPHA) * self._user_prob(word)
            candidates.append((word, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [w for w, _ in candidates[:TOP_K]]

    # ── Background threads ─────────────────────────────────────────────────────

    def _record_loop(self):
        audio  = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1,
                            rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("[WordPredictor] Listening…")
        while True:
            frames = [stream.read(CHUNK, exception_on_overflow=False)
                      for _ in range(int(RATE / CHUNK * RECORD_SECONDS))]
            self._audio_queue.put(frames)

    def _transcribe_loop(self):
        pa = pyaudio.PyAudio()
        while True:
            frames = self._audio_queue.get()
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    wf = wave.open(f, 'wb')
                    wf.setnchannels(1)
                    wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(frames))
                    wf.close()
                    fname = f.name

                result     = self._client.audio.transcriptions.create(
                    model="whisper-1", file=open(fname, "rb")
                )
                transcript = result.text.strip()
                if not transcript:
                    continue

                self._update_word_counts(transcript)
                words = self._predict(transcript)

                with self._lock:
                    self._last_transcript = transcript
                    self._suggestions     = words

                print(f"[WordPredictor] '{transcript}'  →  {words}")

            except Exception as e:
                print(f"[WordPredictor] Error: {e}")


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    p = WordPredictor()
    p.start()
    while True:
        time.sleep(1)
        print("Suggestions:", p.suggestions)