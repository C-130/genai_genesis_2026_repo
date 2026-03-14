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

client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))

# -----------------------------
# Audio Config
# -----------------------------
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5

audio_queue = queue.Queue()

# -----------------------------
# Load Language Model
# -----------------------------
print("Loading language model...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()
print("Model ready.\n")

# -----------------------------
# Rolling transcript history
# Used for personalization scores
# -----------------------------
transcript_history = []
word_counts = Counter()
total_words = 0


def update_word_counts(text):
    global total_words
    for word in text.lower().split():
        word_counts[word] += 1
        total_words += 1


def user_probability(word):
    if total_words == 0:
        return 0
    return word_counts[word.lower()] / total_words if word.lower() in word_counts else 0


# -----------------------------
# Next Word Prediction
# -----------------------------
def predict_next_words(text, top_k=5, alpha=0.7):
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    next_token_logits = logits[:, -1, :]
    probs = torch.softmax(next_token_logits, dim=-1)
    top_probs, top_tokens = torch.topk(probs, 50)

    candidates = []
    for prob, token in zip(top_probs[0], top_tokens[0]):
        word = tokenizer.decode(token).strip()
        if not word:
            continue
        lm_prob = prob.item()
        user_prob = user_probability(word)
        score = alpha * lm_prob + (1 - alpha) * user_prob
        candidates.append((word, score, lm_prob, user_prob))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]


# -----------------------------
# Audio Recording
# -----------------------------
def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Listening...")
    while True:
        frames = [stream.read(CHUNK, exception_on_overflow=False)
                  for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS))]
        audio_queue.put(frames)


# -----------------------------
# Transcription + Prediction
# -----------------------------
def transcribe_audio():
    audio = pyaudio.PyAudio()
    while True:
        frames = audio_queue.get()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wf = wave.open(f, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            result = client.audio.transcriptions.create(
                model="whisper-1", file=open(f.name, "rb")
            )
            transcript = result.text.strip()

            if not transcript:
                continue

            print(f"\nTranscript: {transcript}")

            # Update personalization history with new transcript
            transcript_history.append(transcript)
            update_word_counts(transcript)

            # Predict next words based on the latest transcript
            suggestions = predict_next_words(transcript)
            print("Next word suggestions:")
            for word, score, lm, user in suggestions:
                print(f"  {word:15} score={score:.4f}  LM={lm:.4f}  user={user:.4f}")


# -----------------------------
# Main
# -----------------------------
threading.Thread(target=record_audio, daemon=True).start()
transcribe_audio()