"""
Moorcheh Semantic Memory
========================
Stores every completed sentence the user types and retrieves semantically
similar past phrases as typing suggestions — much more useful for AAC than
generic next-word prediction.

Real Moorcheh SDK API (moorcheh-sdk):
  client.namespaces.create(namespace_name, type="text")
  client.documents.upload(namespace_name, documents=[{"id":..., "text":...}])
  client.similarity_search.query(namespaces=[namespace_name], query, top_k)

Free tier strategy:
  - Query only after 2+ chars typed AND 1s pause (debounced)
  - Store phrases in 30s batches, not one at a time
  - Cache last result so no repeat queries for same text

Setup:
  pip install moorcheh-sdk
  Add to .env: MOORCHEH_API_KEY=your_key
  Get key at:  https://console.moorcheh.ai/api-keys
"""

import os
import time
import threading
from dotenv import load_dotenv

load_dotenv()

try:
    from moorcheh_sdk import MoorchehClient
    _MOORCHEH = True
except ImportError:
    _MOORCHEH = False
    print("[Moorcheh] SDK not installed — pip install moorcheh-sdk")

NAMESPACE_NAME  = "user_phrases"
MAX_SUGGESTIONS = 3
QUERY_DEBOUNCE  = 1.0   # seconds since last keystroke before querying
MIN_CHARS       = 2     # minimum characters typed before querying


class MoorchehMemory:
    """
    Semantic phrase memory for AAC text prediction.

    Usage in eye_tracker.py:
        memory = MoorchehMemory()
        memory.start()
        # each frame:
        memory.on_typing(kb.typed)
        # read suggestions (non-blocking):
        suggestions = memory.suggestions
        # on sentence complete (ENT key):
        memory.store_phrase(sentence)
    """

    def __init__(self):
        self._client          = None
        self._suggestions     = []
        self._lock            = threading.Lock()
        self._last_query      = ""
        self._last_typed      = ""
        self._last_type_t     = 0.0
        self._pending_phrases = []
        self._running         = False

        if not _MOORCHEH:
            return

        key = os.getenv("MOORCHEH_API_KEY", "")
        if not key:
            print("[Moorcheh] No MOORCHEH_API_KEY in .env — skipping")
            return

        try:
            self._client = MoorchehClient(api_key=key)
            self._ensure_namespace()
            print("[Moorcheh] Connected.")
        except Exception as e:
            print(f"[Moorcheh] Connection error: {e}")
            self._client = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        threading.Thread(target=self._query_loop, daemon=True).start()
        threading.Thread(target=self._store_loop, daemon=True).start()

    def stop(self):
        self._running = False
        self._flush_phrases()

    def on_typing(self, current_text):
        """Call every frame — safe, no API calls here."""
        self._last_typed  = current_text
        self._last_type_t = time.time()

    def store_phrase(self, phrase):
        """Call when sentence is completed (ENT pressed)."""
        phrase = phrase.strip()
        if len(phrase) < 4:
            return
        with self._lock:
            self._pending_phrases.append(phrase)
        print(f"[Moorcheh] Queued: '{phrase}'")

    @property
    def suggestions(self):
        with self._lock:
            return list(self._suggestions)

    # ── Background threads ─────────────────────────────────────────────────────

    def _query_loop(self):
        """Fires a Moorcheh query when: ≥2 chars typed AND ≥1s pause AND text changed."""
        while self._running:
            time.sleep(0.1)
            if not self._client:
                continue
            now               = time.time()
            current           = self._last_typed.strip()
            paused_long_enough = (now - self._last_type_t) >= QUERY_DEBOUNCE
            text_changed       = current != self._last_query
            long_enough        = len(current) >= MIN_CHARS

            if paused_long_enough and text_changed and long_enough:
                self._last_query = current
                print(f"[Moorcheh] Querying for: '{current}'")
                self._do_query(current)

    def _store_loop(self):
        """Flushes pending phrases to Moorcheh every 30 seconds."""
        while self._running:
            time.sleep(30)
            self._flush_phrases()

    # ── Moorcheh operations ────────────────────────────────────────────────────

    def _ensure_namespace(self):
        try:
            self._client.namespaces.create(
                namespace_name=NAMESPACE_NAME,
                type="text"
            )
            print(f"[Moorcheh] Created namespace '{NAMESPACE_NAME}'")
            self._seed_defaults()
        except Exception as e:
            if "already exists" in str(e) or "Conflict" in str(e):
                pass   # namespace exists from a previous run — that's fine
            else:
                print(f"[Moorcheh] Namespace setup error: {e}")

    def _seed_defaults(self):
        """Seed common AAC phrases so suggestions work from the first session."""
        defaults = [
            "I need help please",
            "Can you bring me water",
            "I am in pain",
            "Please call my doctor",
            "I need to use the bathroom",
            "I am tired and want to rest",
            "Can you adjust my position",
            "Please turn on the TV",
            "I am feeling okay today",
            "Thank you so much",
            "Good morning",
            "Good night",
            "I love you",
            "Please call me",
            "I am hungry",
            "It is too hot in here",
            "It is too cold in here",
            "I need my medication",
            "Please contact my carer",
            "Can you open the window",
        ]
        docs = [{"id": f"default_{i}", "text": p}
                for i, p in enumerate(defaults)]
        try:
            self._client.documents.upload(
                namespace_name=NAMESPACE_NAME,
                documents=docs
            )
            print(f"[Moorcheh] Seeded {len(docs)} default phrases.")
        except Exception as e:
            print(f"[Moorcheh] Seed error: {e}")

    def _do_query(self, text):
        try:
            results = self._client.similarity_search.query(
                namespaces=[NAMESPACE_NAME],
                query=text,
                top_k=MAX_SUGGESTIONS + 2,
            )
            # Moorcheh returns {'results': [...]} — unwrap before iterating
            result_list = results.get("results", []) if isinstance(results, dict) else list(results)

            phrases = []
            for r in result_list:
                if isinstance(r, dict):
                    p = (r.get("text") or r.get("document") or
                         r.get("content") or "").strip()
                else:
                    p = (getattr(r, "text", "") or
                         getattr(r, "document", "") or
                         getattr(r, "content", "")).strip()
                if p and p.lower() != text.lower():
                    phrases.append(p)
                if len(phrases) >= MAX_SUGGESTIONS:
                    break

            with self._lock:
                self._suggestions = phrases

            if phrases:
                print(f"[Moorcheh] Suggestions for '{text}': {phrases}")
            else:
                print(f"[Moorcheh] No matches for '{text}'")

        except Exception as e:
            print(f"[Moorcheh] Query error: {e}")

    def _flush_phrases(self):
        with self._lock:
            if not self._pending_phrases:
                return
            to_store = list(self._pending_phrases)
            self._pending_phrases = []

        if not self._client:
            return

        try:
            ts   = int(time.time())
            docs = [{"id": f"user_{ts}_{i}", "text": p}
                    for i, p in enumerate(to_store)]
            self._client.documents.upload(
                namespace_name=NAMESPACE_NAME,
                documents=docs
            )
            print(f"[Moorcheh] Stored {len(docs)} phrase(s).")
        except Exception as e:
            print(f"[Moorcheh] Store error: {e}")
            with self._lock:
                self._pending_phrases = to_store + self._pending_phrases