"""
Nightly Health Survey
=====================
Each evening, prompts the user with ALS-specific questions via the
ClarifyOverlay. Responses are stored to Moorcheh and locally.
Gemini analyses trends and fires carer alerts when needed.

Moorcheh SDK API used:
  client.namespaces.create / list
  client.documents.upload
  (no search needed here — we read from local log for trend analysis)
"""

import os
import json
import time
import datetime
import threading
from dotenv import load_dotenv

try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs.play import play as el_play
    _ELEVEN = True
except ImportError:
    _ELEVEN = False

load_dotenv()

try:
    from moorcheh_sdk import MoorchehClient
    _MOORCHEH = True
except ImportError:
    _MOORCHEH = False

try:
    import google.generativeai as genai
    _gemini_key = os.getenv("GEMINI_API_KEY", "")
    if _gemini_key:
        genai.configure(api_key=_gemini_key)
        _gemini = genai.GenerativeModel("gemini-2.5-flash")
    else:
        _gemini = None
    _GEMINI = _gemini is not None
except ImportError:
    _GEMINI = False
    _gemini = None
    print("[HealthSurvey] google-generativeai not installed — pip install google-generativeai")

SURVEY_NAMESPACE = "health_surveys"
SURVEY_HOUR      = 20   # 8pm nightly trigger
SURVEY_QUESTIONS = [
    {
        "id":       "motor",
        "question": "How is your muscle control and movement today?",
        "options":  ["Very difficult", "Difficult", "Manageable", "Pretty good", "Skip"],
        "scores":   [1, 2, 3, 4, None],
    },
    {
        "id":       "fatigue",
        "question": "How tired have you felt today?",
        "options":  ["Exhausted", "Very tired", "Somewhat tired", "Energetic", "Skip"],
        "scores":   [1, 2, 3, 4, None],
    },
    {
        "id":       "mood",
        "question": "How is your mood and mental wellbeing today?",
        "options":  ["Very low", "Low", "Okay", "Good", "Skip"],
        "scores":   [1, 2, 3, 4, None],
    },
    {
        "id":       "communication",
        "question": "How easy was it to communicate today?",
        "options":  ["Very hard", "Hard", "Okay", "Easy", "Skip"],
        "scores":   [1, 2, 3, 4, None],
    },
    {
        "id":       "pain",
        "question": "How much discomfort or pain did you feel today?",
        "options":  ["Severe", "Moderate", "Mild", "None", "Skip"],
        "scores":   [1, 2, 3, 4, None],
    },
]


class HealthSurvey:
    def __init__(self, clarify, on_status, on_speak, on_alert=None, tts_client=None):
        self._clarify     = clarify
        self._on_status   = on_status
        self._on_speak    = on_speak
        self._on_alert    = on_alert
        self._tts_client  = tts_client   # ElevenLabs client for spoken alerts
        self._running     = False
        self._last_survey_date = None

        self._moorcheh = None
        if _MOORCHEH:
            key = os.getenv("MOORCHEH_API_KEY", "")
            if key:
                try:
                    self._moorcheh = MoorchehClient(api_key=key)
                    self._ensure_namespace()
                except Exception as e:
                    print(f"[HealthSurvey] Moorcheh error: {e}")

    def start(self, run_on_start=False):
        self._running = True
        threading.Thread(target=self._timer_loop, daemon=True).start()
        if run_on_start:
            threading.Timer(3.0, self.run_now).start()

    def stop(self):
        self._running = False

    def run_now(self):
        threading.Thread(target=self._run_survey, daemon=True).start()

    # ── Timer ──────────────────────────────────────────────────────────────────

    def _timer_loop(self):
        while self._running:
            now   = datetime.datetime.now()
            today = now.date()
            if now.hour >= SURVEY_HOUR and self._last_survey_date != today:
                self._last_survey_date = today
                self._run_survey()
            time.sleep(60)

    # ── Survey flow ────────────────────────────────────────────────────────────

    def _run_survey(self):
        self._on_status("Starting nightly health check-in…")
        self._on_speak("Time for your evening check-in. I have five quick questions.")

        responses = {}
        for q in SURVEY_QUESTIONS:
            try:
                answer = self._clarify.ask(q["question"], q["options"])
                if answer == "Skip":
                    responses[q["id"]] = None
                    continue
                idx = q["options"].index(answer)
                responses[q["id"]] = q["scores"][idx]
            except Exception as e:
                print(f"[HealthSurvey] Question error: {e}")
                responses[q["id"]] = None

        self._on_speak("Thank you. I've recorded how you're feeling tonight.")
        self._on_status("Health check-in complete.")
        self._store_response(responses)

        # Run trend analysis every 3 entries
        history = self._load_recent_history(days=14)
        if len(history) % 3 == 0:
            self._analyse_trends()
        else:
            remaining = 3 - (len(history) % 3)
            print(f"[HealthSurvey] {len(history)} entries recorded — analysis in {remaining} more")

    # ── Storage ────────────────────────────────────────────────────────────────

    def _store_response(self, responses):
        date_str = datetime.date.today().isoformat()
        record   = {"date": date_str, "responses": responses}

        # Save locally
        local_path = os.path.join(os.path.dirname(__file__), "health_log.jsonl")
        with open(local_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[HealthSurvey] Saved locally: {record}")

        # Also store to Moorcheh
        if self._moorcheh:
            text = self._record_to_text(date_str, responses)
            try:
                self._moorcheh.documents.upload(
                    namespace_name=SURVEY_NAMESPACE,
                    documents=[{"id": f"survey_{date_str}", "text": text}]
                )
                print(f"[HealthSurvey] Stored to Moorcheh: {date_str}")
            except Exception as e:
                print(f"[HealthSurvey] Moorcheh store error: {e}")

    def _record_to_text(self, date_str, responses):
        label_map = {
            "motor":         {1: "very difficult", 2: "difficult", 3: "manageable", 4: "good"},
            "fatigue":       {1: "exhausted",      2: "very tired", 3: "tired",     4: "energetic"},
            "mood":          {1: "very low",       2: "low",        3: "okay",      4: "good"},
            "communication": {1: "very hard",      2: "hard",       3: "okay",      4: "easy"},
            "pain":          {1: "severe",         2: "moderate",   3: "mild",      4: "none"},
        }
        parts = [f"Health check-in for {date_str}."]
        for dim, score in responses.items():
            if score is not None:
                label = label_map.get(dim, {}).get(score, str(score))
                parts.append(f"{dim.capitalize()}: {label} ({score}/4).")
        return " ".join(parts)

    # ── Trend analysis ─────────────────────────────────────────────────────────

    def _analyse_trends(self):
        if not _GEMINI:
            return

        history = self._load_recent_history(days=14)

        history_text = "\n".join(
            self._record_to_text(r["date"], r["responses"])
            for r in history
        )

        prompt = (
            "You are a health monitoring assistant for someone with ALS. "
            "Analyse the following daily check-in responses. "
            "Identify any concerning trends: any dimension declining 3+ days in a row, "
            "any score dropping more than 1 point in a single day, or overall low mood. "
            "If there are no concerns, respond with exactly: NO_ALERT "
            "If there are concerns, respond with a single short sentence (under 20 words) "
            "describing the most urgent concern, starting with ALERT:\n\n"
            + history_text
        )

        try:
            response = _gemini.generate_content(prompt)
            analysis = response.text.strip()
            print(f"[HealthSurvey] Trend analysis: {analysis}")
            if analysis.startswith("ALERT:"):
                self._trigger_alert(analysis[len("ALERT:"):].strip())
        except Exception as e:
            print(f"[HealthSurvey] Analysis error: {e}")

    def _trigger_alert(self, message):
        full_msg = f"Health alert: {message}"
        self._on_status(full_msg)
        print(f"[HealthSurvey] ALERT: {message}")
        if self._on_alert:
            self._on_alert(full_msg)

        # Generate a warm, spoken-language version of the alert with Gemini
        spoken_text = full_msg  # fallback if Gemini unavailable
        if _GEMINI:
            try:
                # First ask Gemini to rate severity 1-3
                sev_resp = _gemini.generate_content(
                    f"Rate the severity of this health alert as a single digit: "
                    f"1 (mild trend, informational), 2 (moderate concern), "
                    f"3 (urgent, needs immediate attention). "
                    f"Reply with only the digit.\n\nAlert: {message}"
                )
                try:
                    severity = int(sev_resp.text.strip()[0])
                except Exception:
                    severity = 2

                tone_map = {
                    1: ("warm and gently informational, no urgency",
                        "something like: 'Just a gentle heads up — ...'"),
                    2: ("caring but clearly concerned, encouraging action",
                        "something like: 'I've noticed something worth paying attention to — ...'"),
                    3: ("calm but urgent and direct, making clear this needs immediate attention",
                        "something like: 'This is important — please get help right away. ...'"),
                }
                tone, example = tone_map.get(severity, tone_map[2])

                phrase_prompt = (
                    f"You are a caring health assistant. Convert this clinical alert into a "
                    f"spoken sentence (under 25 words) addressed directly to the patient "
                    f"— it will be read aloud to them from their own device. "
                    f"Tone should be {tone}. For example, {example} "
                    f"Do not start with 'Alert'. Just the sentence, nothing else.\n\n"
                    f"Alert: {message}"
                )
                print(f"[HealthSurvey] Alert severity: {severity}/3")
                resp = _gemini.generate_content(phrase_prompt)
                spoken_text = resp.text.strip()
                print(f"[HealthSurvey] Spoken alert: {spoken_text}")
            except Exception as e:
                print(f"[HealthSurvey] Gemini phrase error: {e}")

        # Speak via ElevenLabs
        if _ELEVEN and self._tts_client:
            try:
                audio = self._tts_client.text_to_speech.convert(
                    text=spoken_text,
                    voice_id="JBFqnCBsd6RMkjVDRZzb",
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128",
                )
                el_play(audio)
            except Exception as e:
                print(f"[HealthSurvey] TTS error: {e}")
        else:
            self._on_speak(spoken_text)

    def _load_recent_history(self, days=14):
        local_path = os.path.join(os.path.dirname(__file__), "health_log.jsonl")
        if not os.path.exists(local_path):
            return []
        cutoff = datetime.date.today() - datetime.timedelta(days=days)
        records = []
        with open(local_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if datetime.date.fromisoformat(r["date"]) >= cutoff:
                        records.append(r)
                except Exception:
                    continue
        return sorted(records, key=lambda r: r["date"])

    # ── Moorcheh namespace setup ───────────────────────────────────────────────

    def _ensure_namespace(self):
        try:
            existing = self._moorcheh.namespaces.list()
            names = []
            for ns in existing:
                if isinstance(ns, dict):
                    names.append(ns.get("name") or ns.get("namespace_name", ""))
                else:
                    names.append(getattr(ns, "name", "") or getattr(ns, "namespace_name", ""))
            if SURVEY_NAMESPACE not in names:
                self._moorcheh.namespaces.create(
                    namespace_name=SURVEY_NAMESPACE,
                    type="text"
                )
                print(f"[HealthSurvey] Created namespace '{SURVEY_NAMESPACE}'")
        except Exception as e:
            print(f"[HealthSurvey] Namespace setup error: {e}")