"""
Intent Agent
============
Uses Gemini to understand intent, ask clarifying questions via the
ClarifyOverlay, draft content, and execute via APIs.

Integrates with the existing codebase:
  - TTS via ElevenLabs (same client as eye_tracker.py)
  - API keys from .env (GEMINI_API_KEY, ELEVEN_LAB_API_KEY)
  - clarify_overlay.ClarifyOverlay for eye-driven Q&A
  - sentence state from eye_tracker.py passed in on execute

Execution backends:
  Email   → Gmail API        → gmail_credentials.json
  SMS     → Email-to-SMS     → free carrier gateway, uses Gmail
  Search  → DuckDuckGo API   → no key needed
  Media   → system media keys via pyautogui
  Speak   → ElevenLabs TTS of current typed sentence
  Emergency → email all contacts + email-to-SMS all contacts
"""

import os
import json
import threading
import traceback
import webbrowser
import urllib.request
import urllib.parse
import time

from dotenv import load_dotenv
load_dotenv()

# ── ElevenLabs ────────────────────────────────────────────────────────────────
try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs.play import play as el_play
    _eleven_key = os.getenv("ELEVEN_LAB_API_KEY", "")
    _tts_client = ElevenLabs(api_key=_eleven_key) if _eleven_key else None
    _TTS = _tts_client is not None
except ImportError:
    _TTS = False
    _tts_client = None

ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"

# ── Gemini ────────────────────────────────────────────────────────────────────
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
    print("[Agent] google-generativeai not installed — pip install google-generativeai")

# ── Gmail (used for both email and email-to-SMS) ──────────────────────────────
try:
    from googleapiclient.discovery import build as _gmail_build
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request as GRequest
    import pickle, base64
    from email.mime.text import MIMEText
    _GMAIL = True
except ImportError:
    _GMAIL = False

# ── pyautogui (media keys) ────────────────────────────────────────────────────
try:
    import pyautogui
    _PYAUTO = True
except ImportError:
    _PYAUTO = False

# ── Contacts ──────────────────────────────────────────────────────────────────
# For SMS gateway: add each contact's carrier gateway address.
# Format: <10-digit-number>@<carrier-gateway>
# Common gateways:
#   AT&T:       number@txt.att.net
#   T-Mobile:   number@tmomail.net
#   Verizon:    number@vtext.com
#   Rogers:     number@pcs.rogers.com
#   Bell:       number@txt.bell.ca
#   Telus:      number@msg.telus.com
CONTACTS = {
    "Mum":    {"email": "mum@example.com",    "sms_gateway": "16135550001@txt.att.net"},
    "Doctor": {"email": "doctor@example.com", "sms_gateway": "16135550002@tmomail.net"},
    "Carer":  {"email": "mohammdalm81@gmail.com",  "sms_gateway": "16477640030@pcs.rogers.com"},
    "Friend": {"email": "friend@example.com", "sms_gateway": "16135550004@txt.bell.ca"},
}

GMAIL_CREDS  = os.path.join(os.path.dirname(__file__), "gmail_credentials.json")
GMAIL_TOKEN  = os.path.join(os.path.dirname(__file__), "gmail_token.pickle")
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


class IntentAgent:
    """
    Handles one intent at a time on a background thread.

    Usage in eye_tracker.py:
        agent = IntentAgent(clarify, on_status, on_speak)
        # when intent tile selected:
        agent.start(intent_id, current_sentence)
        # in up-arrow handler:
        if clarify.visible:
            clarify.press()
        elif intent.visible:
            selected = intent.press()
            if selected:
                agent.start(selected, sentence)
    """

    def __init__(self, clarify, on_status, on_speak):
        self._clarify   = clarify
        self._on_status = on_status
        self._on_speak  = on_speak
        self._thread    = None

    def start(self, intent_id, current_sentence=""):
        self._thread = threading.Thread(
            target=self._run,
            args=(intent_id, current_sentence),
            daemon=True,
        )
        self._thread.start()

    def busy(self):
        return self._thread is not None and self._thread.is_alive()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run(self, intent_id, sentence):
        try:
            if intent_id == "email":
                self._flow_email()
            elif intent_id == "sms":
                self._flow_sms()
            elif intent_id == "search":
                self._flow_search()
            elif intent_id == "media":
                self._flow_media()
            elif intent_id == "emergency":
                self._flow_emergency()
            elif intent_id == "speak":
                self._flow_speak(sentence)
        except Exception as e:
            self._on_status(f"Agent error: {e}")
            traceback.print_exc()

    def _ask(self, question, options):
        return self._clarify.ask(question, options)

    def _gemini_chat(self, system, user):
        if not _GEMINI:
            return "Gemini not available — check GEMINI_API_KEY in .env"
        try:
            response = _gemini.generate_content(f"{system}\n\n{user}")
            return response.text.strip()
        except Exception as e:
            print(f"[Gemini] Error: {e}")
            return f"Gemini error: {e}"

    def _speak(self, text):
        self._on_speak(text)
        if not _TTS or not _tts_client:
            print(f"[TTS] {text}")
            return
        try:
            audio = _tts_client.text_to_speech.convert(
                text=text,
                voice_id=ELEVENLABS_VOICE_ID,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
            )
            el_play(audio)
        except Exception as e:
            print(f"[TTS error] {e}")

    # ── Intent flows ──────────────────────────────────────────────────────────

    def _flow_email(self):
        names = list(CONTACTS.keys()) + ["Cancel"]
        to    = self._ask("Who should I email?", names)
        if to == "Cancel":
            self._on_status("Email cancelled.")
            return

        topics = ["I'm okay", "I need help", "Call me", "Appointment", "Cancel"]
        topic  = self._ask("What's it about?", topics)
        if topic == "Cancel":
            self._on_status("Email cancelled.")
            return

        self._on_status("Gemini is drafting your email…")
        draft = self._gemini_chat(
            system=(
                "Write a short, warm email (2-4 sentences) on behalf of someone "
                "with a physical disability who uses eye tracking to communicate. "
                "Return ONLY valid JSON: {\"subject\": \"...\", \"body\": \"...\"}"
            ),
            user=f"Email to {to} about: {topic}",
        )
        # Strip markdown fences Gemini sometimes wraps around JSON
        clean = draft.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
            clean = clean.strip()
        try:
            data    = json.loads(clean)
            subject = data.get("subject", topic)
            body    = data.get("body", clean)
        except (json.JSONDecodeError, ValueError):
            # Gemini didn't return JSON — use raw text as body
            subject = topic
            body    = clean

        preview = f"To {to}: \"{subject}\""
        confirm = self._ask(preview[:70], ["Send", "Cancel"])
        if confirm == "Cancel":
            self._on_status("Email cancelled.")
            return

        self._on_status("Sending email…")
        sent = self._send_email(CONTACTS[to]["email"], subject, body)
        if sent:
            msg = f"Email sent to {to}."
        else:
            q   = urllib.parse.quote
            url = (f"https://mail.google.com/mail/?view=cm"
                   f"&to={q(CONTACTS[to]['email'])}"
                   f"&su={q(subject)}&body={q(body)}")
            webbrowser.open(url)
            msg = f"Opened Gmail to {to} — Gmail API not configured."

        self._on_status(msg)
        self._speak(msg)

    def _flow_sms(self):
        names = list(CONTACTS.keys()) + ["Cancel"]
        to    = self._ask("Who should I message?", names)
        if to == "Cancel":
            self._on_status("Message cancelled.")
            return

        messages = ["I'm okay", "I need help", "Call me", "On my way", "Cancel"]
        chosen   = self._ask("What message?", messages)
        if chosen == "Cancel":
            self._on_status("Message cancelled.")
            return

        self._on_status("Gemini is writing your message…")
        body = self._gemini_chat(
            system=(
                "Write a short, natural SMS (1-2 sentences) for someone with a "
                "disability. Return ONLY the message text, nothing else."
            ),
            user=f"Message to {to}: {chosen}",
        )

        preview = f"\"{body[:55]}...\"" if len(body) > 55 else f"\"{body}\""
        confirm = self._ask(preview, ["Send", "Cancel"])
        if confirm == "Cancel":
            self._on_status("Message cancelled.")
            return

        self._on_status("Sending message…")
        gateway = CONTACTS[to].get("sms_gateway", "")
        sent    = self._send_email(gateway, "", body) if gateway else False

        if sent:
            msg = f"Message sent to {to}."
        else:
            # Fallback: open WhatsApp Web
            encoded = urllib.parse.quote(body)
            # Try to get a bare phone number from the gateway address
            phone = gateway.split("@")[0] if gateway else ""
            webbrowser.open(f"https://wa.me/{phone}?text={encoded}" if phone
                            else "https://web.whatsapp.com")
            msg = f"Opened WhatsApp to {to}."

        self._on_status(msg)
        self._speak(msg)

    def _flow_search(self):
        topics = ["Latest news", "Weather today", "Health tips",
                  "Nearby restaurants", "Entertainment", "Cancel"]
        topic = self._ask("Search for what?", topics)
        if topic == "Cancel":
            self._on_status("Search cancelled.")
            return

        self._on_status(f"Searching: {topic}…")
        try:
            query   = urllib.parse.quote(topic)
            api_url = (f"https://api.duckduckgo.com/?q={query}"
                       f"&format=json&no_html=1&skip_disambig=1")
            with urllib.request.urlopen(api_url, timeout=8) as r:
                data     = json.loads(r.read().decode())
            abstract = data.get("AbstractText", "") or data.get("Answer", "")

            if abstract:
                summary = self._gemini_chat(
                    system="Summarise in 1-2 clear spoken sentences for text-to-speech.",
                    user=abstract[:800],
                )
                self._on_status("Search result ready.")
                self._speak(summary)
            else:
                self._on_status(f"Opening browser for: {topic}")
                self._speak(f"Opening search results for {topic}.")

            webbrowser.open(f"https://duckduckgo.com/?q={query}")

        except Exception as e:
            self._on_status("Search error — opening browser.")
            webbrowser.open(f"https://duckduckgo.com/?q={urllib.parse.quote(topic)}")

    def _flow_media(self):
        actions = ["Play / Pause", "Next track", "Volume up", "Volume down", "Cancel"]
        action  = self._ask("Media control?", actions)
        if action == "Cancel":
            self._on_status("Cancelled.")
            return

        self._on_status(f"Media: {action}")
        if _PYAUTO:
            key_map = {
                "Play / Pause": "playpause",
                "Next track":   "nexttrack",
                "Volume up":    "volumeup",
                "Volume down":  "volumedown",
            }
            k = key_map.get(action)
            if k:
                pyautogui.press(k)
        self._speak(action)

    def _flow_emergency(self):
        confirm = self._ask("SEND EMERGENCY ALERT to all contacts?",
                            ["YES — send now", "Cancel"])
        if "YES" not in confirm:
            self._on_status("Emergency cancelled.")
            return

        self._on_status("Sending emergency alerts…")
        body    = "EMERGENCY: I need immediate assistance. Please call or come to me now."
        emailed = []
        texted  = []

        for name, info in CONTACTS.items():
            if self._send_email(info["email"], "URGENT: Emergency assistance needed", body):
                emailed.append(name)
            gateway = info.get("sms_gateway", "")
            if gateway and self._send_email(gateway, "", body):
                texted.append(name)

        parts = []
        if emailed:
            parts.append(f"emails to {', '.join(emailed)}")
        if texted:
            parts.append(f"texts to {', '.join(texted)}")
        msg = f"Emergency alerts sent: {'; '.join(parts)}." if parts else \
              "Emergency sent — configure Gmail for delivery."
        self._on_status(msg)
        self._speak("Emergency alert sent. Help is on the way.")

    def _flow_speak(self, sentence):
        if not sentence.strip():
            self._on_status("Nothing typed to speak yet.")
            self._speak("There is nothing typed yet.")
            return
        self._on_status("Speaking…")
        self._speak(sentence.strip())
        self._on_status("Done speaking.")

    # ── Gmail helper (used for both email and email-to-SMS) ───────────────────

    def _send_email(self, to, subject, body):
        """
        Send via Gmail API. Works for both regular email and email-to-SMS
        gateway addresses — the carrier converts the email to a text message.
        """
        if not _GMAIL or not os.path.exists(GMAIL_CREDS):
            return False
        try:
            creds = None
            if os.path.exists(GMAIL_TOKEN):
                with open(GMAIL_TOKEN, "rb") as f:
                    creds = pickle.load(f)
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(GRequest())
                else:
                    flow  = InstalledAppFlow.from_client_secrets_file(
                        GMAIL_CREDS, GMAIL_SCOPES)
                    creds = flow.run_local_server(port=0)
                with open(GMAIL_TOKEN, "wb") as f:
                    pickle.dump(creds, f)

            svc = _gmail_build("gmail", "v1", credentials=creds)
            msg = MIMEText(body)
            msg["to"]      = to
            msg["subject"] = subject
            raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
            svc.users().messages().send(userId="me", body={"raw": raw}).execute()
            return True
        except Exception as e:
            print(f"[Gmail] {e}")
            return False