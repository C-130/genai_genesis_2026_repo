# GazeKeys
### *ALS takes their hands. We give them back their voice.*

> **"Look. Blink. Speak."**

GazeKeys turns a standard laptop webcam into a full eye-tracking accessibility system for people with ALS and other motor disabilities — no specialist hardware required.

---

## The Problem

Tomorrow, three people will be diagnosed with ALS and given a life expectancy of two to five years. The day after, three more. Amyotrophic Lateral Sclerosis is a relentless, progressive neurodegenerative disease that strips away every voluntary movement — your hands, your voice, your ability to breathe — while leaving your mind completely intact. There is no cure. Over 32,000 Americans are living with ALS today, and that number is projected to grow by 25% by 2040.

What struck us was not just the cruelty of the disease itself — but the cruelty of what patients are left with. As ALS progresses, most patients lose the ability to use their hands entirely. And yet the tools that exist to help them communicate cost **$5,000 to $15,000**. Dedicated infrared eye tracking hardware is financially out of reach for most families. Many patients are left staring at a screen they cannot control, in a body they cannot move, unable to ask for water.

We built GazeKeys because we believe that's unacceptable — and fixable.

---

## What It Does

Using only their eyes, a GazeKeys user can:

- **Type** using a gaze-controlled on-screen keyboard with AI-powered semantic phrase prediction
- **Send emails and texts** — select an intent, let the AI agent draft and deliver
- **Browse the web** with full OS cursor control across the entire screen
- **Complete nightly health check-ins** tracked over time, with Gemini detecting declining trends and speaking personalised alerts aloud
- **Speak typed sentences aloud** via ElevenLabs text-to-speech

**Hardware required: a webcam.**

---

## Demo Controls

| Key | Action |
|-----|--------|
| `UP ARROW` | Select hovered key / confirm / click (in browse mode) |
| `SPACE` | Start / confirm calibration point |
| `I` | Open intent / actions menu |
| `H` | Trigger health survey now |
| `F9` | Exit browse mode |
| `N` | Reset neutral gaze baseline |
| `R` | Full reset |
| `Q / ESC` | Quit |

---

## Architecture

```
Webcam → MediaPipe FaceMesh (478 landmarks, iris tracking)
       → Gaze cursor (baseline subtraction + EMA smoothing)
       → Intent selection (large dwell tiles, no pixel precision needed)
       → Gemini 2.5 Flash agent (clarifying questions → draft → execute)
       → Gmail API / email-to-SMS / pyautogui media keys
       → ElevenLabs TTS (spoken output)
       → Moorcheh semantic memory (personalised phrase prediction)
       → Health survey → Gemini trend analysis → spoken alert
```

---

## Setup

**Requirements:** Python 3.10–3.12

```bash
git clone https://github.com/yourteam/gazekeys
cd gazekeys
pip install -r requirements.txt
pip install moorcheh-sdk pynput google-genai pyautogui
```

### .env file

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=...
ELEVEN_LAB_API_KEY=...
OPEN_AI_API_KEY=...
MOORCHEH_API_KEY=...
```

Get your keys:
- **Gemini:** [aistudio.google.com](https://aistudio.google.com) — free tier
- **ElevenLabs:** [elevenlabs.io](https://elevenlabs.io) — free tier
- **Moorcheh:** [console.moorcheh.ai/api-keys](https://console.moorcheh.ai/api-keys) — free tier
- **OpenAI:** [platform.openai.com](https://platform.openai.com) — used for Whisper STT

### Gmail setup (for email + SMS features)

1. Go to [console.cloud.google.com](https://console.cloud.google.com) → New Project
2. Enable the **Gmail API**
3. Create **OAuth 2.0 credentials** (Desktop app) → download JSON
4. Rename to `gmail_credentials.json` and place in the project folder
5. Add your Gmail address as a test user under OAuth consent screen → Test users
6. On first email send, a browser window will open for one-time authorisation

### Contacts + SMS gateways

Edit `CONTACTS` in `agent.py`:

```python
CONTACTS = {
    "Mum":  {"email": "mum@example.com", "sms_gateway": "16135550001@txt.bell.ca"},
    ...
}
```

Common carrier SMS gateways:

| Carrier | Gateway |
|---------|---------|
| Bell | `number@txt.bell.ca` |
| Rogers | `number@pcs.rogers.com` |
| Telus | `number@msg.telus.com` |
| AT&T | `number@txt.att.net` |
| T-Mobile | `number@tmomail.net` |
| Verizon | `number@vtext.com` |

### Run

```bash
python eye_tracker.py
```

The face landmark model (~7 MB) downloads automatically on first run.

---

## File Structure

```
eye_tracker.py          — main loop, all module wiring
virtual_keyboard.py     — on-screen keyboard + button bar + suggestion chips
gaze_cursor.py          — iris tracking → cursor + browse mode OS control
agent.py                — Gemini AI agent: email, SMS, search, media, emergency
health_survey.py        — nightly check-in, trend analysis, spoken alerts
moorcheh_memory.py      — semantic phrase memory + AAC prediction
intent_overlay.py       — large-tile intent selection UI
clarify_overlay.py      — AI follow-up question tiles
calibrator.py           — 9-point gaze calibration
cursor_aura.py          — glowing ring overlay for browse mode cursor
utils.py                — shared constants, landmark indices, drawing helpers
answer_predictor_realtime.py — Whisper STT + GPT-2 fallback predictor
```

---

## Tuning

All sensitivity values are in `utils.py`:

```python
CURSOR_SPEED    = 1000   # px/sec at full deflection
DEAD_ZONE_X     = 0.06   # smaller = responds to subtler eye movements
DEAD_ZONE_Y     = 0.06
SMOOTH_ALPHA    = 0.35   # higher = snappier, lower = smoother
```

---

## Built With

- [MediaPipe](https://ai.google.dev/edge/mediapipe) — iris landmark detection
- [Gemini 2.5 Flash](https://aistudio.google.com) — AI agent + health trend analysis
- [ElevenLabs](https://elevenlabs.io) — text-to-speech
- [Moorcheh](https://moorcheh.ai) — semantic memory + phrase prediction
- [OpenCV](https://opencv.org) — camera feed + UI rendering
- [pynput](https://pynput.readthedocs.io) — global hotkey listener
- [pyautogui](https://pyautogui.readthedocs.io) — OS cursor control + media keys
- Gmail API — email delivery

---

## Team

Built at **GenAI Genesis 2026** in 24 hours by a team of 3.
