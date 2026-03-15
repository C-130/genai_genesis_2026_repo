"""
Eye Tracker Utils — Helper functions and constants
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import os
import urllib.request
from collections import deque

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, FaceLandmarker
from mediapipe import Image, ImageFormat

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("📥  Downloading face landmarker model (~7 MB) …")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception:
            # macOS Python 3.13 SSL cert issue — bypass verification
            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(MODEL_URL, context=ctx) as r, \
                 open(MODEL_PATH, "wb") as f:
                f.write(r.read())
        print(f"✅  Saved to {MODEL_PATH}")

# ── Landmark indices ───────────────────────────────────────────────────────────
LEFT_IRIS         = [474, 475, 476, 477]
RIGHT_IRIS        = [469, 470, 471, 472]
LEFT_EYE_CORNERS  = [362, 263]
RIGHT_EYE_CORNERS = [33,  133]
LEFT_EYE_LIDS     = [386, 374]
RIGHT_EYE_LIDS    = [159, 145]

# Stable vertical anchors: inner eye corners sit at a fixed Y when gaze is centred.
# Top anchor = midpoint of eye corners (doesn't move with lid).
# We measure iris Y relative to the eye corner midpoint, normalised by
# the horizontal eye span (a proxy for face scale).
LEFT_EYE_INNER    = 362   # inner corner of left eye
RIGHT_EYE_INNER   = 133   # inner corner of right eye

# ── Helpers ────────────────────────────────────────────────────────────────────

def iris_center(lm_list, indices, w, h):
    pts = np.array([[lm_list[i].x * w, lm_list[i].y * h] for i in indices])
    return pts.mean(axis=0)

def eye_bounds(lm_list, corner_idx, lid_idx, w, h):
    xs = [lm_list[i].x * w for i in corner_idx]
    ys = [lm_list[i].y * h for i in lid_idx]
    return min(xs), max(xs), min(ys), max(ys)

def gaze_ratio(iris_pt, x0, x1, y0, y1):
    rx = (iris_pt[0] - x0) / max(x1 - x0, 1)
    ry = (iris_pt[1] - y0) / max(y1 - y0, 1)
    return float(np.clip(rx, 0, 1)), float(np.clip(ry, 0, 1))

def gaze_offset(iris_pt, x0, x1, y0, y1, corner_y):
    """
    Return (dx, dy) normalised gaze offsets.

    dx: iris X vs eye horizontal centre — reliable, lids don't affect X.
    dy: iris Y vs the eye CORNER Y (corner_y) — corners are fixed in the
        skull so they don't move when the eye rotates up/down, making this
        a stable vertical reference. Normalised by half the eye width.

    Positive dx = looking right.
    Positive dy = looking down.
    """
    cx  = (x0 + x1) / 2
    hw  = max((x1 - x0) / 2, 1)
    dx  = (iris_pt[0] - cx)     / hw
    dy  = (iris_pt[1] - corner_y) / hw   # use hw as scale — same unit as dx
    return dx, dy * 3.0   # amplify dy: vertical range is much smaller than horizontal

# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_gaze_dot(frame, x, y, r=22):
    overlay = frame.copy()
    cv2.circle(overlay, (x, y), r, (0, 220, 255), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.circle(frame, (x, y), r, (0, 220, 255), 2)
    cv2.circle(frame, (x, y), 4,  (255, 255, 255), -1)

def draw_cal_dot(frame, x, y, pulse):
    r = int(14 + 6 * pulse)
    cv2.circle(frame, (x, y), r+6, (0, 0, 180), 2)
    cv2.circle(frame, (x, y), r,   (0, 0, 255), -1)
    cv2.circle(frame, (x, y), 4,   (255, 255, 255), -1)

def draw_eye_box(frame, iris_pt, x0, x1, y0, y1):
    ix, iy = int(iris_pt[0]), int(iris_pt[1])
    cv2.circle(frame, (ix, iy), 4, (0, 255, 160), -1)
    cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 160), 1)

def draw_hud(frame, cal, w, h, cursor_active=False, cursor=None):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if cursor is not None and not cursor.calibrated:
        pct = int(cursor.baseline_progress() * 100)
        msg = f"Look straight ahead — measuring neutral gaze...  {pct}%"
        col = (80, 200, 255)
        # Progress bar under the text
        bar_w = int((w - 20) * cursor.baseline_progress())
        cv2.rectangle(frame, (10, 33), (10 + bar_w, 38), (80, 200, 255), -1)
    elif cursor_active:
        msg, col = "Gaze Cursor Active  |  [N] Reset neutral  [R] Reset all  [Q] Quit", (80, 220, 80)
    elif cal.active:
        left = len(CAL_GRID) - cal.point_idx
        msg, col = f"Calibrating — look at dot, press SPACE  ({left} left)", (80, 180, 255)
    else:
        msg, col = "Press SPACE to start calibration  |  [Q] Quit", (200, 200, 200)
    cv2.putText(frame, msg, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)

def draw_direction_hint(frame, sdx, sdy, w, h):
    """Draw a subtle arrow showing which way the cursor is moving."""
    hints = []
    if abs(sdx) > DEAD_ZONE_X:
        hints.append('◀' if sdx < 0 else '▶')
    if abs(sdy) > DEAD_ZONE_Y:
        hints.append('▲' if sdy < 0 else '▼')
    if hints:
        label = ' '.join(hints)
        cv2.putText(frame, label, (w - 80, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)

# ── Gaze-Driven Cursor ─────────────────────────────────────────────────────────
# Tuning knobs — adjust to taste
CURSOR_SPEED      = 1000   # max pixels/second at full deflection
DEAD_ZONE_X       = 0.15  # horizontal dead zone (smaller = triggers on subtler movements)
DEAD_ZONE_Y       = 0.15  # vertical dead zone
SMOOTH_ALPHA      = 0.35  # EMA smoothing — higher = snappier, lower = smoother but sluggish
NEUTRAL_SAMPLES   = 45    # frames to average for neutral baseline (~1.5 sec)

# ── Calibration ────────────────────────────────────────────────────────────────
CAL_GRID = [(cx, cy)
            for cy in (0.15, 0.50, 0.85)
            for cx in (0.15, 0.50, 0.85)]

# ── Virtual Keyboard ───────────────────────────────────────────────────────────
# 4 rows: numbers/symbols, QWERTY, ASDF, ZXCV + specials
KB_ROWS = [
    ['1','2','3','4','5','6','7','8','9','0','.'],
    ['Q','W','E','R','T','Y','U','I','O','P'],
    ['A','S','D','F','G','H','J','K','L','ENT'],
    ['Z','X','C','V','B','N','M',',','<-','SPC'],
]

# The keyboard fills the full frame (set dynamically in VirtualKeyboard.__init__)
KEY_MARGIN     = 6          # gap between keys in px
DWELL_SECS     = 1.0        # seconds of gaze to trigger a key
KB_ALPHA       = 0.35       # key background opacity (lower = more transparent)
PANEL_ALPHA    = 0.18       # overall dark tint over webcam feed
TEXT_BAR_H     = 48         # height of the typed-text bar at the bottom

# Colours (BGR)
COL_KEY_BG     = (50,  50,  50)
COL_KEY_BORDER = (100, 100, 100)
COL_KEY_HOVER  = (200, 130,  40)    # warm amber highlight
COL_KEY_DWELL  = (60,  220,  60)    # green when about to fire
COL_KEY_FLASH  = (255, 255, 255)
COL_KEY_TEXT   = (240, 240, 240)
COL_ARC        = (0,   240, 180)
COL_TEXT_BAR   = (20,  20,  20)
COL_TEXT_FG    = (200, 230, 255)