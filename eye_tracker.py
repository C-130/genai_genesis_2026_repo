"""
Eye Tracker App — OpenCV + MediaPipe Tasks API
================================================
Compatible with mediapipe >= 0.10.x (uses FaceLandmarker Tasks API,
NOT the deprecated mp.solutions interface).

SETUP:
  pip install opencv-python mediapipe numpy

  The script auto-downloads the face_landmarker.task model (~7 MB)
  on first run and caches it next to this file.

CONTROLS:
  [SPACE]  — Start / confirm each calibration point
  [R]      — Reset calibration
  [Q/ESC]  — Quit

CALIBRATION:
  Look at each red pulsing dot, then press SPACE.
  After all 9 points the cyan gaze dot activates automatically.
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

# ── Gaze-Driven Cursor ─────────────────────────────────────────────────────────
# Tuning knobs — adjust to taste
CURSOR_SPEED      = 400   # max pixels/second at full deflection
DEAD_ZONE_X       = 0.12  # horizontal dead zone
DEAD_ZONE_Y       = 0.12  # vertical dead zone — raised; baseline subtraction handles drift
SMOOTH_ALPHA      = 0.20  # EMA smoothing for raw offset
NEUTRAL_SAMPLES   = 45    # frames to average for neutral baseline (~1.5 sec)

class GazeCursor:
    """
    Moves a cursor by integrating gaze direction over time.
    On startup it spends ~1.5 sec sampling the resting iris position and
    subtracts that baseline from every subsequent reading, so neutral gaze
    produces zero offset and the cursor stays still.
    """
    def __init__(self, frame_w, frame_h):
        self.fw  = frame_w
        self.fh  = frame_h
        self.x   = float(frame_w  // 2)
        self.y   = float(frame_h  // 2)
        self._sdx = 0.0
        self._sdy = 0.0
        self._last_t = time.time()

        # Neutral baseline
        self._baseline_dx  = 0.0
        self._baseline_dy  = 0.0
        self._cal_buf_dx   = []
        self._cal_buf_dy   = []
        self.calibrated    = False   # True once baseline is established

    def reset_baseline(self):
        self._cal_buf_dx = []
        self._cal_buf_dy = []
        self.calibrated  = False
        self._baseline_dx = 0.0
        self._baseline_dy = 0.0

    @property
    def calibrating(self):
        return not self.calibrated

    def update(self, raw_dx, raw_dy):
        """Feed raw normalised iris offset each frame; moves cursor."""
        now = time.time()
        dt  = now - self._last_t
        self._last_t = now
        dt = min(dt, 0.05)

        # ── Phase 1: collect neutral baseline ─────────────────────────────
        if not self.calibrated:
            self._cal_buf_dx.append(raw_dx)
            self._cal_buf_dy.append(raw_dy)
            if len(self._cal_buf_dx) >= NEUTRAL_SAMPLES:
                self._baseline_dx = float(np.mean(self._cal_buf_dx))
                self._baseline_dy = float(np.mean(self._cal_buf_dy))
                self.calibrated   = True
            return   # don't move cursor yet

        # ── Phase 2: subtract baseline then apply dead zone ───────────────
        adj_dx = raw_dx - self._baseline_dx
        adj_dy = raw_dy - self._baseline_dy

        # EMA smooth
        self._sdx += SMOOTH_ALPHA * (adj_dx - self._sdx)
        self._sdy += SMOOTH_ALPHA * (adj_dy - self._sdy)

        # Dead zone
        vx = self._sdx if abs(self._sdx) > DEAD_ZONE_X else 0.0
        vy = self._sdy if abs(self._sdy) > DEAD_ZONE_Y else 0.0

        def scale(v, dz):
            if v == 0:
                return 0.0
            sign = 1 if v > 0 else -1
            mag  = (abs(v) - dz) / max(1.0 - dz, 1e-6)
            return sign * min(mag, 1.0)

        self.x += scale(vx, DEAD_ZONE_X) * CURSOR_SPEED * dt
        self.y += scale(vy, DEAD_ZONE_Y) * CURSOR_SPEED * dt

        self.x = float(np.clip(self.x, 0, self.fw - 1))
        self.y = float(np.clip(self.y, 0, self.fh - 1))

    def pos(self):
        return int(self.x), int(self.y)

    def baseline_progress(self):
        """0.0 → 1.0 progress of neutral calibration."""
        return min(len(self._cal_buf_dx) / NEUTRAL_SAMPLES, 1.0)

    def draw(self, frame):
        cx, cy = self.pos()
        col = (120, 120, 120) if not self.calibrated else (0, 220, 255)
        cv2.circle(frame, (cx, cy), 18, col, 2)
        cv2.circle(frame, (cx, cy),  5, col, -1)
        cv2.line(frame, (cx-26, cy), (cx-20, cy), col, 1)
        cv2.line(frame, (cx+20, cy), (cx+26, cy), col, 1)
        cv2.line(frame, (cx, cy-26), (cx, cy-20), col, 1)
        cv2.line(frame, (cx, cy+20), (cx, cy+26), col, 1)

# ── Calibration ────────────────────────────────────────────────────────────────
CAL_GRID = [(cx, cy)
            for cy in (0.15, 0.50, 0.85)
            for cx in (0.15, 0.50, 0.85)]

class Calibrator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.point_idx    = 0
        self.gaze_samples = []
        self.screen_pts   = []
        self.gaze_pts     = []
        self.matrix       = None
        self.active       = False
        self.done         = False

    def start(self):
        self.reset()
        self.active = True

    def current_target(self, sw, sh):
        if self.point_idx >= len(CAL_GRID):
            return None
        fx, fy = CAL_GRID[self.point_idx]
        return int(fx * sw), int(fy * sh)

    def collect(self, rx, ry):
        if self.active and not self.done:
            self.gaze_samples.append((rx, ry))

    def confirm_point(self, sw, sh):
        if not self.active or self.done or len(self.gaze_samples) < 5:
            return False
        mrx = float(np.mean([s[0] for s in self.gaze_samples]))
        mry = float(np.mean([s[1] for s in self.gaze_samples]))
        tx, ty = self.current_target(sw, sh)
        self.screen_pts.append([tx, ty])
        self.gaze_pts.append([mrx, mry])
        self.gaze_samples = []
        self.point_idx += 1
        if self.point_idx >= len(CAL_GRID):
            self._fit()
            return True
        return False

    def _fit(self):
        src = np.array(self.gaze_pts,   dtype=np.float32)
        dst = np.array(self.screen_pts, dtype=np.float32)
        M, _ = cv2.estimateAffinePartial2D(src, dst)
        if M is None:
            sx = (dst[:,0].max()-dst[:,0].min()) / max(src[:,0].max()-src[:,0].min(), 1e-6)
            sy = (dst[:,1].max()-dst[:,1].min()) / max(src[:,1].max()-src[:,1].min(), 1e-6)
            tx = dst[:,0].mean() - sx*src[:,0].mean()
            ty = dst[:,1].mean() - sy*src[:,1].mean()
            M  = np.array([[sx,0,tx],[0,sy,ty]], dtype=np.float32)
        self.matrix = M
        self.done   = True
        self.active = False

    def map_gaze(self, rx, ry):
        if self.matrix is None:
            return None
        pt     = np.array([[[rx, ry]]], dtype=np.float32)
        mapped = cv2.transform(pt, self.matrix)
        return int(mapped[0,0,0]), int(mapped[0,0,1])

# ── Virtual Keyboard ───────────────────────────────────────────────────────────
# 4 rows: numbers/symbols, QWERTY, ASDF, ZXCV + specials
KB_ROWS = [
    ['1','2','3','4','5','6','7','8','9','0','-','.'],
    ['Q','W','E','R','T','Y','U','I','O','P'],
    ['A','S','D','F','G','H','J','K','L',';'],
    ['Z','X','C','V','B','N','M',',','⌫','SPC'],
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


class VirtualKeyboard:
    """Full-screen transparent keyboard overlay with gaze-dwell typing."""

    def __init__(self, frame_w, frame_h):
        self.frame_w   = frame_w
        self.frame_h   = frame_h
        # Keyboard fills everything except the bottom text bar and top HUD
        self.kb_top    = 40                          # below HUD strip
        self.kb_bot    = frame_h - TEXT_BAR_H        # above text bar
        self.kb_h      = self.kb_bot - self.kb_top
        self.keys      = []
        self.typed     = ""
        self.hover_key = None
        self.dwell_key = None
        self.dwell_t   = 0.0
        self.flash_key = None
        self.flash_t   = 0.0
        self._build_layout()

    def _build_layout(self):
        self.keys = []
        n_rows = len(KB_ROWS)
        row_h  = (self.kb_h - (n_rows + 1) * KEY_MARGIN) // n_rows

        for row_i, row in enumerate(KB_ROWS):
            n_keys = len(row)
            # Base key width — divide full width equally
            total_m = (n_keys + 1) * KEY_MARGIN
            key_w   = (self.frame_w - total_m) // n_keys
            y       = self.kb_top + KEY_MARGIN + row_i * (row_h + KEY_MARGIN)

            for col_i, label in enumerate(row):
                x = KEY_MARGIN + col_i * (key_w + KEY_MARGIN)
                kw = key_w
                # Give SPC extra width (absorbs the gap after it)
                if label == 'SPC':
                    kw = key_w * 2 + KEY_MARGIN
                self.keys.append((label, x, y, kw, row_h))

    def key_at(self, gx, gy):
        """Return label of key at screen position (gx, gy), or None."""
        for label, x, y, kw, kh in self.keys:
            if x <= gx < x + kw and y <= gy < y + kh:
                return label
        return None

    def update_gaze(self, gx, gy):
        """Call every frame. Returns typed char/action string if key fired, else None."""
        label = self.key_at(gx, gy)
        now   = time.time()
        typed = None

        if label != self.dwell_key:
            self.dwell_key = label
            self.dwell_t   = now

        if label is not None:
            self.hover_key = label
            if now - self.dwell_t >= DWELL_SECS and label != self.flash_key:
                typed          = self._fire(label)
                self.flash_key = label
                self.flash_t   = now
                self.dwell_t   = now + 0.5   # refractory period
        else:
            self.hover_key = None

        if self.flash_key and now - self.flash_t > 0.25:
            self.flash_key = None

        return typed

    def _fire(self, label):
        if label == '⌫':
            self.typed = self.typed[:-1]
        elif label == 'SPC':
            self.typed += ' '
        else:
            self.typed += label
        return label

    def draw(self, frame):
        """Render full-screen transparent keyboard onto frame in-place."""
        h_f, w_f = frame.shape[:2]
        now = time.time()

        # ── 1. Subtle dark tint over the whole keyboard area ───────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, self.kb_top), (w_f, self.kb_bot), (0, 0, 0), -1)
        cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1 - PANEL_ALPHA, 0, frame)

        # ── 2. Keys ────────────────────────────────────────────────────────
        for label, kx, ky, kw, kh in self.keys:
            # Per-key transparent fill
            if label == self.flash_key:
                key_col = COL_KEY_FLASH
                alpha   = 0.9
            elif label == self.hover_key:
                frac    = min((now - self.dwell_t) / DWELL_SECS, 1.0)
                # Interpolate amber → green as dwell progresses
                key_col = (
                    int(COL_KEY_HOVER[0] + (COL_KEY_DWELL[0]-COL_KEY_HOVER[0])*frac),
                    int(COL_KEY_HOVER[1] + (COL_KEY_DWELL[1]-COL_KEY_HOVER[1])*frac),
                    int(COL_KEY_HOVER[2] + (COL_KEY_DWELL[2]-COL_KEY_HOVER[2])*frac),
                )
                alpha   = 0.55
            else:
                key_col = COL_KEY_BG
                alpha   = KB_ALPHA

            # Blend key rectangle onto frame
            key_overlay = frame.copy()
            cv2.rectangle(key_overlay, (kx, ky), (kx+kw, ky+kh), key_col, -1)
            cv2.addWeighted(key_overlay, alpha, frame, 1 - alpha, 0, frame)

            # Thin border (always fully opaque for clarity)
            border_col = (200,200,200) if label == self.flash_key else COL_KEY_BORDER
            cv2.rectangle(frame, (kx, ky), (kx+kw, ky+kh), border_col, 1)

            # Dwell progress arc
            if label == self.hover_key and label != self.flash_key:
                frac = min((now - self.dwell_t) / DWELL_SECS, 1.0)
                if frac > 0.01:
                    cx_ = kx + kw // 2
                    cy_ = ky + kh // 2
                    rad = min(kw, kh) // 2 - 4
                    if rad > 4:
                        cv2.ellipse(frame, (cx_, cy_), (rad, rad),
                                    -90, 0, int(360 * frac), COL_ARC, 3)

            # Key label — larger font since keys are big
            disp   = 'SPACE' if label == 'SPC' else label
            font   = cv2.FONT_HERSHEY_SIMPLEX
            fscale = 0.5 if len(disp) > 1 else 0.65
            thick  = 1 if len(disp) > 1 else 2
            (tw, th), _ = cv2.getTextSize(disp, font, fscale, thick)
            tx = kx + (kw - tw) // 2
            ty = ky + (kh + th) // 2
            tcol = (20, 20, 20) if label == self.flash_key else COL_KEY_TEXT
            cv2.putText(frame, disp, (tx, ty), font, fscale, tcol, thick, cv2.LINE_AA)

        # ── 3. Text output bar at the very bottom ──────────────────────────
        bar_y = self.kb_bot
        bar_overlay = frame.copy()
        cv2.rectangle(bar_overlay, (0, bar_y), (w_f, h_f), COL_TEXT_BAR, -1)
        cv2.addWeighted(bar_overlay, 0.80, frame, 0.20, 0, frame)

        # Thin separator line
        cv2.line(frame, (0, bar_y), (w_f, bar_y), (80, 80, 80), 1)

        # Show last N chars with blinking cursor
        max_chars = 90
        display   = self.typed[-max_chars:] if len(self.typed) > max_chars else self.typed
        cursor    = '|' if int(now * 2) % 2 == 0 else ' '
        cv2.putText(frame, display + cursor,
                    (14, bar_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_TEXT_FG, 1, cv2.LINE_AA)


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

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ensure_model()

    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = FaceLandmarkerOptions(
        base_options=base_opts,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    detector = FaceLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Cannot open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📷  Camera: {w}×{h}")
    print("   Gaze direction moves the cursor — look left/right/up/down.")
    print("   Press SPACE to start keyboard calibration (for dwell typing).")
    print("   Press Q or ESC to quit.\n")

    # Calibration is still used to anchor the keyboard hit-zones (optional)
    cal          = Calibrator()

    # Direction-based cursor — works immediately, no calibration required
    cursor       = GazeCursor(w, h)

    # Virtual keyboard
    kb           = VirtualKeyboard(frame_w=w, frame_h=h)

    # Smoothed raw gaze offsets (for direction hint display)
    _sdx = 0.0
    _sdy = 0.0

    t0           = time.time()
    GRACE_FRAMES = 60
    frame_count  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)

        raw_dx, raw_dy = 0.0, 0.0   # gaze direction this frame
        gaze_rx, gaze_ry = 0.5, 0.5  # for calibration dot collection

        if result.face_landmarks:
            lm = result.face_landmarks[0]

            l_iris = iris_center(lm, LEFT_IRIS,  w, h)
            r_iris = iris_center(lm, RIGHT_IRIS, w, h)

            lx0, lx1, ly0, ly1 = eye_bounds(lm, LEFT_EYE_CORNERS,  LEFT_EYE_LIDS,  w, h)
            rx0, rx1, ry0, ry1 = eye_bounds(lm, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS, w, h)

            # Stable vertical anchors: inner eye corner Y (doesn't move with lid)
            l_corner_y = lm[LEFT_EYE_INNER].y  * h
            r_corner_y = lm[RIGHT_EYE_INNER].y * h

            # Direction offsets from each eye centre, averaged
            l_dx, l_dy = gaze_offset(l_iris, lx0, lx1, ly0, ly1, l_corner_y)
            r_dx, r_dy = gaze_offset(r_iris, rx0, rx1, ry0, ry1, r_corner_y)
            raw_dx = (l_dx + r_dx) / 2
            raw_dy = (l_dy + r_dy) / 2

            # Also compute ratio for optional calibration collection
            l_rx, l_ry = gaze_ratio(l_iris, lx0, lx1, ly0, ly1)
            r_rx, r_ry = gaze_ratio(r_iris, rx0, rx1, ry0, ry1)
            gaze_rx    = (l_rx + r_rx) / 2
            gaze_ry    = (l_ry + r_ry) / 2

            if cal.active and not cal.done:
                cal.collect(gaze_rx, gaze_ry)

            draw_eye_box(frame, l_iris, lx0, lx1, ly0, ly1)
            draw_eye_box(frame, r_iris, rx0, rx1, ry0, ry1)

        # ── Update & draw cursor (always active) ──────────────────────────
        cursor.update(raw_dx, raw_dy)
        cx, cy = cursor.pos()

        # ── Smooth direction signal for hint display ───────────────────────
        _sdx += SMOOTH_ALPHA * (raw_dx - _sdx)
        _sdy += SMOOTH_ALPHA * (raw_dy - _sdy)

        # ── Keyboard overlay (shown once calibration done OR always?) ──────
        # Show keyboard always after first SPACE press — calibration just
        # improves the initial layout offset; cursor works without it.
        kb.update_gaze(cx, cy)
        kb.draw(frame)

        # Draw cursor on top of keyboard
        cursor.draw(frame)

        # Direction hint arrows
        draw_direction_hint(frame, _sdx, _sdy, w, h)

        # ── Calibration dots ───────────────────────────────────────────────
        if cal.active and not cal.done:
            target = cal.current_target(w, h)
            if target:
                pulse = (np.sin((time.time()-t0)*4)+1)/2
                draw_cal_dot(frame, target[0], target[1], pulse)
                n = len(cal.gaze_samples)
                cv2.putText(frame, f"Samples: {n}  — press SPACE when ready",
                            (target[0]-150, target[1]+50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        for i, (fx, fy) in enumerate(CAL_GRID):
            if i < cal.point_idx:
                cv2.circle(frame, (int(fx*w), int(fy*h)), 6, (0, 200, 80), -1)

        # ── HUD ───────────────────────────────────────────────────────────
        draw_hud(frame, cal, w, h, cursor_active=not cal.active, cursor=cursor)
        cv2.putText(frame,
                    f"gaze offset  dx:{raw_dx:+.2f}  dy:{raw_dy:+.2f}",
                    (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140,140,140), 1, cv2.LINE_AA)
        cv2.putText(frame,
                    f"cursor  x:{cx}  y:{cy}",
                    (10, h - 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140,140,140), 1, cv2.LINE_AA)

        cv2.imshow("Eye Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27) and frame_count > GRACE_FRAMES:
            break
        elif key == ord(' '):
            if not cal.active and not cal.done:
                cal.start()
                print("▶  Calibration started.")
            elif cal.active and not cal.done:
                done = cal.confirm_point(w, h)
                if done:
                    print("✅  Calibration complete!")
                else:
                    print(f"   Point {cal.point_idx}/{len(CAL_GRID)} recorded.")
        elif key == ord('r'):
            cal.reset()
            cursor.reset_baseline()
            cursor.x = float(w // 2)
            cursor.y = float(h // 2)
            print("🔄  Reset — look straight ahead for neutral calibration.")
        elif key == ord('n'):
            cursor.reset_baseline()
            print("🔄  Neutral baseline reset — look straight ahead.")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("👋  Done.")

if __name__ == "__main__":
    main()