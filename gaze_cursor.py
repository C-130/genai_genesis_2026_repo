"""
Gaze Cursor Module
"""

import cv2
import numpy as np
import time

from utils import CURSOR_SPEED, DEAD_ZONE_X, DEAD_ZONE_Y, SMOOTH_ALPHA, NEUTRAL_SAMPLES

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