"""
Gaze Cursor Module
"""

import cv2
import numpy as np
import time

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE    = 0
    _PYAUTO = True
except ImportError:
    _PYAUTO = False

from utils import CURSOR_SPEED, DEAD_ZONE_X, DEAD_ZONE_Y, SMOOTH_ALPHA, NEUTRAL_SAMPLES



class GazeCursor:
    """
    Moves a cursor by integrating gaze direction over time.

    Two modes:
      frame_mode  (default) — cursor constrained to the cv2 window frame
      browse_mode           — cursor drives the real OS mouse via pyautogui,
                              mapped to full screen coordinates
    """

    def __init__(self, frame_w, frame_h):
        self.fw          = frame_w
        self.fh          = frame_h
        self.x           = float(frame_w  // 2)
        self.y           = float(frame_h  // 2)
        self._sdx        = 0.0
        self._sdy        = 0.0
        self._last_t     = time.time()
        self.browse_mode = False   # toggled by eye_tracker.py

        # Screen dimensions for browse mode
        if _PYAUTO:
            import pyautogui as _pg
            self._sw, self._sh = _pg.size()
        else:
            self._sw, self._sh = frame_w, frame_h

        # Neutral baseline
        self._baseline_dx = 0.0
        self._baseline_dy = 0.0
        self._cal_buf_dx  = []
        self._cal_buf_dy  = []
        self.calibrated   = False


    def reset_baseline(self):
        self._cal_buf_dx  = []
        self._cal_buf_dy  = []
        self.calibrated   = False

        self._baseline_dx = 0.0
        self._baseline_dy = 0.0

    @property
    def calibrating(self):
        return not self.calibrated

    def update(self, raw_dx, raw_dy):
        now = time.time()
        dt  = min(now - self._last_t, 0.05)
        self._last_t = now

        if not self.calibrated:
            self._cal_buf_dx.append(raw_dx)
            self._cal_buf_dy.append(raw_dy)
            if len(self._cal_buf_dx) >= NEUTRAL_SAMPLES:
                self._baseline_dx = float(np.mean(self._cal_buf_dx))
                self._baseline_dy = float(np.mean(self._cal_buf_dy))
                self.calibrated   = True
            return

        adj_dx = raw_dx - self._baseline_dx
        adj_dy = raw_dy - self._baseline_dy
        self._sdx += SMOOTH_ALPHA * (adj_dx - self._sdx)
        self._sdy += SMOOTH_ALPHA * (adj_dy - self._sdy)

        def scale(v, dz):
            if abs(v) <= dz:
                return 0.0
            sign = 1 if v > 0 else -1
            mag  = (abs(v) - dz) / max(1.0 - dz, 1e-6)
            return sign * min(mag, 1.0)

        if self.browse_mode:
            # Drive full screen
            self.x += scale(self._sdx, DEAD_ZONE_X) * CURSOR_SPEED * dt * (self._sw / self.fw)
            self.y += scale(self._sdy, DEAD_ZONE_Y) * CURSOR_SPEED * dt * (self._sh / self.fh)
            self.x  = float(np.clip(self.x, 0, self._sw - 1))
            self.y  = float(np.clip(self.y, 0, self._sh - 1))
            if _PYAUTO:
                import pyautogui as _pg
                _pg.moveTo(int(self.x), int(self.y))
        else:
            self.x += scale(self._sdx, DEAD_ZONE_X) * CURSOR_SPEED * dt
            self.y += scale(self._sdy, DEAD_ZONE_Y) * CURSOR_SPEED * dt
            self.x  = float(np.clip(self.x, 0, self.fw - 1))
            self.y  = float(np.clip(self.y, 0, self.fh - 1))

    def pos(self):
        """Frame-space position for drawing and hit-testing within the cv2 window."""
        if self.browse_mode:
            # Map screen coords back to frame coords for drawing
            fx = int(self.x / self._sw * self.fw)
            fy = int(self.y / self._sh * self.fh)
            return fx, fy
        return int(self.x), int(self.y)

    def baseline_progress(self):
        return min(len(self._cal_buf_dx) / NEUTRAL_SAMPLES, 1.0)

    def draw(self, frame):
        """Always call this LAST so cursor renders on top of everything."""
        cx, cy = self.pos()

        if not self.calibrated:
            col = (120, 120, 120)
        elif self.browse_mode:
            col = (0, 200, 80)   # green in browse mode
        else:
            col = (0, 220, 255)  # cyan in normal mode

        # Outer glow ring (slightly transparent)
        ov = frame.copy()
        cv2.circle(ov, (cx, cy), 20, col, 2)
        cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)

        # Inner filled dot — fully opaque, always on top
        cv2.circle(frame, (cx, cy), 6, col, -1)
        cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 1)

        # Crosshair lines
        for x1, y1, x2, y2 in [
            (cx-28, cy, cx-14, cy),
            (cx+14, cy, cx+28, cy),
            (cx, cy-28, cx, cy-14),
            (cx, cy+14, cx, cy+28),
        ]:
            cv2.line(frame, (x1,y1), (x2,y2), col, 1, cv2.LINE_AA)

        # Browse mode indicator
        if self.browse_mode:
            cv2.putText(frame, "BROWSE", (cx+14, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)