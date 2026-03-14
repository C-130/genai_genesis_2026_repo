"""
Virtual Keyboard Module
"""

import cv2
import time

from utils import KB_ROWS, KEY_MARGIN, DWELL_SECS, KB_ALPHA, PANEL_ALPHA, TEXT_BAR_H, COL_KEY_BG, COL_KEY_BORDER, COL_KEY_HOVER, COL_KEY_DWELL, COL_KEY_FLASH, COL_KEY_TEXT, COL_ARC, COL_TEXT_BAR, COL_TEXT_FG

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
        """Call every frame to update which key is being looked at."""
        self.hover_key = self.key_at(gx, gy)

        # Clear flash after 0.25 s
        now = time.time()
        if self.flash_key and now - self.flash_t > 0.25:
            self.flash_key = None

    def press(self):
        """Call when the user presses the up arrow — fires the hovered key."""
        if self.hover_key is None:
            return None
        typed          = self._fire(self.hover_key)
        self.flash_key = self.hover_key
        self.flash_t   = time.time()
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
            if label == self.flash_key:
                key_col = COL_KEY_FLASH
                alpha   = 0.9
            elif label == self.hover_key:
                key_col = COL_KEY_HOVER
                alpha   = 0.65
            else:
                key_col = COL_KEY_BG
                alpha   = KB_ALPHA

            # Blend key rectangle onto frame
            key_overlay = frame.copy()
            cv2.rectangle(key_overlay, (kx, ky), (kx+kw, ky+kh), key_col, -1)
            cv2.addWeighted(key_overlay, alpha, frame, 1 - alpha, 0, frame)

            # Thin border
            border_col = (200,200,200) if label == self.flash_key else COL_KEY_BORDER
            cv2.rectangle(frame, (kx, ky), (kx+kw, ky+kh), border_col, 1)

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