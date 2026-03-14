# ── keyboard.py ───────────────────────────────────────────────────────────────
# Full-screen transparent virtual keyboard overlay.
# Typing is triggered by pressing the UP ARROW key while the gaze cursor
# hovers over the desired key.

import cv2
import time

from utils import (
    KB_ROWS, KEY_MARGIN, KB_ALPHA, PANEL_ALPHA, TEXT_BAR_H,
    COL_KEY_BG, COL_KEY_BORDER, COL_KEY_HOVER,
    COL_KEY_FLASH, COL_KEY_TEXT, COL_TEXT_BAR, COL_TEXT_FG,
)


class VirtualKeyboard:
    """Full-screen transparent keyboard overlay with gaze + keypress typing."""

    def __init__(self, frame_w, frame_h):
        self.frame_w   = frame_w
        self.frame_h   = frame_h
        self.kb_top    = 40                     # below HUD strip
        self.kb_bot    = frame_h - TEXT_BAR_H   # above text bar
        self.kb_h      = self.kb_bot - self.kb_top
        self.keys      = []
        self.typed     = ""
        self.hover_key = None
        self.flash_key = None
        self.flash_t   = 0.0
        self._build_layout()

    # ── Layout ─────────────────────────────────────────────────────────────────

    def _build_layout(self):
        self.keys = []
        n_rows = len(KB_ROWS)
        row_h  = (self.kb_h - (n_rows + 1) * KEY_MARGIN) // n_rows

        for row_i, row in enumerate(KB_ROWS):
            n_keys  = len(row)
            total_m = (n_keys + 1) * KEY_MARGIN
            key_w   = (self.frame_w - total_m) // n_keys
            y       = self.kb_top + KEY_MARGIN + row_i * (row_h + KEY_MARGIN)

            for col_i, label in enumerate(row):
                x  = KEY_MARGIN + col_i * (key_w + KEY_MARGIN)
                kw = key_w * 2 + KEY_MARGIN if label == 'SPC' else key_w
                self.keys.append((label, x, y, kw, row_h))

    # ── Hit testing ────────────────────────────────────────────────────────────

    def key_at(self, gx, gy):
        """Return label of key at screen pos (gx, gy), or None."""
        for label, x, y, kw, kh in self.keys:
            if x <= gx < x + kw and y <= gy < y + kh:
                return label
        return None

    # ── Per-frame update ───────────────────────────────────────────────────────

    def update_gaze(self, gx, gy):
        """Update hover state from current cursor position."""
        self.hover_key = self.key_at(gx, gy)
        if self.flash_key and time.time() - self.flash_t > 0.25:
            self.flash_key = None

    # ── Key press ──────────────────────────────────────────────────────────────

    def press(self):
        """Fire the currently hovered key (call on UP ARROW keydown)."""
        if self.hover_key is None:
            return ''
        self._fire(self.hover_key)
        self.flash_key = self.hover_key
        self.flash_t   = time.time()
        return self.hover_key

    def _fire(self, label):
        if label == '<-':
            self.typed = self.typed[:-1]
        elif label == 'SPC':
            self.typed += ' '
        elif label == 'ENT':
            self.typed = ''
        else:
            self.typed += label

    # ── Drawing ────────────────────────────────────────────────────────────────

    def draw(self, frame):
        """Render keyboard overlay onto frame in-place."""
        h_f, w_f = frame.shape[:2]
        now = time.time()

        # 1. Faint dark tint so keys are readable against the webcam feed
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, self.kb_top), (w_f, self.kb_bot), (0, 0, 0), -1)
        cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1 - PANEL_ALPHA, 0, frame)

        # 2. Keys
        for label, kx, ky, kw, kh in self.keys:
            if label == self.flash_key:
                key_col, alpha = COL_KEY_FLASH, 0.9
            elif label == self.hover_key:
                key_col, alpha = COL_KEY_HOVER, 0.65
            else:
                key_col, alpha = COL_KEY_BG, KB_ALPHA

            key_ov = frame.copy()
            cv2.rectangle(key_ov, (kx, ky), (kx+kw, ky+kh), key_col, -1)
            cv2.addWeighted(key_ov, alpha, frame, 1 - alpha, 0, frame)

            border = (200, 200, 200) if label == self.flash_key else COL_KEY_BORDER
            cv2.rectangle(frame, (kx, ky), (kx+kw, ky+kh), border, 1)

            disp   = 'SPACE' if label == 'SPC' else label
            font   = cv2.FONT_HERSHEY_SIMPLEX
            fscale = 0.5 if len(disp) > 1 else 0.65
            thick  = 1 if len(disp) > 1 else 2
            (tw, th), _ = cv2.getTextSize(disp, font, fscale, thick)
            tx = kx + (kw - tw) // 2
            ty = ky + (kh + th) // 2
            tcol = (20, 20, 20) if label == self.flash_key else COL_KEY_TEXT
            cv2.putText(frame, disp, (tx, ty), font, fscale, tcol, thick, cv2.LINE_AA)

        # 3. Text output bar at the bottom
        bar_y = self.kb_bot
        bar_ov = frame.copy()
        cv2.rectangle(bar_ov, (0, bar_y), (w_f, h_f), COL_TEXT_BAR, -1)
        cv2.addWeighted(bar_ov, 0.80, frame, 0.20, 0, frame)
        cv2.line(frame, (0, bar_y), (w_f, bar_y), (80, 80, 80), 1)

        display = self.typed[-90:]
        blink   = '|' if int(now * 2) % 2 == 0 else ' '
        cv2.putText(frame, display + blink, (14, bar_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_TEXT_FG, 1, cv2.LINE_AA)