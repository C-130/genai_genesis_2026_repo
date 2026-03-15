"""
Intent Overlay
==============
Full-screen tile grid for high-level intent selection.

Interaction model matches the existing keyboard:
  - Gaze cursor hovers over a tile
  - User presses UP ARROW to confirm (same key as keyboard typing)

This means no new input method needs to be learned.
"""

import cv2
import numpy as np
import time

# (id, display label, subtitle)
INTENTS = [
    ("email",     "Send email",       "draft + send via Gmail"),
    ("sms",       "Send message",     "SMS or WhatsApp"),
    ("search",    "Search web",       "find + read aloud"),
    ("media",     "Media control",    "play / pause / next"),
    ("emergency", "Emergency alert",  "alert all contacts now"),
    ("speak",     "Speak sentence",   "read my typed text aloud"),
]

# BGR colours per intent
INTENT_COLOURS = {
    "email":     (160,  70,  30),
    "sms":       ( 30, 120, 160),
    "search":    ( 40, 140,  60),
    "media":     (130,  40, 140),
    "emergency": ( 20,  20, 180),
    "speak":     ( 80, 110,  40),
}


class IntentOverlay:
    """
    Renders a 2×3 grid of large intent tiles over the camera frame.
    Hover is tracked by gaze cursor position.
    Selection fires on UP ARROW (caller calls .press() on that keydown).
    """

    def __init__(self, frame_w, frame_h):
        self.fw      = frame_w
        self.fh      = frame_h
        self.visible = False
        self._hover  = None     # intent_id currently under cursor
        self._tiles  = []       # [(intent_id, x, y, w, h), ...]
        self._build_tiles()

    # ── Public ─────────────────────────────────────────────────────────────────

    def show(self):
        self.visible = True
        self._hover  = None

    def hide(self):
        self.visible = False
        self._hover  = None

    def update_gaze(self, gx, gy):
        """Call every frame with frame-space cursor position."""
        if not self.visible:
            return
        hit = None
        for intent_id, x, y, w, h in self._tiles:
            if x <= gx <= x + w and y <= gy <= y + h:
                hit = intent_id
                break
        self._hover = hit

    def press(self):
        """
        Call when UP ARROW is pressed.
        Returns the hovered intent_id, or None if nothing hovered.
        Hides the overlay on a successful selection.
        """
        if not self.visible or self._hover is None:
            return None
        selected    = self._hover
        self.hide()
        return selected

    @property
    def hover(self):
        return self._hover

    def draw(self, frame):
        if not self.visible:
            return

        # Dark background overlay
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (self.fw, self.fh), (8, 10, 18), -1)
        cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)

        for intent_id, x, y, w, h in self._tiles:
            label, subtitle = next(
                (l, s) for i, l, s in INTENTS if i == intent_id
            )
            colour    = INTENT_COLOURS[intent_id]
            is_hover  = (intent_id == self._hover)
            alpha     = 0.80 if is_hover else 0.45

            tile_ov = frame.copy()
            cv2.rectangle(tile_ov, (x, y), (x+w, y+h), colour, -1)
            cv2.addWeighted(tile_ov, alpha, frame, 1 - alpha, 0, frame)

            border_col = (255, 255, 255) if is_hover else (100, 100, 110)
            border_w   = 2 if is_hover else 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), border_col, border_w)

            # Main label
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.85, 2)
            cv2.putText(frame, label,
                        (x + (w - tw)//2, y + h//2 - 8),
                        font, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

            # Subtitle
            (sw, _), _ = cv2.getTextSize(subtitle, font, 0.42, 1)
            cv2.putText(frame, subtitle,
                        (x + (w - sw)//2, y + h//2 + 20),
                        font, 0.42, (190, 190, 190), 1, cv2.LINE_AA)

        # Header + instruction
        cv2.putText(frame,
                    "Look at an action  —  press UP ARROW to select  —  ESC to cancel",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1, cv2.LINE_AA)

    # ── Layout ─────────────────────────────────────────────────────────────────

    def _build_tiles(self):
        cols, rows = 3, 2
        pad    = 14
        tile_w = (self.fw - pad * (cols + 1)) // cols
        tile_h = (self.fh - pad * (rows + 1) - 44) // rows   # 44px for header
        self._tiles = []
        for i, (intent_id, _, _) in enumerate(INTENTS):
            col = i % cols
            row = i // cols
            x   = pad + col * (tile_w + pad)
            y   = 44 + pad + row * (tile_h + pad)
            self._tiles.append((intent_id, x, y, tile_w, tile_h))
