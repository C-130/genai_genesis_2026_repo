"""
Clarify Overlay
===============
Shows a question + up to 4 large answer tiles.

Same input model as the keyboard and intent overlay:
  - Gaze cursor hovers over an answer tile
  - User presses UP ARROW to confirm

The agent calls .ask(question, options) from a background thread.
The main loop calls .update_gaze() and .press() each frame/keydown.
When an answer is selected, the overlay calls back into the agent
via a threading.Event so the agent thread can unblock.
"""

import cv2
import threading


# BGR palette for answer tiles (cycles if more than 4 options)
TILE_COLOURS = [
    (50,  90, 140),   # blue
    (50, 120,  60),   # green
    (130, 60,  50),   # red-ish (used for Cancel)
    (100, 70, 130),   # purple
]


class ClarifyOverlay:
    def __init__(self, frame_w, frame_h):
        self.fw       = frame_w
        self.fh       = frame_h
        self.visible  = False
        self._question = ""
        self._options  = []
        self._tiles    = []
        self._hover    = None   # index into _options
        self._event    = None   # threading.Event set when answered
        self._answer   = None

    # ── Agent-facing API (called from background thread) ───────────────────────

    def ask(self, question, options):
        """
        Block the calling thread until the user selects an answer.
        Returns the selected option string.
        """
        self._question = question
        self._options  = list(options)
        self._hover    = None
        self._answer   = None
        self._event    = threading.Event()
        self._build_tiles()
        self.visible   = True
        self._event.wait(timeout=120)   # 2-minute timeout
        self.visible   = False
        return self._answer or options[-1]  # default to last on timeout

    # ── Main-loop-facing API ───────────────────────────────────────────────────

    def update_gaze(self, gx, gy):
        if not self.visible:
            return
        hit = None
        for i, (x, y, w, h) in enumerate(self._tiles):
            if x <= gx <= x + w and y <= gy <= y + h:
                hit = i
                break
        self._hover = hit

    def press(self):
        """Call on UP ARROW. Returns selected answer string or None."""
        if not self.visible or self._hover is None:
            return None
        answer        = self._options[self._hover]
        self._answer  = answer
        if self._event:
            self._event.set()
        return answer

    def draw(self, frame):
        if not self.visible:
            return

        # Dark overlay
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (self.fw, self.fh), (8, 10, 18), -1)
        cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)

        # Question — word-wrap at frame width
        font  = cv2.FONT_HERSHEY_SIMPLEX
        words = self._question.split()
        lines, line = [], ""
        for word in words:
            test = (line + " " + word).strip()
            (tw, _), _ = cv2.getTextSize(test, font, 0.72, 2)
            if tw > self.fw - 80:
                lines.append(line)
                line = word
            else:
                line = test
        if line:
            lines.append(line)

        y_q = 70
        for ln in lines:
            (tw, th), _ = cv2.getTextSize(ln, font, 0.72, 2)
            cv2.putText(frame, ln, ((self.fw - tw)//2, y_q),
                        font, 0.72, (230, 225, 255), 2, cv2.LINE_AA)
            y_q += th + 12

        # Answer tiles
        for i, (x, y, w, h) in enumerate(self._tiles):
            label     = self._options[i]
            is_hover  = (i == self._hover)
            is_cancel = label.lower() in ("cancel", "no", "back")
            base      = (55, 30, 30) if is_cancel else TILE_COLOURS[i % len(TILE_COLOURS)]
            if is_hover:
                base = tuple(min(c + 55, 255) for c in base)

            tile_ov = frame.copy()
            cv2.rectangle(tile_ov, (x, y), (x+w, y+h), base, -1)
            cv2.addWeighted(tile_ov, 0.82, frame, 0.18, 0, frame)

            border = (255, 80, 80) if is_cancel else ((255,255,255) if is_hover else (90,100,120))
            cv2.rectangle(frame, (x, y), (x+w, y+h), border, 2 if is_hover else 1)

            (lw, lh), _ = cv2.getTextSize(label, font, 0.70, 2)
            cv2.putText(frame, label,
                        (x + (w - lw)//2, y + (h + lh)//2 - 4),
                        font, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

        # Instruction
        cv2.putText(frame,
                    "Look at an answer  —  press UP ARROW to confirm",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1, cv2.LINE_AA)

    # ── Layout ─────────────────────────────────────────────────────────────────

    def _build_tiles(self):
        n      = len(self._options)
        pad    = 16
        cols   = min(n, 4)
        tile_w = (self.fw - pad * (cols + 1)) // cols
        tile_h = min(120, (self.fh - 200 - pad * 2) // 1)
        y      = self.fh - tile_h - 60
        self._tiles = []
        for i in range(n):
            x = pad + i * (tile_w + pad)
            self._tiles.append((x, y, tile_w, tile_h))
