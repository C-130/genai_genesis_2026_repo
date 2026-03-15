# ── virtual_keyboard.py ───────────────────────────────────────────────────────
# Full-screen transparent virtual keyboard overlay.
# Typing is triggered by pressing the UP ARROW key while the gaze cursor
# hovers over the desired key.
#
# Additions:
#   • Suggestion bar rendered above the keyboard rows.
#     Call kb.set_suggestions(["word1", "word2", "word3"]) each frame.
#     Hovering a suggestion chip + UP ARROW types that word + a space.

import cv2
import time
from handle_email import open_email_draft, interpret_email_request

from utils import (
    KB_ROWS, KEY_MARGIN, KB_ALPHA, PANEL_ALPHA, TEXT_BAR_H,
    COL_KEY_BG, COL_KEY_BORDER, COL_KEY_HOVER,
    COL_KEY_FLASH, COL_KEY_TEXT, COL_TEXT_BAR, COL_TEXT_FG,
)

# Height reserved for the suggestion bar (sits just above the first key row)
SUGGEST_BAR_H = 48
SEND_BAR_H = 48
# Prefix used internally to distinguish suggestion keys from letter keys
_SUGG_PREFIX  = "__SUGG__"


class VirtualKeyboard:
    """Full-screen transparent keyboard overlay with gaze + keypress typing."""

    def __init__(self, frame_w, frame_h):
        self.frame_w   = frame_w
        self.frame_h   = frame_h
        # Leave room at top for HUD and suggestion bar
        self.kb_top    = 40 + SUGGEST_BAR_H        # shifted down
        self.kb_bot    = frame_h - TEXT_BAR_H
        self.kb_h      = self.kb_bot - self.kb_top
        self.keys      = []
        self.typed     = ""
        self.hover_key = None
        self.flash_key = None
        self.flash_t   = 0.0
        self.send_key = None
        # Suggestion state
        self._suggestions  = []   # list of up to 3 strings
        self._sugg_rects   = []   # [(label, x, y, w, h), …] rebuilt each frame
        self.upper = False # Upper case or lower case (toggle with 'CAP' key)
        self.email_status = ''
        self._build_layout()

    # ── Layout ─────────────────────────────────────────────────────────────────

    def _build_layout(self):
        send_key_w = (self.frame_w - 4 * KEY_MARGIN) // 3
        self.send_key = ('SEND', self.frame_w - send_key_w - KEY_MARGIN, 0, send_key_w, SEND_BAR_H)
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

    # ── Suggestions API ────────────────────────────────────────────────────────

    def set_suggestions(self, words):
        """Update the top-3 word suggestions shown above the keyboard."""
        self._suggestions = list(words[:3])

    def _build_sugg_rects(self):
        """Recompute suggestion chip geometry (called inside draw)."""
        self._sugg_rects = []
        if not self._suggestions:
            return

        n      = len(self._suggestions)
        margin = KEY_MARGIN
        bar_y  = 40                          # just below the HUD strip
        chip_h = SUGGEST_BAR_H - margin * 2
        total_w = self.frame_w - margin * (n + 1)
        chip_w  = total_w // n

        for i, word in enumerate(self._suggestions):
            x = margin + i * (chip_w + margin)
            y = bar_y + margin
            self._sugg_rects.append((_SUGG_PREFIX + word, x, y, chip_w, chip_h))

    # ── Hit testing ────────────────────────────────────────────────────────────

    def key_at(self, gx, gy):
        """Return label of key/chip at screen pos (gx, gy), or None."""
        # Check suggestion chips first
        for label, x, y, kw, kh in self._sugg_rects:
            if x <= gx < x + kw and y <= gy < y + kh:
                return label
        # Then regular keys
        for label, x, y, kw, kh in self.keys:
            if x <= gx < x + kw and y <= gy < y + kh:
                return label
        # Then send key
        _, x, y, kw, kh = self.send_key if self.send_key else (None, -1, -1, -1, -1)
        if x <= gx < x + kw and y <= gy < y + kh:
            return 'SEND'
        return None

    # ── Per-frame update ───────────────────────────────────────────────────────

    def update_gaze(self, gx, gy):
        """Update hover state from current cursor position."""
        self.hover_key = self.key_at(gx, gy)
        if self.flash_key and time.time() - self.flash_t > 0.25:
            self.flash_key = None

    # ── Key press ──────────────────────────────────────────────────────────────

    def press(self):
        """Fire the currently hovered key (call on UP ARROW keydown).
        Returns the typed string (word for suggestions, single char for keys).
        """
        self.email_status = ''
        if self.hover_key is None:
            return ''
        self._fire(self.hover_key)
        self.flash_key = self.hover_key
        self.flash_t   = time.time()
        if self.hover_key == 'SPC':
            return ' '
        if self.hover_key == 'SEND':
            try:
                email_data = interpret_email_request(self.typed)
                if len(email_data['address']) == 0:
                    self.email_status = 'Error: No recipient email address found.'
                else:
                    open_email_draft(email_data['address'], email_data['subject'], email_data['cc'], email_data['body'])
                    self.email_status = 'Email draft opened successfully.'
            except Exception as e:
                self.email_status = f'Error interpreting email request: {str(e)}'
        return self.hover_key

    def _fire(self, label):
        if label.startswith(_SUGG_PREFIX):
            word = label[len(_SUGG_PREFIX):]
            # Replace the current partial word (text after last space)
            parts = self.typed.rsplit(' ', 1)
            self.typed = (parts[0] + ' ' if len(parts) > 1 else '') + word + ' '
        elif label == '<-':
            self.typed = self.typed[:-1]
        elif label == 'SPC':
            self.typed += ' '
        elif label == 'ENT':
            self.typed = ''
        elif label == 'CAP':
            self.upper = not self.upper
        elif label == 'SEND':
            pass
        else:
            if self.upper:
                label = label.split(" ")[0].upper()
            else:
                label = label.split(" ")[-1].lower()
            self.typed += label

    # ── Drawing ────────────────────────────────────────────────────────────────

    def draw(self, frame):
        """Render keyboard + suggestion bar overlay onto frame in-place."""
        h_f, w_f = frame.shape[:2]
        now = time.time()

        # Rebuild suggestion chip geometry each frame
        self._build_sugg_rects()

        # ── 0. Suggestion bar background ──────────────────────────────────
        sugg_top = 40
        sugg_bot = 40 + SUGGEST_BAR_H
        if self._sugg_rects:
            bar_ov = frame.copy()
            cv2.rectangle(bar_ov, (0, sugg_top), (w_f, sugg_bot), (20, 20, 20), -1)
            cv2.addWeighted(bar_ov, 0.55, frame, 0.45, 0, frame)
            cv2.line(frame, (0, sugg_bot), (w_f, sugg_bot), (60, 60, 60), 1)

        # -- 0a. Send button ───────────────
        if self.send_key:
            label, cx, cy, cw, ch = self.send_key
            is_hover = (label == self.hover_key)
            is_flash = (label == self.flash_key)
            if is_flash:
                key_col, alpha = COL_KEY_FLASH, 0.90
            elif is_hover:
                key_col, alpha = COL_KEY_HOVER, 0.75
            else:
                key_col, alpha = (50, 50, 80), 0.60

            key_ov = frame.copy()
            cv2.rectangle(key_ov, (cx, cy), (cx+cw, cy+ch), key_col, -1)
            cv2.addWeighted(key_ov, alpha, frame, 1 - alpha, 0, frame)
            border = (200, 200, 200) if is_flash else (100, 120, 160)
            cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), border, 1)
            font   = cv2.FONT_HERSHEY_SIMPLEX
            fscale = 0.55
            thick  = 1
            tcol = (20, 20, 20) if is_flash else (230, 230, 255)
            (tw, th), _ = cv2.getTextSize(label, font, fscale, thick)
            tx = cx + (cw - tw) // 2
            ty = cy + (ch + th) // 2
            cv2.putText(frame, label, (tx, ty), font, fscale, tcol, thick, cv2.LINE_AA)
        # ── 0b. Suggestion chips ──────────────────────────────────────────
        for label, cx, cy, cw, ch in self._sugg_rects:
            word     = label[len(_SUGG_PREFIX):]
            is_hover = (label == self.hover_key)
            is_flash = (label == self.flash_key)

            if is_flash:
                chip_col, alpha = COL_KEY_FLASH, 0.90
            elif is_hover:
                chip_col, alpha = COL_KEY_HOVER, 0.75
            else:
                chip_col, alpha = (50, 50, 80), 0.60

            chip_ov = frame.copy()
            cv2.rectangle(chip_ov, (cx, cy), (cx+cw, cy+ch), chip_col, -1)
            cv2.addWeighted(chip_ov, alpha, frame, 1 - alpha, 0, frame)

            border = (200, 200, 200) if is_flash else (100, 120, 160)
            cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), border, 1)

            font   = cv2.FONT_HERSHEY_SIMPLEX
            fscale = 0.55
            thick  = 1
            tcol   = (20, 20, 20) if is_flash else (230, 230, 255)
            (tw, th), _ = cv2.getTextSize(word, font, fscale, thick)
            tx = cx + (cw - tw) // 2
            ty = cy + (ch + th) // 2
            cv2.putText(frame, word, (tx, ty), font, fscale, tcol, thick, cv2.LINE_AA)

        # ── 1. Faint dark tint over keyboard area ─────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, self.kb_top), (w_f, self.kb_bot), (0, 0, 0), -1)
        cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1 - PANEL_ALPHA, 0, frame)

        # ── 2. Keys ───────────────────────────────────────────────────────
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

        # ── 3. Text output bar at the bottom ──────────────────────────────
        bar_y  = self.kb_bot
        bar_ov = frame.copy()
        cv2.rectangle(bar_ov, (0, bar_y), (w_f, h_f), COL_TEXT_BAR, -1)
        cv2.addWeighted(bar_ov, 0.80, frame, 0.20, 0, frame)
        cv2.line(frame, (0, bar_y), (w_f, bar_y), (80, 80, 80), 1)

        display = self.typed[-90:]
        blink   = '|' if int(now * 2) % 2 == 0 else ' '
        cv2.putText(frame, display + blink, (14, bar_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_TEXT_FG, 1, cv2.LINE_AA)