# virtual_keyboard.py
import cv2
import time
import numpy as np

from utils import (
    KB_ROWS, KEY_MARGIN, KB_ALPHA, PANEL_ALPHA, TEXT_BAR_H,
    COL_KEY_BG, COL_KEY_BORDER, COL_KEY_HOVER,
    COL_KEY_FLASH, COL_KEY_TEXT, COL_TEXT_BAR, COL_TEXT_FG,
)

SUGGEST_BAR_H = 48
_SUGG_PREFIX  = "__SUGG__"

# ── Top button bar ─────────────────────────────────────────────────────────────
HUD_H        = 40          # height of the top HUD strip (drawn by draw_hud)
TOP_BAR_H    = 52          # height of the action button bar (sits below HUD)
BTN_MARGIN   = 8
BTN_RADIUS   = 6           # corner rounding via polylines

# Button definitions: (id, label, colour_bgr_normal, colour_bgr_hover)
TOP_BUTTONS = [
    ("actions",  "Actions",     (55, 65, 90),   (90, 105, 145)),
    ("browse",   "Browse Mode", (40, 90, 55),    (60, 140, 80)),
]


def _rounded_rect(frame, x, y, w, h, r, colour, alpha, border=None):
    """Draw a filled rounded rectangle blended onto frame."""
    ov = frame.copy()
    # Fill via ellipses at corners + inner rects
    cv2.rectangle(ov, (x+r, y), (x+w-r, y+h), colour, -1)
    cv2.rectangle(ov, (x, y+r), (x+w, y+h-r), colour, -1)
    for cx2, cy2 in [(x+r, y+r), (x+w-r, y+r), (x+r, y+h-r), (x+w-r, y+h-r)]:
        cv2.circle(ov, (cx2, cy2), r, colour, -1)
    cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
    if border is not None:
        cv2.rectangle(frame, (x+r, y), (x+w-r, y+h), border, 1)
        cv2.rectangle(frame, (x, y+r), (x+w, y+h-r), border, 1)
        for cx2, cy2 in [(x+r, y+r), (x+w-r, y+r), (x+r, y+h-r), (x+w-r, y+h-r)]:
            cv2.circle(frame, (cx2, cy2), r, border, 1)


class VirtualKeyboard:
    """Full-screen transparent keyboard with top action bar."""

    def __init__(self, frame_w, frame_h):
        self.frame_w   = frame_w
        self.frame_h   = frame_h
        # Layout zones
        self.top_bar_y  = HUD_H
        self.sugg_bar_y = HUD_H + TOP_BAR_H
        self.kb_top     = HUD_H + TOP_BAR_H + SUGGEST_BAR_H
        self.kb_bot     = frame_h - TEXT_BAR_H
        self.kb_h       = self.kb_bot - self.kb_top
        self.keys       = []
        self.typed      = ""
        self.hover_key  = None
        self.flash_key  = None
        self.flash_t    = 0.0

        self._suggestions = []
        self._sugg_rects  = []
        self._btn_rects   = []   # [(id, x, y, w, h), ...]

        self._build_layout()
        self._build_buttons()

    # ── Layout ─────────────────────────────────────────────────────────────────

    def _build_layout(self):
        self.keys = []
        n_rows = len(KB_ROWS)
        row_h  = (self.kb_h - (n_rows + 1) * KEY_MARGIN) // n_rows
        for row_i, row in enumerate(KB_ROWS):
            n_keys  = len(row)
            total_m = (n_keys + 1) * KEY_MARGIN
            key_w   = (self.frame_w - total_m) // n_keys
            y = self.kb_top + KEY_MARGIN + row_i * (row_h + KEY_MARGIN)
            for col_i, label in enumerate(row):
                x  = KEY_MARGIN + col_i * (key_w + KEY_MARGIN)
                kw = key_w * 2 + KEY_MARGIN if label == 'SPC' else key_w
                self.keys.append((label, x, y, kw, row_h))

    def _build_buttons(self):
        n       = len(TOP_BUTTONS)
        pad     = BTN_MARGIN
        btn_w   = (self.frame_w - pad * (n + 1)) // n
        btn_h   = TOP_BAR_H - pad * 2
        self._btn_rects = []
        for i, (bid, _, _, _) in enumerate(TOP_BUTTONS):
            x = pad + i * (btn_w + pad)
            y = HUD_H + pad
            self._btn_rects.append((bid, x, y, btn_w, btn_h))

    # ── Suggestions ────────────────────────────────────────────────────────────

    def set_suggestions(self, words):
        self._suggestions = list(words[:3])

    def _build_sugg_rects(self):
        self._sugg_rects = []
        if not self._suggestions:
            return
        n      = len(self._suggestions)
        margin = KEY_MARGIN
        bar_y  = self.sugg_bar_y
        chip_h = SUGGEST_BAR_H - margin * 2
        chip_w = (self.frame_w - margin * (n + 1)) // n
        for i, word in enumerate(self._suggestions):
            x = margin + i * (chip_w + margin)
            y = bar_y + margin
            self._sugg_rects.append((_SUGG_PREFIX + word, x, y, chip_w, chip_h))

    # ── Hit testing ────────────────────────────────────────────────────────────

    def key_at(self, gx, gy):
        for bid, x, y, bw, bh in self._btn_rects:
            if x <= gx < x + bw and y <= gy < y + bh:
                return "__BTN__" + bid
        for label, x, y, cw, ch in self._sugg_rects:
            if x <= gx < x + cw and y <= gy < y + ch:
                return label
        for label, x, y, kw, kh in self.keys:
            if x <= gx < x + kw and y <= gy < y + kh:
                return label
        return None

    def update_gaze(self, gx, gy):
        self.hover_key = self.key_at(gx, gy)
        if self.flash_key and time.time() - self.flash_t > 0.2:
            self.flash_key = None

    # ── Key press ──────────────────────────────────────────────────────────────

    def press(self):
        if self.hover_key is None:
            return ''
        self._fire(self.hover_key)
        self.flash_key = self.hover_key
        self.flash_t   = time.time()
        if self.hover_key == 'SPC':
            return ' '
        return self.hover_key

    def _fire(self, label):
        if label.startswith("__BTN__"):
            return   # handled by eye_tracker.py
        if label.startswith(_SUGG_PREFIX):
            word  = label[len(_SUGG_PREFIX):]
            parts = self.typed.rsplit(' ', 1)
            self.typed = (parts[0] + ' ' if len(parts) > 1 else '') + word + ' '
        elif label == '<-':
            self.typed = self.typed[:-1]
        elif label == 'SPC':
            self.typed += ' '
        elif label == 'ENT':
            self.typed = ''
        else:
            self.typed += label

    # ── Drawing ────────────────────────────────────────────────────────────────

    def draw(self, frame):
        h_f, w_f = frame.shape[:2]
        now = time.time()
        self._build_sugg_rects()

        # ── Top bar background ────────────────────────────────────────────
        ov = frame.copy()
        cv2.rectangle(ov, (0, HUD_H), (w_f, HUD_H + TOP_BAR_H), (18, 20, 28), -1)
        cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)
        cv2.line(frame, (0, HUD_H + TOP_BAR_H), (w_f, HUD_H + TOP_BAR_H), (50, 55, 70), 1)

        # ── Top buttons ───────────────────────────────────────────────────
        for (bid, label, col_n, col_h), (_, bx, by, bw, bh) in zip(TOP_BUTTONS, self._btn_rects):
            is_hover = (self.hover_key == "__BTN__" + bid)
            is_flash = (self.flash_key == "__BTN__" + bid)
            col   = col_h if (is_hover or is_flash) else col_n
            alpha = 0.95 if is_flash else (0.90 if is_hover else 0.80)
            border = (160, 175, 210) if is_hover else (70, 78, 100)
            _rounded_rect(frame, bx, by, bw, bh, BTN_RADIUS, col, alpha, border)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.52, 1)
            cv2.putText(frame, label,
                        (bx + (bw-tw)//2, by + (bh+th)//2 - 1),
                        font, 0.52, (220, 225, 240), 1, cv2.LINE_AA)

        # ── Suggestion bar ────────────────────────────────────────────────
        if self._sugg_rects:
            ov2 = frame.copy()
            cv2.rectangle(ov2, (0, self.sugg_bar_y),
                          (w_f, self.sugg_bar_y + SUGGEST_BAR_H), (22, 24, 32), -1)
            cv2.addWeighted(ov2, 0.65, frame, 0.35, 0, frame)
            cv2.line(frame, (0, self.sugg_bar_y + SUGGEST_BAR_H),
                     (w_f, self.sugg_bar_y + SUGGEST_BAR_H), (50, 55, 70), 1)

        for label, cx, cy, cw, ch in self._sugg_rects:
            word     = label[len(_SUGG_PREFIX):]
            is_hover = (label == self.hover_key)
            is_flash = (label == self.flash_key)
            if is_flash:
                chip_col, alpha = COL_KEY_FLASH, 0.90
            elif is_hover:
                chip_col, alpha = COL_KEY_HOVER, 0.80
            else:
                chip_col, alpha = (42, 48, 68), 0.70
            _rounded_rect(frame, cx, cy, cw, ch, 4, chip_col, alpha,
                          (200,200,200) if is_flash else (80, 95, 130))
            font = cv2.FONT_HERSHEY_SIMPLEX
            tcol = (20, 20, 20) if is_flash else (210, 215, 240)
            (tw, th), _ = cv2.getTextSize(word, font, 0.52, 1)
            cv2.putText(frame, word, (cx+(cw-tw)//2, cy+(ch+th)//2),
                        font, 0.52, tcol, 1, cv2.LINE_AA)

        # ── Keyboard panel tint ───────────────────────────────────────────
        ov3 = frame.copy()
        cv2.rectangle(ov3, (0, self.kb_top), (w_f, self.kb_bot), (0, 0, 0), -1)
        cv2.addWeighted(ov3, PANEL_ALPHA, frame, 1-PANEL_ALPHA, 0, frame)

        # ── Keys ──────────────────────────────────────────────────────────
        for label, kx, ky, kw, kh in self.keys:
            if label == self.flash_key:
                key_col, alpha = COL_KEY_FLASH, 0.92
            elif label == self.hover_key:
                key_col, alpha = COL_KEY_HOVER, 0.72
            else:
                key_col, alpha = COL_KEY_BG, KB_ALPHA
            _rounded_rect(frame, kx, ky, kw, kh, 4, key_col, alpha,
                          (200,200,200) if label==self.flash_key else COL_KEY_BORDER)
            disp   = 'SPACE' if label == 'SPC' else label
            font   = cv2.FONT_HERSHEY_SIMPLEX
            fscale = 0.48 if len(disp) > 1 else 0.62
            thick  = 1 if len(disp) > 1 else 2
            tcol   = (20,20,20) if label==self.flash_key else COL_KEY_TEXT
            (tw, th), _ = cv2.getTextSize(disp, font, fscale, thick)
            cv2.putText(frame, disp, (kx+(kw-tw)//2, ky+(kh+th)//2),
                        font, fscale, tcol, thick, cv2.LINE_AA)

        # ── Text output bar ───────────────────────────────────────────────
        bar_y = self.kb_bot
        ov4   = frame.copy()
        cv2.rectangle(ov4, (0, bar_y), (w_f, h_f), COL_TEXT_BAR, -1)
        cv2.addWeighted(ov4, 0.85, frame, 0.15, 0, frame)
        cv2.line(frame, (0, bar_y), (w_f, bar_y), (60, 65, 80), 1)
        display = self.typed[-90:]
        blink   = '|' if int(now*2)%2==0 else ' '
        cv2.putText(frame, display + blink, (14, bar_y+32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_TEXT_FG, 1, cv2.LINE_AA)