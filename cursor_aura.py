"""
Cursor Aura
===========
Draws a glowing ring around the real OS mouse cursor when in browse mode.
Uses a borderless, transparent, always-on-top tkinter window that follows
the mouse position every 30ms.

The aura makes it easy to see where your gaze cursor currently is on screen,
especially when the browser has focus and the cv2 window is in the background.
"""

import threading
import time

try:
    import tkinter as tk
    _TK = True
except ImportError:
    _TK = False

try:
    import pyautogui
    _PYAUTO = True
except ImportError:
    _PYAUTO = False

# Aura appearance
AURA_RADIUS  = 28       # radius of the ring in pixels
AURA_WIDTH   = 3        # ring stroke width
AURA_COLOUR  = "#00dc82"  # bright green — visible on any background
AURA_COLOUR2 = "#00a8ff"  # inner ring colour
CANVAS_SIZE  = (AURA_RADIUS + 8) * 2   # canvas is a square big enough for the ring
POLL_MS      = 30       # how often to update position (ms)


class CursorAura:
    """
    Spawns a transparent tkinter overlay that follows the OS cursor.
    Call start() when entering browse mode, stop() when leaving.
    Runs entirely on its own thread — safe to call from any thread.
    """

    def __init__(self):
        self._thread  = None
        self._running = False
        self._root    = None

    def start(self):
        if not _TK or not _PYAUTO:
            return
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        # Signal the tk mainloop to quit on its own thread
        if self._root:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass
        self._root = None

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run(self):
        if not _TK or not _PYAUTO:
            return

        self._root = tk.Tk()
        root = self._root

        # Completely transparent, borderless, always on top
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-transparentcolor", "black")
        root.configure(bg="black")
        root.resizable(False, False)

        # Make the window click-through on Windows — mouse events pass
        # straight through to whatever app is underneath (the browser etc.)
        try:
            import ctypes
            hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)  # GWL_EXSTYLE
            # Add WS_EX_LAYERED (0x80000) + WS_EX_TRANSPARENT (0x20)
            ctypes.windll.user32.SetWindowLongW(hwnd, -20, style | 0x80000 | 0x20)
        except Exception:
            pass  # non-Windows or ctypes unavailable — aura still shows, just not click-through

        size = CANVAS_SIZE
        root.geometry(f"{size}x{size}+0+0")

        canvas = tk.Canvas(root, width=size, height=size,
                           bg="black", highlightthickness=0)
        canvas.pack()

        cx = cy = size // 2

        # Outer glow ring
        r = AURA_RADIUS
        canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                           outline=AURA_COLOUR, width=AURA_WIDTH, tags="outer")
        # Inner ring
        r2 = AURA_RADIUS - 6
        canvas.create_oval(cx-r2, cy-r2, cx+r2, cy+r2,
                           outline=AURA_COLOUR2, width=1, tags="inner")
        # Centre dot
        canvas.create_oval(cx-3, cy-3, cx+3, cy+3,
                           fill=AURA_COLOUR, outline="", tags="dot")

        def poll():
            if not self._running:
                try:
                    root.destroy()
                except Exception:
                    pass
                return
            try:
                mx, my = pyautogui.position()
                offset = size // 2
                root.geometry(f"{size}x{size}+{mx - offset}+{my - offset}")
            except Exception:
                pass
            root.after(POLL_MS, poll)

        root.after(POLL_MS, poll)
        root.mainloop()
        self._root    = None
        self._running = False