"""
Eye Tracker App
===============
CONTROLS:
  UP ARROW  — press hovered key / confirm overlay selection
  SPACE     — start / confirm calibration point
  R         — reset calibration
  N         — reset neutral baseline
  I         — toggle intent menu
  H         — trigger health survey now
  Q / ESC   — quit
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import os
import webbrowser

try:
    from pynput import keyboard as pynput_kb
    _PYNPUT = True
except ImportError:
    _PYNPUT = False
    print("[Hotkey] pynput not installed — pip install pynput")
    print("[Hotkey] Up arrow will only work when cv2 window has focus")

if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass

from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
from dotenv import load_dotenv

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, FaceLandmarker
from mediapipe import Image, ImageFormat

from utils import (
    ensure_model, MODEL_PATH,
    LEFT_IRIS, RIGHT_IRIS, LEFT_EYE_CORNERS, RIGHT_EYE_CORNERS,
    LEFT_EYE_LIDS, RIGHT_EYE_LIDS, LEFT_EYE_INNER, RIGHT_EYE_INNER,
    iris_center, eye_bounds, gaze_ratio, gaze_offset,
    draw_gaze_dot, draw_cal_dot, draw_eye_box, draw_hud,
    draw_direction_hint, CAL_GRID, SMOOTH_ALPHA,
)
from gaze_cursor      import GazeCursor
from calibrator       import Calibrator
from virtual_keyboard import VirtualKeyboard
from moorcheh_memory  import MoorchehMemory
from health_survey    import HealthSurvey
from intent_overlay   import IntentOverlay
from clarify_overlay  import ClarifyOverlay
from agent            import IntentAgent

# URLs opened in browse mode
BROWSE_URLS = [
    "https://youtube.com",
    "https://instagram.com",
    "https://discord.com/app",
    "https://google.com",
]


def main():
    ensure_model()
    load_dotenv()

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
        print("Cannot open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {w}x{h}")

    # ── Modules ───────────────────────────────────────────────────────────────
    cal     = Calibrator()
    cursor  = GazeCursor(w, h)
    kb      = VirtualKeyboard(frame_w=w, frame_h=h)
    memory  = MoorchehMemory()
    memory.start()

    # ── ElevenLabs ────────────────────────────────────────────────────────────
    eleven_api_key = os.getenv("ELEVEN_LAB_API_KEY")
    tts_client     = ElevenLabs(api_key=eleven_api_key)

    clarify = ClarifyOverlay(frame_w=w, frame_h=h)
    intent  = IntentOverlay(frame_w=w, frame_h=h)
    status_line = [""]

    def on_status(msg):
        print(f"[Agent] {msg}")
        status_line[0] = msg

    def on_speak(msg):
        print(f"[Speak] {msg}")

    def on_alert(msg):
        print(f"[HEALTH ALERT] {msg}")

    agent  = IntentAgent(clarify, on_status=on_status, on_speak=on_speak)
    survey = HealthSurvey(clarify, on_status=on_status,
                          on_speak=on_speak, on_alert=on_alert,
                          tts_client=tts_client)
    survey.start(run_on_start=True)

    # ── Cursor aura (shown in browse mode) ───────────────────────────────────
    from cursor_aura import CursorAura
    cursor_aura = CursorAura()

    # ── Window setup ──────────────────────────────────────────────────────────
    WIN_NAME = "Eye Tracker"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, w, h)

    sentence     = ""
    browse_mode  = False
    t0           = time.time()
    GRACE_FRAMES = 60
    frame_count  = 0
    _sdx = _sdy  = 0.0

    def enter_browse_mode():
        nonlocal browse_mode
        browse_mode        = True
        cursor.browse_mode = True
        try:
            import pyautogui as _pg
            sh = _pg.size().height
            sw = _pg.size().width
            # Move window to bottom of screen, keep original w x h dimensions
            win_y = sh - h
            cv2.moveWindow(WIN_NAME, (sw - w) // 2, win_y)
            # Keep window same size — don't squash the keyboard
            cv2.resizeWindow(WIN_NAME, w, h)
            # Always on top
            cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_TOPMOST, 1)
        except Exception as e:
            print(f"[Browse] Window move error: {e}")
        for url in BROWSE_URLS:
            webbrowser.open(url)
        cursor_aura.start()
        on_status("Browse mode — cursor controls full screen")
        print("[Browse] Mode ON")

    def exit_browse_mode():
        nonlocal browse_mode
        browse_mode        = False
        cursor.browse_mode = False
        cursor.x           = float(w // 2)
        cursor.y           = float(h // 2)
        cv2.resizeWindow(WIN_NAME, w, h)
        cv2.moveWindow(WIN_NAME, 0, 0)
        cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_TOPMOST, 0)
        cursor_aura.stop()
        on_status("Normal mode")
        print("[Browse] Mode OFF")

    # ── Global hotkey listener (works even when cv2 window lacks focus) ────────
    # Fires the same logic as the up-arrow handler in the main loop
    _up_pressed = [False]   # flag read each frame

    if _PYNPUT:
        def _on_press(key):
            try:
                if key == pynput_kb.Key.up:
                    _up_pressed[0] = True
            except Exception:
                pass

        _hotkey_listener = pynput_kb.Listener(on_press=_on_press)
        _hotkey_listener.daemon = True
        _hotkey_listener.start()
    
    cv2.namedWindow("Eye Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Eye Tracker", 2560, 1440)
    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)

        raw_dx = raw_dy = 0.0
        gaze_rx = gaze_ry = 0.5

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            l_iris = iris_center(lm, LEFT_IRIS,  w, h)
            r_iris = iris_center(lm, RIGHT_IRIS, w, h)
            lx0, lx1, ly0, ly1 = eye_bounds(lm, LEFT_EYE_CORNERS, LEFT_EYE_LIDS,  w, h)
            rx0, rx1, ry0, ry1 = eye_bounds(lm, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS, w, h)
            l_corner_y = lm[LEFT_EYE_INNER].y * h
            r_corner_y = lm[RIGHT_EYE_INNER].y * h
            l_dx, l_dy = gaze_offset(l_iris, lx0, lx1, ly0, ly1, l_corner_y)
            r_dx, r_dy = gaze_offset(r_iris, rx0, rx1, ry0, ry1, r_corner_y)
            raw_dx = (l_dx + r_dx) / 2
            raw_dy = (l_dy + r_dy) / 2
            l_rx, l_ry = gaze_ratio(l_iris, lx0, lx1, ly0, ly1)
            r_rx, r_ry = gaze_ratio(r_iris, rx0, rx1, ry0, ry1)
            gaze_rx    = (l_rx + r_rx) / 2
            gaze_ry    = (l_ry + r_ry) / 2
            if cal.active and not cal.done:
                cal.collect(gaze_rx, gaze_ry)
            draw_eye_box(frame, l_iris, lx0, lx1, ly0, ly1)
            draw_eye_box(frame, r_iris, rx0, rx1, ry0, ry1)

        cursor.update(raw_dx, raw_dy)
        cx, cy = cursor.pos()

        _sdx += SMOOTH_ALPHA * (raw_dx - _sdx)
        _sdy += SMOOTH_ALPHA * (raw_dy - _sdy)

        # ── Route gaze to active overlay or keyboard ──────────────────────
        memory.on_typing(sentence)
        kb.set_suggestions(memory.suggestions)

        if clarify.visible:
            clarify.update_gaze(cx, cy)
        elif intent.visible:
            intent.update_gaze(cx, cy)
        else:
            kb.update_gaze(cx, cy)

        # ── Draw order: kb → overlays → HUD → cursor (always last) ───────
        kb.draw(frame)

        if clarify.visible:
            clarify.draw(frame)
        elif intent.visible:
            intent.draw(frame)

        draw_direction_hint(frame, _sdx, _sdy, w, h)

        # Calibration dots
        if cal.active and not cal.done:
            target = cal.current_target(w, h)
            if target:
                pulse = (np.sin((time.time()-t0)*4)+1)/2
                draw_cal_dot(frame, target[0], target[1], pulse)
                cv2.putText(frame,
                            f"Samples: {len(cal.gaze_samples)}  — press SPACE",
                            (target[0]-140, target[1]+50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        for i, (fx, fy) in enumerate(CAL_GRID):
            if i < cal.point_idx:
                cv2.circle(frame, (int(fx*w), int(fy*h)), 6, (0,200,80), -1)

        # HUD strip
        draw_hud(frame, cal, w, h,
                 cursor_active=not cal.active, cursor=cursor)

        # Status line
        if status_line[0]:
            cv2.putText(frame, status_line[0][:90], (10, h-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,200,255), 1, cv2.LINE_AA)

        # Debug info
        cv2.putText(frame, f"dx:{raw_dx:+.2f} dy:{raw_dy:+.2f}",
                    (10, h-42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100,100,100), 1)
        cv2.putText(frame, f"cursor {cx},{cy}",
                    (10, h-26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100,100,100), 1)

        # ── Cursor drawn LAST — always on top ─────────────────────────────
        cursor.draw(frame)

        cv2.imshow(WIN_NAME, frame)

        # ── Key handling ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27) and frame_count > GRACE_FRAMES:
            break

        elif key == ord(' '):
            if not cal.active and not cal.done:
                cal.start()
            elif cal.active and not cal.done:
                done = cal.confirm_point(w, h)
                if done:
                    print("Calibration complete!")

        elif key == ord('i') or key == ord('I'):
            if intent.visible:
                intent.hide()
            elif not clarify.visible and not agent.busy():
                intent.show()

        elif key == ord('h') or key == ord('H'):
            if not clarify.visible:
                survey.run_now()

        elif key == ord('r'):
            cal.reset()
            cursor.reset_baseline()
            cursor.x = float(w // 2)
            cursor.y = float(h // 2)

        elif key == ord('n'):
            cursor.reset_baseline()

        # Up arrow — fires from cv2 (when window focused) OR pynput (global)
        up_fired = (key == 0) or _up_pressed[0]
        _up_pressed[0] = False   # consume the flag
        if up_fired:   # up arrow
            if browse_mode:
                # In browse mode the cursor controls the real screen — click it
                try:
                    import pyautogui as _pg
                    _pg.click()
                except Exception as e:
                    print(f"[Browse] Click error: {e}")

            elif clarify.visible:
                clarify.press()

            elif intent.visible:
                selected = intent.press()
                if selected:
                    agent.start(selected, sentence)

            else:
                typed = kb.press()

                # Top bar button pressed
                if typed == "__BTN__actions":
                    if not agent.busy():
                        intent.show()

                elif typed == "__BTN__browse":
                    if browse_mode:
                        exit_browse_mode()
                    else:
                        enter_browse_mode()

                elif typed.startswith("__SUGG__"):
                    sentence = kb.typed

                elif typed == 'ENT':
                    memory.store_phrase(sentence)
                    audio = tts_client.text_to_speech.convert(
                        text=sentence,
                        voice_id="JBFqnCBsd6RMkjVDRZzb",
                        model_id="eleven_multilingual_v2",
                        output_format="mp3_44100_128",
                    )
                    play(audio)
                    print(sentence)
                    sentence = ""

                elif typed == '<-':
                    sentence = sentence[:-1]

                elif typed:
                    sentence += typed

    memory.stop()
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Done.")


if __name__ == "__main__":
    main()