"""
Eye Tracker App — OpenCV + MediaPipe Tasks API
================================================
Compatible with mediapipe >= 0.10.x (uses FaceLandmarker Tasks API,
NOT the deprecated mp.solutions interface).

SETUP:
  pip install opencv-python mediapipe numpy

  The script auto-downloads the face_landmarker.task model (~7 MB)
  on first run and caches it next to this file.

CONTROLS:
  [SPACE]  — Start / confirm each calibration point
  [R]      — Reset calibration
  [Q/ESC]  — Quit

CALIBRATION:
  Look at each red pulsing dot, then press SPACE.
  After all 9 points the cyan gaze dot activates automatically.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import os

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
from gaze_cursor import GazeCursor
from calibrator import Calibrator
from virtual_keyboard import VirtualKeyboard
from answer_predictor_realtime import WordPredictor

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ensure_model()

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
        print("❌  Cannot open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📷  Camera: {w}×{h}")
    print("   Gaze direction moves the cursor — look left/right/up/down.")
    print("   Press SPACE to start keyboard calibration (for dwell typing).")
    print("   Press Q or ESC to quit.\n")

    cal    = Calibrator()
    cursor = GazeCursor(w, h)
    kb     = VirtualKeyboard(frame_w=w, frame_h=h)

    # ── Word Predictor (background threads) ───────────────────────────────────
    predictor = WordPredictor()
    predictor.start()

    # Smoothed raw gaze offsets (for direction hint display)
    _sdx = 0.0
    _sdy = 0.0

    # ── ElevenLabs TTS ────────────────────────────────────────────────────────
    load_dotenv()
    eleven_api_key = os.getenv("ELEVEN_LAB_API_KEY")
    tts_client = ElevenLabs(api_key=eleven_api_key)

    sentence    = ""
    t0          = time.time()
    GRACE_FRAMES = 60
    frame_count  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)

        raw_dx, raw_dy = 0.0, 0.0
        gaze_rx, gaze_ry = 0.5, 0.5

        if result.face_landmarks:
            lm = result.face_landmarks[0]

            l_iris = iris_center(lm, LEFT_IRIS,  w, h)
            r_iris = iris_center(lm, RIGHT_IRIS, w, h)

            lx0, lx1, ly0, ly1 = eye_bounds(lm, LEFT_EYE_CORNERS,  LEFT_EYE_LIDS,  w, h)
            rx0, rx1, ry0, ry1 = eye_bounds(lm, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS, w, h)

            l_corner_y = lm[LEFT_EYE_INNER].y  * h
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

        # ── Update cursor ─────────────────────────────────────────────────
        cursor.update(raw_dx, raw_dy)
        cx, cy = cursor.pos()

        _sdx += SMOOTH_ALPHA * (raw_dx - _sdx)
        _sdy += SMOOTH_ALPHA * (raw_dy - _sdy)

        # ── Feed latest suggestions to keyboard each frame ─────────────────
        kb.set_suggestions(predictor.suggestions)

        # ── Keyboard overlay ──────────────────────────────────────────────
        kb.update_gaze(cx, cy)
        kb.draw(frame)

        # Draw cursor on top of keyboard
        cursor.draw(frame)

        draw_direction_hint(frame, _sdx, _sdy, w, h)

        # ── Calibration dots ──────────────────────────────────────────────
        if cal.active and not cal.done:
            target = cal.current_target(w, h)
            if target:
                pulse = (np.sin((time.time()-t0)*4)+1)/2
                draw_cal_dot(frame, target[0], target[1], pulse)
                n = len(cal.gaze_samples)
                cv2.putText(frame, f"Samples: {n}  — press SPACE when ready",
                            (target[0]-150, target[1]+50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        for i, (fx, fy) in enumerate(CAL_GRID):
            if i < cal.point_idx:
                cv2.circle(frame, (int(fx*w), int(fy*h)), 6, (0, 200, 80), -1)

        # ── HUD ───────────────────────────────────────────────────────────
        draw_hud(frame, cal, w, h, kb.upper, kb.email_status, cursor_active=not cal.active, cursor=cursor)
        cv2.putText(frame,
                    f"gaze offset  dx:{raw_dx:+.2f}  dy:{raw_dy:+.2f}",
                    (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140,140,140), 1, cv2.LINE_AA)
        cv2.putText(frame,
                    f"cursor  x:{cx}  y:{cy}",
                    (10, h - 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140,140,140), 1, cv2.LINE_AA)

        # Show last transcript in HUD
        transcript = predictor.last_transcript
        if transcript:
            cv2.putText(frame,
                        f"heard: {transcript[-60:]}",
                        (10, h - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 200, 100), 1, cv2.LINE_AA)

        cv2.imshow("Eye Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27) and frame_count > GRACE_FRAMES:
            break
        elif key == ord(' '):
            if not cal.active and not cal.done:
                cal.start()
                print("▶  Calibration started.")
            elif cal.active and not cal.done:
                done = cal.confirm_point(w, h)
                if done:
                    print("✅  Calibration complete!")
                else:
                    print(f"   Point {cal.point_idx}/{len(CAL_GRID)} recorded.")
        elif key == ord('r'):
            cal.reset()
            cursor.reset_baseline()
            cursor.x = float(w // 2)
            cursor.y = float(h // 2)
            print("🔄  Reset — look straight ahead for neutral calibration.")
        elif key == ord('n'):
            cursor.reset_baseline()
            print("🔄  Neutral baseline reset — look straight ahead.")
        elif key == 0:   # up arrow
            typed = kb.press()
            print(f"Key pressed: {typed}")

            # Suggestion chip selected — extract the actual word
            if typed.startswith("__SUGG__"):
                word = typed[len("__SUGG__"):]
                # kb._fire already updated kb.typed; sync sentence to kb.typed
                sentence = kb.typed
                print(f"💡  Suggestion selected: '{word}'  →  {kb.typed!r}")

            elif typed == 'ENT':
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
                print(f"⌨  '{typed}'  →  {kb.typed}")

            elif typed:
                sentence += typed
                print(f"⌨  '{typed}'  →  {kb.typed}")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("👋  Done.")


if __name__ == "__main__":
    main()