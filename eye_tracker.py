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
import urllib.request
from collections import deque

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, FaceLandmarker
from mediapipe import Image, ImageFormat

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("📥  Downloading face landmarker model (~7 MB) …")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception:
            # macOS Python 3.13 SSL cert issue — bypass verification
            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(MODEL_URL, context=ctx) as r, \
                 open(MODEL_PATH, "wb") as f:
                f.write(r.read())
        print(f"✅  Saved to {MODEL_PATH}")

# ── Landmark indices ───────────────────────────────────────────────────────────
LEFT_IRIS         = [474, 475, 476, 477]
RIGHT_IRIS        = [469, 470, 471, 472]
LEFT_EYE_CORNERS  = [362, 263]
RIGHT_EYE_CORNERS = [33,  133]
LEFT_EYE_LIDS     = [386, 374]
RIGHT_EYE_LIDS    = [159, 145]

# ── Helpers ────────────────────────────────────────────────────────────────────

def iris_center(lm_list, indices, w, h):
    pts = np.array([[lm_list[i].x * w, lm_list[i].y * h] for i in indices])
    return pts.mean(axis=0)

def eye_bounds(lm_list, corner_idx, lid_idx, w, h):
    xs = [lm_list[i].x * w for i in corner_idx]
    ys = [lm_list[i].y * h for i in lid_idx]
    return min(xs), max(xs), min(ys), max(ys)

def gaze_ratio(iris_pt, x0, x1, y0, y1):
    rx = (iris_pt[0] - x0) / max(x1 - x0, 1)
    ry = (iris_pt[1] - y0) / max(y1 - y0, 1)
    return float(np.clip(rx, 0, 1)), float(np.clip(ry, 0, 1))

# ── Calibration ────────────────────────────────────────────────────────────────
CAL_GRID = [(cx, cy)
            for cy in (0.15, 0.50, 0.85)
            for cx in (0.15, 0.50, 0.85)]

class Calibrator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.point_idx    = 0
        self.gaze_samples = []
        self.screen_pts   = []
        self.gaze_pts     = []
        self.matrix       = None
        self.active       = False
        self.done         = False

    def start(self):
        self.reset()
        self.active = True

    def current_target(self, sw, sh):
        if self.point_idx >= len(CAL_GRID):
            return None
        fx, fy = CAL_GRID[self.point_idx]
        return int(fx * sw), int(fy * sh)

    def collect(self, rx, ry):
        if self.active and not self.done:
            self.gaze_samples.append((rx, ry))

    def confirm_point(self, sw, sh):
        if not self.active or self.done or len(self.gaze_samples) < 5:
            return False
        mrx = float(np.mean([s[0] for s in self.gaze_samples]))
        mry = float(np.mean([s[1] for s in self.gaze_samples]))
        tx, ty = self.current_target(sw, sh)
        self.screen_pts.append([tx, ty])
        self.gaze_pts.append([mrx, mry])
        self.gaze_samples = []
        self.point_idx += 1
        if self.point_idx >= len(CAL_GRID):
            self._fit()
            return True
        return False

    def _fit(self):
        src = np.array(self.gaze_pts,   dtype=np.float32)
        dst = np.array(self.screen_pts, dtype=np.float32)
        M, _ = cv2.estimateAffinePartial2D(src, dst)
        if M is None:
            sx = (dst[:,0].max()-dst[:,0].min()) / max(src[:,0].max()-src[:,0].min(), 1e-6)
            sy = (dst[:,1].max()-dst[:,1].min()) / max(src[:,1].max()-src[:,1].min(), 1e-6)
            tx = dst[:,0].mean() - sx*src[:,0].mean()
            ty = dst[:,1].mean() - sy*src[:,1].mean()
            M  = np.array([[sx,0,tx],[0,sy,ty]], dtype=np.float32)
        self.matrix = M
        self.done   = True
        self.active = False

    def map_gaze(self, rx, ry):
        if self.matrix is None:
            return None
        pt     = np.array([[[rx, ry]]], dtype=np.float32)
        mapped = cv2.transform(pt, self.matrix)
        return int(mapped[0,0,0]), int(mapped[0,0,1])

# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_gaze_dot(frame, x, y, r=22):
    overlay = frame.copy()
    cv2.circle(overlay, (x, y), r, (0, 220, 255), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.circle(frame, (x, y), r, (0, 220, 255), 2)
    cv2.circle(frame, (x, y), 4,  (255, 255, 255), -1)

def draw_cal_dot(frame, x, y, pulse):
    r = int(14 + 6 * pulse)
    cv2.circle(frame, (x, y), r+6, (0, 0, 180), 2)
    cv2.circle(frame, (x, y), r,   (0, 0, 255), -1)
    cv2.circle(frame, (x, y), 4,   (255, 255, 255), -1)

def draw_eye_box(frame, iris_pt, x0, x1, y0, y1):
    ix, iy = int(iris_pt[0]), int(iris_pt[1])
    cv2.circle(frame, (ix, iy), 4, (0, 255, 160), -1)
    cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 160), 1)

def draw_hud(frame, cal, w, h):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    if cal.done:
        msg, col = "Tracking  |  [R] Recalibrate  [Q] Quit", (80, 220, 80)
    elif cal.active:
        left = len(CAL_GRID) - cal.point_idx
        msg, col = f"Calibrating — look at dot, press SPACE  ({left} left)", (80, 180, 255)
    else:
        msg, col = "Press SPACE to start calibration  |  [Q] Quit", (200, 200, 200)
    cv2.putText(frame, msg, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

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
    print("   Press SPACE in the window to begin calibration.")
    print("   Press Q or ESC to quit.\n")

    cal          = Calibrator()
    gaze_history = deque(maxlen=12)
    t0           = time.time()

    # On macOS, the first N frames often produce spurious key events
    # (especially ESC=27) before the window is properly focused.
    # We ignore Q/ESC for the first 60 frames (~2 sec) to prevent
    # the app from closing immediately on launch.
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

        gaze_rx, gaze_ry = 0.5, 0.5

        if result.face_landmarks:
            lm = result.face_landmarks[0]

            l_iris = iris_center(lm, LEFT_IRIS,  w, h)
            r_iris = iris_center(lm, RIGHT_IRIS, w, h)

            lx0,lx1,ly0,ly1 = eye_bounds(lm, LEFT_EYE_CORNERS,  LEFT_EYE_LIDS,  w, h)
            rx0,rx1,ry0,ry1 = eye_bounds(lm, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS, w, h)

            l_rx, l_ry = gaze_ratio(l_iris, lx0, lx1, ly0, ly1)
            r_rx, r_ry = gaze_ratio(r_iris, rx0, rx1, ry0, ry1)
            gaze_rx    = (l_rx + r_rx) / 2
            gaze_ry    = (l_ry + r_ry) / 2

            if cal.active and not cal.done:
                cal.collect(gaze_rx, gaze_ry)

            draw_eye_box(frame, l_iris, lx0, lx1, ly0, ly1)
            draw_eye_box(frame, r_iris, rx0, rx1, ry0, ry1)

        if cal.done:
            mapped = cal.map_gaze(gaze_rx, gaze_ry)
            if mapped:
                mx = int(np.clip(mapped[0], 0, w-1))
                my = int(np.clip(mapped[1], 0, h-1))
                gaze_history.append((mx, my))
                sx = int(np.mean([p[0] for p in gaze_history]))
                sy = int(np.mean([p[1] for p in gaze_history]))
                draw_gaze_dot(frame, sx, sy)

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

        draw_hud(frame, cal, w, h)
        cv2.putText(frame, f"iris  x:{gaze_rx:.2f}  y:{gaze_ry:.2f}",
                    (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140,140,140), 1, cv2.LINE_AA)

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
                    print("✅  Calibration complete! Gaze tracking active.")
                else:
                    print(f"   Point {cal.point_idx}/{len(CAL_GRID)} recorded.")
        elif key == ord('r'):
            cal.reset()
            gaze_history.clear()
            print("🔄  Calibration reset.")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("👋  Done.")

if __name__ == "__main__":
    main()