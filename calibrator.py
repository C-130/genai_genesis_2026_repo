"""
Calibrator Module
"""

import cv2
import numpy as np

from utils import CAL_GRID

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