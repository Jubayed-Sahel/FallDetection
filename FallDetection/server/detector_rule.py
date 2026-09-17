"""
Rule-based fall detector. No model file needed.
Reuses the same tracker + state machine we'll plug the trained GRU into later.
"""
import time
from collections import deque

import numpy as np
from ultralytics import YOLO

from features import extract_features, rule_based_fall, FEATURE_DIM

WINDOW = 30

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0


class Track:
    def __init__(self, tid, box):
        self.id = tid
        self.box = box
        self.last_seen = time.time()
        self.feat_buf = deque(maxlen=WINDOW)
        self.prev_state = None
        self.last_t = None
        self.is_fall = False
        self.fall_frames = 0
        self.normal_frames = 0
        self.alarmed = False
        self.last_debug = {}


class RuleFallDetector:
    def __init__(self, pose_weights='yolov8n-pose.pt', device='cpu'):
        self.pose = YOLO(pose_weights)
        self.device = device
        self.tracks = {}
        self._next_id = 1
        self._last_frame_t = None

    def _update_tracks(self, boxes):
        now = time.time()
        for tid in [t for t, tr in self.tracks.items()
                    if now - tr.last_seen > 1.5]:
            del self.tracks[tid]

        assigned = set()
        for box in boxes:
            best_tid, best_iou = None, 0.35
            for tid, tr in self.tracks.items():
                if tid in assigned: continue
                i = _iou(box, tr.box)
                if i > best_iou:
                    best_tid, best_iou = tid, i
            if best_tid is None:
                best_tid = self._next_id
                self._next_id += 1
                self.tracks[best_tid] = Track(best_tid, box)
            tr = self.tracks[best_tid]
            tr.box = box
            tr.last_seen = now
            assigned.add(best_tid)
        return assigned

    def process_frame(self, bgr):
        now = time.time()
        dt = (now - self._last_frame_t) if self._last_frame_t else 1/15.0
        dt = float(np.clip(dt, 1/60.0, 0.5))
        self._last_frame_t = now

        res = self.pose.predict(bgr, verbose=False, conf=0.35,
                                imgsz=640, device=self.device)[0]

        boxes, kpts_all, kconf_all = [], [], []
        if res.keypoints is not None and res.boxes is not None:
            xy = res.keypoints.xy.cpu().numpy()
            kc = (res.keypoints.conf.cpu().numpy()
                  if res.keypoints.conf is not None
                  else np.ones(xy.shape[:2], dtype=np.float32))
            bx = res.boxes.xyxy.cpu().numpy()
            for i in range(len(bx)):
                boxes.append(tuple(bx[i].tolist()))
                kpts_all.append(xy[i].astype(np.float32))
                kconf_all.append(kc[i].astype(np.float32))

        active = self._update_tracks(boxes)

        results = []
        for i, box in enumerate(boxes):
            tid, best = None, 0.35
            for t, tr in self.tracks.items():
                if t not in active: continue
                j = _iou(box, tr.box)
                if j > best:
                    tid, best = t, j
            if tid is None: continue

            tr = self.tracks[tid]
            feat, tr.prev_state = extract_features(
                kpts_all[i], kconf_all[i], box, tr.prev_state, dt=dt)
            tr.feat_buf.append(feat)
            if len(tr.feat_buf) < 5:
                continue

            window = np.stack(tr.feat_buf)
            is_fall, score, debug = rule_based_fall(window)
            tr.last_debug = debug

            if is_fall:
                tr.fall_frames += 1
                tr.normal_frames = 0
            else:
                tr.normal_frames += 1
                tr.fall_frames = max(0, tr.fall_frames - 1)

            event = None
            if not tr.alarmed and tr.fall_frames >= 5:
                tr.alarmed = True
                event = 'FALL_CONFIRMED'
            if tr.alarmed and tr.normal_frames >= 30:
                tr.alarmed = False

            results.append({
                'track_id': tr.id,
                'box': [round(float(v), 1) for v in box],
                'class': 'fallen' if tr.alarmed else 'normal',
                'score': round(score, 3),
                'fall': tr.alarmed,
                'event': event,
                'debug': debug,
                'keypoints': kpts_all[i].tolist(),
            })

        return {'persons': results, 'ts': now}