"""
Live debug viewer for the fall detector.

Runs the pose model + rule-based fall detector directly on the webcam feed
and draws a full diagnostic overlay so you can SEE what the model sees.

Usage:
    python live_view.py              # default camera (0)
    python live_view.py --camera 1   # different camera index
    python live_view.py --no-flip    # don't mirror the image

Keys:
    q  or  ESC  → quit
    s           → save current frame to debug_live.jpg
"""
import argparse
import time
import sys

import cv2
import numpy as np
from ultralytics import YOLO

from features import extract_features, rule_based_fall
from collections import deque

KP = {'nose':0,'leye':1,'reye':2,'lear':3,'rear':4,
      'lsho':5,'rsho':6,'lelb':7,'relb':8,'lwri':9,'rwri':10,
      'lhip':11,'rhip':12,'lkne':13,'rkne':14,'lank':15,'rank':16}

SKELETON = [
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
    (0,5),(0,6),
]

WINDOW = 30


def draw_skeleton(img, kxy, kconf, min_conf=0.35):
    for a, b in SKELETON:
        if kconf[a] >= min_conf and kconf[b] >= min_conf:
            pa = tuple(int(v) for v in kxy[a])
            pb = tuple(int(v) for v in kxy[b])
            cv2.line(img, pa, pb, (0, 220, 0), 2)
    for i in range(17):
        if kconf[i] >= min_conf:
            p = tuple(int(v) for v in kxy[i])
            cv2.circle(img, p, 3, (0, 180, 255), -1)


def draw_box(img, box, color, label):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
    cv2.putText(img, label, (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--width',  type=int, default=960)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--no-flip', action='store_true',
                    help='do not mirror the webcam image')
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print('ERROR: cannot open camera. Grant Terminal camera permission.')
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print('loading pose model...')
    pose = YOLO('yolov8n-pose.pt')

    # per-track state: we only track one person in this viewer, the one
    # with the highest detection confidence.
    feat_buf = deque(maxlen=WINDOW)
    prev_state = None
    last_t = None
    fall_frames = 0
    normal_frames = 0
    alarmed = False

    print('running. press q or ESC to quit, s to save a frame.')

    fps_t0 = time.time()
    fps_n = 0
    fps_display = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print('frame read failed')
            break

        if not args.no_flip:
            frame = cv2.flip(frame, 1)   # mirror like a selfie

        now = time.time()
        dt = (now - last_t) if last_t else 1/15.0
        dt = float(np.clip(dt, 1/60.0, 0.5))
        last_t = now

        res = pose.predict(frame, verbose=False, conf=0.35,
                           imgsz=640, device='cpu')[0]

        overlay_lines = []
        overlay_lines.append(f"FPS: {fps_display:.1f}")

        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            overlay_lines.append("NO PERSON DETECTED")
            overlay_lines.append("move into frame / turn on the light")
            # decay fall counters
            fall_frames = max(0, fall_frames - 1)
            normal_frames += 1
            if alarmed and normal_frames >= 30:
                alarmed = False
        else:
            # take the highest-confidence person
            confs = res.boxes.conf.cpu().numpy()
            i = int(np.argmax(confs))
            box = tuple(res.boxes.xyxy.cpu().numpy()[i].tolist())
            kxy = res.keypoints.xy.cpu().numpy()[i].astype(np.float32)
            kc = (res.keypoints.conf.cpu().numpy()[i].astype(np.float32)
                  if res.keypoints.conf is not None
                  else np.ones(17, dtype=np.float32))

            draw_skeleton(frame, kxy, kc)

            feat, prev_state = extract_features(kxy, kc, box, prev_state, dt=dt)
            feat_buf.append(feat)

            is_fall = False
            score = 0.0
            debug = {'reason': 'warmup'}

            if len(feat_buf) >= 5:
                window = np.stack(feat_buf)
                is_fall, score, debug = rule_based_fall(window)

            if is_fall:
                fall_frames += 1
                normal_frames = 0
            else:
                normal_frames += 1
                fall_frames = max(0, fall_frames - 1)

            if not alarmed and fall_frames >= 5:
                alarmed = True
            if alarmed and normal_frames >= 30:
                alarmed = False

            color = (0, 0, 255) if alarmed else (0, 200, 0)
            label = f"FALL  {score:.2f}" if alarmed else f"ok  {score:.2f}"
            draw_box(frame, box, color, label)

            overlay_lines.append(f"det conf:  {float(confs[i]):.2f}")
            overlay_lines.append(f"keypoint conf: {float(np.mean(kc)):.2f}")
            overlay_lines.append(f"torso_angle: {debug.get('torso', '-')}")
            overlay_lines.append(f"body_angle:  {debug.get('body', '-')}")
            overlay_lines.append(f"aspect:      {debug.get('aspect', '-')}")
            overlay_lines.append(f"vy:          {debug.get('vy', '-')}")
            overlay_lines.append(f"prev_vy:     {debug.get('prev_vy', '-')}")
            overlay_lines.append(f"reason:      {debug.get('reason', '-')}")
            overlay_lines.append(f"fall_frames: {fall_frames} / 5")
            overlay_lines.append(f"normal_frames: {normal_frames} / 30")
            overlay_lines.append(f"ALARMED: {alarmed}")

        # big banner
        banner = "FALL DETECTED" if alarmed else "monitoring"
        bcolor = (0, 0, 255) if alarmed else (200, 200, 200)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, banner, (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, bcolor, 2)

        # overlay text panel (left side, below banner)
        y = 70
        for line in overlay_lines:
            cv2.putText(frame, line, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
            y += 22

        cv2.imshow('Fall Detector Live View', frame)

        # fps calc
        fps_n += 1
        if now - fps_t0 > 1.0:
            fps_display = fps_n / (now - fps_t0)
            fps_n = 0
            fps_t0 = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('s'):
            cv2.imwrite('debug_live.jpg', frame)
            print('saved debug_live.jpg')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()