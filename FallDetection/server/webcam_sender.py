"""
Mimics the ESP32-CAM: captures frames from the local webcam and POSTs
JPEG bytes to the server at a target FPS. Same wire format as the firmware.

Usage:
    python webcam_sender.py --server http://127.0.0.1:8000 --fps 8
"""
import argparse, time, sys
import cv2
import requests

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', default='http://127.0.0.1:8000')
    ap.add_argument('--fps', type=float, default=8.0)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--device-id', default='webcam-mac')
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print('ERROR: cannot open camera. Grant Terminal camera access in '
              'System Settings → Privacy & Security → Camera.')
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    url = args.server.rstrip('/') + '/frame'
    headers = {'Content-Type': 'image/jpeg', 'X-Device-Id': args.device_id}
    period = 1.0 / args.fps

    print(f'sending from camera {args.camera} → {url} at {args.fps} FPS')
    print('press Ctrl-C to stop')

    sent = failed = 0
    t_last_report = time.time()

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                failed += 1
                time.sleep(0.05)
                continue

            ok, buf = cv2.imencode('.jpg', frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                failed += 1
                continue

            try:
                r = requests.post(url, data=buf.tobytes(),
                                  headers=headers, timeout=2.0)
                if r.status_code == 200:
                    sent += 1
                else:
                    failed += 1
            except requests.RequestException:
                failed += 1

            if time.time() - t_last_report > 5.0:
                t_last_report = time.time()
                print(f'sent={sent} failed={failed}')

            # pace to target FPS
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print(f'\nstopped. total sent={sent} failed={failed}')

if __name__ == '__main__':
    main()