import time, os
import cv2
import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from detector_rule import RuleFallDetector

app = FastAPI(title='Fall Detection Server')
detector = RuleFallDetector(pose_weights='yolov8n-pose.pt', device='cpu')

os.makedirs('snapshots', exist_ok=True)
STATS = {'frames': 0, 'fails': 0, 'alarms': 0, 't0': time.time()}
_last_alert = {}

LAST_DEBUG = {'ts': 0, 'persons': []}


@app.post('/frame')
async def frame(request: Request):
    raw = await request.body()
    if not raw:
        STATS['fails'] += 1
        return JSONResponse({'error': 'empty'}, status_code=400)

    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        STATS['fails'] += 1
        return JSONResponse({'error': 'decode_failed'}, status_code=400)

    device_id = request.headers.get('X-Device-Id', 'unknown')
    STATS['frames'] += 1

    out = detector.process_frame(img)

    # stash the latest per-person debug info for /debug
    LAST_DEBUG['ts'] = out['ts']
    LAST_DEBUG['persons'] = [
        {
            'track_id': p['track_id'],
            'class':    p['class'],
            'fall':     p['fall'],
            'score':    p['score'],
            'debug':    p.get('debug', {}),
            'box':      p['box'],
        }
        for p in out['persons']
    ]

    fall = False
    best = 0.0
    for p in out['persons']:
        if p['fall']:
            fall = True
            best = max(best, p['score'])
        if p['event'] == 'FALL_CONFIRMED':
            STATS['alarms'] += 1
            key = f"{device_id}:{p['track_id']}"
            if time.time() - _last_alert.get(key, 0) > 20:
                _last_alert[key] = time.time()
                ts = time.strftime('%Y%m%d-%H%M%S')
                path = f"snapshots/fall_{ts}_track{p['track_id']}.jpg"
                cv2.imwrite(path, img)
                print(f"\n🚨 FALL CONFIRMED  device={device_id}  "
                      f"track={p['track_id']}  score={p['score']}  -> {path}\n")

    return {
        'fall': fall,
        'score': round(best, 3),
        'persons': len(out['persons']),
        'ts': out['ts'],
    }


@app.get('/health')
def health():
    up = time.time() - STATS['t0']
    return {
        'ok': True,
        'frames': STATS['frames'],
        'fails':  STATS['fails'],
        'alarms': STATS['alarms'],
        'fps':    round(STATS['frames'] / max(up, 1), 2),
        'uptime': round(up, 1),
    }


@app.get('/debug')
def debug():
    return LAST_DEBUG