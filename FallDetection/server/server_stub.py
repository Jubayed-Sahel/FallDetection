"""
Minimal frame receiver. Saves 1 in 30 frames to disk so you can eyeball
image quality, and reports FPS on /health. This is the exact endpoint
the ESP32-CAM firmware will POST to.
"""
import time, os
import cv2
import numpy as np
from fastapi import FastAPI, Request, Response

app = FastAPI()
os.makedirs('incoming', exist_ok=True)

state = {'n': 0, 'fails': 0, 't0': time.time(), 'last': time.time()}

@app.post('/frame')
async def frame(request: Request):
    raw = await request.body()
    if not raw:
        state['fails'] += 1
        return Response(status_code=400)

    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        state['fails'] += 1
        return Response(status_code=400)

    state['n'] += 1
    state['last'] = time.time()

    if state['n'] % 30 == 0:
        cv2.imwrite(f"incoming/frame_{state['n']:06d}.jpg", img)

    return {'ok': True, 'n': state['n']}

@app.get('/health')
def health():
    up = time.time() - state['t0']
    fps = state['n'] / max(up, 1.0)
    return {
        'frames': state['n'],
        'fails':  state['fails'],
        'fps':    round(fps, 2),
        'uptime': round(up, 1),
    }

@app.get('/')
def root():
    return {'msg': 'Fall-detection frame receiver. POST JPEG to /frame.'}