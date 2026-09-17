import numpy as np

KP = {'nose':0,'leye':1,'reye':2,'lear':3,'rear':4,
      'lsho':5,'rsho':6,'lelb':7,'relb':8,'lwri':9,'rwri':10,
      'lhip':11,'rhip':12,'lkne':13,'rkne':14,'lank':15,'rank':16}

FEATURE_DIM = 12

def _mid(a, b):
    return 0.5 * (a + b)

def _angle_from_vertical(v):
    n = np.linalg.norm(v) + 1e-6
    return float(np.degrees(np.arccos(np.clip(abs(v[1]) / n, -1.0, 1.0))))

def extract_features(kxy, kconf, box, prev_state=None, dt=1/15.0):
    x1, y1, x2, y2 = box
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)

    sho    = _mid(kxy[KP['lsho']], kxy[KP['rsho']])
    hip    = _mid(kxy[KP['lhip']], kxy[KP['rhip']])
    ank    = _mid(kxy[KP['lank']], kxy[KP['rank']])
    head   = kxy[KP['nose']]
    center = _mid(sho, hip)

    torso_angle = _angle_from_vertical(hip - sho)
    body_angle  = _angle_from_vertical(ank - sho)

    aspect   = min(h / w, 4.0)
    sho_hip  = float(np.linalg.norm(hip - sho) / h)
    head_hip = float((hip[1] - head[1]) / h)
    head_sho = float((sho[1] - head[1]) / h)
    cy       = float((center[1] - y1) / h)
    conf     = float(np.mean(kconf)) if kconf is not None else 0.0

    nx = (center[0] - x1) / w
    ny = (center[1] - y1) / h
    if prev_state is None:
        vx = vy = 0.0
    else:
        vx = (nx - prev_state['nx']) / dt
        vy = (ny - prev_state['ny']) / dt
    vx = float(np.clip(vx, -6, 6))
    vy = float(np.clip(vy, -6, 6))

    feat = np.array([
        aspect / 4.0,
        torso_angle / 90.0,
        body_angle  / 90.0,
        np.clip(sho_hip, 0, 2) / 2.0,
        np.clip(head_hip, -1, 1),
        np.clip(head_sho, -1, 1),
        np.clip(cy, 0, 1),
        conf,
        vy / 6.0,
        vx / 6.0,
        np.clip(np.hypot(vx, vy), 0, 6) / 6.0,
        1.0 if torso_angle > 60.0 else 0.0,
    ], dtype=np.float32)

    return feat, {'nx': nx, 'ny': ny}


def rule_based_fall(feat_window):
    """
    Stricter rules:
      - require sustained downward motion for a 'falling' trigger
      - require *strongly* horizontal body for a 'lying' trigger
      - gate on high keypoint confidence
    """
    if len(feat_window) < 5:
        return False, 0.0, {'reason': 'warmup'}

    f_prev  = feat_window[-3]
    f       = feat_window[-1]

    torso_angle = f[1] * 90.0
    body_angle  = f[2] * 90.0
    aspect      = f[0] * 4.0
    vy          = f[8] * 6.0
    prev_vy     = f_prev[8] * 6.0
    conf        = f[7]

    if conf < 0.35:
        return False, 0.0, {'reason': 'low_conf', 'conf': round(conf, 2)}

    lying = (torso_angle > 70.0 and
             body_angle  > 60.0 and
             aspect      < 1.2)

    falling = (vy > 2.5 and
               prev_vy > 2.0 and
               torso_angle > 45.0)

    score = 0.9 if lying else (0.7 if falling else 0.0)
    reason = 'lying' if lying else ('falling' if falling else 'ok')

    return (lying or falling), float(score), {
        'reason':  reason,
        'torso':   round(torso_angle, 1),
        'body':    round(body_angle, 1),
        'aspect':  round(aspect, 2),
        'vy':      round(vy, 2),
        'prev_vy': round(prev_vy, 2),
        'conf':    round(conf, 2),
    }