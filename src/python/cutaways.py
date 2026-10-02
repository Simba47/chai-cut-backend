"""
The visuals inside a video: stretches with no face on screen (screen recordings, product shots,
slides, cutaway footage) — candidates to show under a speaker when they talk about that thing.

Only key frames are decoded (fast: about 2 s per 10 minutes of video), each checked for faces
with OpenCV YuNet. Neighbouring faceless key frames form one visual; each gets one picture (its
middle key frame) for describing it.

Output: JSON to stdout:
  [{"start_ms": …, "end_ms": …, "image": "<path of a 384 px wide JPEG>"}, …]
"""
import argparse
import json
import os
import re
import subprocess
import sys

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YUNET_PATH = os.path.join(SCRIPT_DIR, 'face_detection_yunet_2023mar.onnx')

MIN_VISUAL_MS = 1500     # shorter faceless stretches are transitions, not visuals
MAX_KEY_GAP_MS = 6000    # key frames further apart than this don't join into one visual
MAX_VISUALS = 60         # the longest ones are kept
MIN_DETAIL = 12.0        # a picture with less contrast (grey std) than this is blank / a fade
PERSON_FACE = 0.006      # a face this share of the frame or bigger is a person on camera


def find_visuals(video: str, out_dir: str) -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, 'key_%05d.jpg')
    proc = subprocess.run(
        ['ffmpeg', '-hide_banner', '-nostats', '-skip_frame', 'nokey', '-i', video, '-an',
         '-vf', 'scale=640:-1,showinfo', '-fps_mode', 'vfr', '-q:v', '4', '-y', pattern],
        capture_output=True, text=True, encoding='utf-8', errors='replace')
    times = [float(t) for t in re.findall(r'pts_time:([0-9.]+)', proc.stderr)]
    files = sorted(f for f in os.listdir(out_dir) if f.startswith('key_') and f.endswith('.jpg'))
    if not files:
        return []
    duration_ms = None
    m = re.search(r'Duration: (\d+):(\d+):([\d.]+)', proc.stderr)
    if m:
        duration_ms = int((int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))) * 1000)

    det = cv2.FaceDetectorYN.create(YUNET_PATH, '', (320, 320), 0.6, 0.3, 5000)
    keys = []   # (t_ms, path, faceless)
    for f, t in zip(files, times):
        path = os.path.join(out_dir, f)
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        det.setInputSize((w, h))
        _, faces = det.detect(img)
        detail = float(np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        # No person on screen: no face of real size (a tiny one can be in a graphic), and not
        # several small ones (a wide shot of a group)
        areas = [] if faces is None else [float(f[2] * f[3]) / (w * h) for f in faces]
        faceless = not any(a >= PERSON_FACE for a in areas) and len(areas) < 2
        keys.append((int(t * 1000), path, faceless and detail >= MIN_DETAIL))

    visuals = []
    i = 0
    while i < len(keys):
        if not keys[i][2]:
            i += 1
            continue
        j = i
        while j + 1 < len(keys) and keys[j + 1][2] and keys[j + 1][0] - keys[j][0] <= MAX_KEY_GAP_MS:
            j += 1
        start = keys[i][0]
        # It lasts until the next key frame (with a face), within reason
        nxt = keys[j + 1][0] if j + 1 < len(keys) else (duration_ms or keys[j][0] + 2000)
        end = min(nxt, keys[j][0] + MAX_KEY_GAP_MS)
        if end - start >= MIN_VISUAL_MS:
            mid = keys[(i + j) // 2][1]
            small = os.path.join(out_dir, f'visual_{len(visuals):03d}.jpg')
            img = cv2.imread(mid)
            h, w = img.shape[:2]
            cv2.imwrite(small, cv2.resize(img, (384, max(1, int(h * 384 / w)))), [cv2.IMWRITE_JPEG_QUALITY, 80])
            visuals.append({'start_ms': start, 'end_ms': end, 'image': small})
        i = j + 1

    for f in files:   # the key frames themselves aren't needed any more
        try:
            os.remove(os.path.join(out_dir, f))
        except OSError:
            pass
    visuals.sort(key=lambda v: v['start_ms'] - v['end_ms'])
    visuals = sorted(visuals[:MAX_VISUALS], key=lambda v: v['start_ms'])
    print(f'[visuals] {len(keys)} key frames, {len(visuals)} visual(s) without faces', file=sys.stderr)
    return visuals


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(find_visuals(args.video, args.out_dir)))
    except Exception as e:
        print(f'[visuals] Error: {e}', file=sys.stderr)
        print('[]')
