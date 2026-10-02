"""
The exact box of a graphic panel the video's editor put on screen: the part of the picture that
differs between a moment without the panel and a moment with it (same camera shot), inside the
rough box a vision model gave. Prints {"x","y","w","h"} (fractions of the frame), or the rough box
unchanged when no clear panel shows up in the difference.

  python panel_box.py --video V --without SECONDS --with SECONDS --box x,y,w,h
"""
import argparse
import json

import cv2
import numpy as np


def frame_at(cap, t: float):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, img = cap.read()
    return img if ok else None


def refine(video: str, t_without: float, t_with: float, box: list[float]) -> dict:
    rough = {"x": box[0], "y": box[1], "w": box[2], "h": box[3]}
    cap = cv2.VideoCapture(video)
    a, b = frame_at(cap, t_without), frame_at(cap, t_with)
    cap.release()
    if a is None or b is None or a.shape != b.shape:
        return rough
    h_px, w_px = a.shape[:2]
    scale = 640 / w_px
    a = cv2.resize(a, (640, int(h_px * scale)))
    b = cv2.resize(b, (640, int(h_px * scale)))
    h_px, w_px = a.shape[:2]
    diff = cv2.absdiff(cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (5, 5), 0),
                       cv2.GaussianBlur(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (5, 5), 0))
    mask = (diff > 25).astype(np.uint8) * 255
    # Only inside the rough box (a little bigger): the speaker moving elsewhere doesn't count
    x0, y0 = int(max(0, box[0] - 0.05) * w_px), int(max(0, box[1] - 0.05) * h_px)
    x1, y1 = int(min(1, box[0] + box[2] + 0.05) * w_px), int(min(1, box[1] + box[3] + 0.05) * h_px)
    roi = np.zeros_like(mask)
    roi[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return rough
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    # A real panel fills most of its box and is a good part of the rough one
    if cv2.contourArea(c) < 0.6 * w * h or w * h < 0.3 * (x1 - x0) * (y1 - y0):
        return rough
    return {"x": x / w_px, "y": y / h_px, "w": w / w_px, "h": h / h_px}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--without", type=float, required=True)
    p.add_argument("--with", dest="with_", type=float, required=True)
    p.add_argument("--box", required=True)
    args = p.parse_args()
    box = [float(v) for v in args.box.split(",")]
    try:
        print(json.dumps(refine(args.video, args.without, args.with_, box)))
    except Exception:
        print(json.dumps({"x": box[0], "y": box[1], "w": box[2], "h": box[3]}))
