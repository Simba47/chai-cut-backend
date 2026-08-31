"""
Subject detection on a directory of JPEG frames.

Per-frame pipeline (in order):
  1. Haar frontal-face cascade  — precise positions for forward-facing subjects
  2. Wide-blob analysis          — when blob width >= 35% of frame, treat as 2+ people
  3. Background subtraction      — fallback for side-on/full-body subjects

Output: JSON to stdout:
  {
    "face_count": N,             # max person count seen across all frames
    "face_boxes": [{x,y,w,h}],  # averaged positions per slot (for backward compat)
    "frames": [
      {
        "frame_index": 0,
        "person_count": N,       # how many distinct people are in this frame
        "faces": [{x,y,w,h}]    # positions of detected subjects, left→right
      }, ...
    ]
  }
  All coordinates are fractions of the frame (0.0–1.0).
"""
import argparse
import json
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_PATH = os.path.join(SCRIPT_DIR, 'haarcascade_frontalface_default.xml')



def _normalize(x, y, w, h, w_px, h_px):
    return {"x": x / w_px, "y": y / h_px, "w": w / w_px, "h": h / h_px,
            "cx": (x + w / 2) / w_px, "area": (w / w_px) * (h / h_px)}


def _filter_by_size(boxes, min_rel=0.4):
    """Drop boxes smaller than min_rel * largest box area (removes false positives)."""
    if not boxes:
        return []
    max_area = max(b["area"] for b in boxes)
    return [b for b in boxes if b["area"] >= min_rel * max_area]


def _detect_haar(cascade, gray, w_px, h_px):
    rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(50, 50))
    boxes = [_normalize(x, y, w, h, w_px, h_px) for (x, y, w, h) in (rects if len(rects) > 0 else [])]
    boxes.sort(key=lambda b: -b["area"])
    return _filter_by_size(boxes)


def _motion_center(prev_gray, curr_gray, w_px, h_px, threshold=18):
    """
    Find the horizontal center of inter-frame motion using pixel difference.

    Compares consecutive frames — no background model needed, no scene-change
    contamination. A talking / moving person registers as motion; a static
    backdrop does not. Returns normalised cx (0-1) or None if motion is too
    small to be meaningful.
    """
    import cv2

    diff = cv2.absdiff(prev_gray, curr_gray)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    # Dilate to connect nearby moving pixels (e.g. hand + face)
    mask = cv2.dilate(mask, np.ones((9, 9), "uint8"), iterations=2)

    n_pixels = int(mask.sum() / 255)
    if n_pixels < 200:  # too few moving pixels — treat as static frame
        return None

    ys, xs = np.where(mask > 0)
    return float(xs.mean()) / w_px


def detect_with_opencv(frames_dir: str) -> dict:
    import cv2

    cascade = None
    if os.path.exists(CASCADE_PATH):
        cascade = cv2.CascadeClassifier(CASCADE_PATH)

    frame_files = sorted(
        f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    if not frame_files:
        return {"face_count": 0, "face_boxes": [], "frames": []}

    # Load all frames
    loaded = []
    for fname in frame_files:
        img = cv2.imread(os.path.join(frames_dir, fname))
        if img is None:
            continue
        h_px, w_px = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        loaded.append((gray, w_px, h_px))

    if not loaded:
        return {"face_count": 0, "face_boxes": [], "frames": []}

    w_px, h_px = loaded[0][1], loaded[0][2]

    # ── Per-frame analysis ─────────────────────────────────────────────────────
    per_frame = []
    slot_data: list[list[dict]] = [[] for _ in range(4)]
    max_person_count = 0

    for fi, (gray, wp, hp) in enumerate(loaded):
        subjects = []

        # Step 1: Haar frontal faces — sole source of truth for person COUNT.
        # Background subtraction is NOT used for counting: global median
        # backgrounds contaminate mixed scenes (e.g., judges panel + solo
        # performer in same clip) and produce wildly incorrect foreground masks.
        if cascade is not None:
            subjects = _detect_haar(cascade, gray, wp, hp)

        # Step 2: Position fallback — when Haar finds nobody (non-frontal
        # subject), use frame-differencing against the PREVIOUS frame to find
        # WHERE motion is happening right now. This is scene-change safe: we
        # compare frame[i] vs frame[i-1], no global background model needed.
        # A talking / moving person registers as motion; static backdrop does not.
        if len(subjects) == 0 and fi > 0:
            prev_gray = loaded[fi - 1][0]
            cx = _motion_center(prev_gray, gray, wp, hp)
            if cx is not None:
                # Virtual face box centred on the motion — count treated as 1 person.
                subjects = [{"x": max(0, cx - 0.1), "y": 0.0,
                             "w": 0.2, "h": 1.0,
                             "cx": cx, "area": 0.05}]

        # Prominence filter: if one subject is 2× larger than all others the
        # camera is on a close-up of the speaker; others are background.
        if len(subjects) >= 2:
            by_area = sorted(subjects, key=lambda s: -s["area"])
            if by_area[0]["area"] >= 2.0 * by_area[1]["area"]:
                subjects = [by_area[0]]

        # Proximity filter: if the 2 closest detected faces are within 15% of
        # frame width they can't be split into meaningful individual crops.
        if len(subjects) >= 2:
            by_cx = sorted(subjects, key=lambda s: s["cx"])
            min_gap = min(by_cx[i+1]["cx"] - by_cx[i]["cx"] for i in range(len(by_cx)-1))
            if min_gap < 0.15:
                subjects = [max(subjects, key=lambda s: s["area"])]

        # Sort left → right
        subjects.sort(key=lambda b: b["cx"])

        person_count = len(subjects)
        max_person_count = max(max_person_count, person_count)

        # ── Per-frame log ──────────────────────────────────────────────────────
        haar_raw = _detect_haar(cascade, gray, wp, hp) if cascade is not None else []
        log_method = "haar" if haar_raw else ("motion" if (fi > 0 and person_count > 0) else "none")
        cx_list = [f"{s['cx']:.2f}" for s in subjects]
        print(
            f"[detect] frame={fi:02d} haar={len(haar_raw)} "
            f"after_filters={person_count} method={log_method} "
            f"cx=[{','.join(cx_list)}]",
            file=sys.stderr,
        )

        per_frame.append({
            "frame_index": fi,
            "person_count": person_count,
            "faces": [{"x": s["x"], "y": s["y"], "w": s["w"], "h": s["h"]}
                      for s in subjects[:4]],
        })

        for i, s in enumerate(subjects[:4]):
            slot_data[i].append({"x": s["x"], "y": s["y"], "w": s["w"], "h": s["h"]})

    # ── Averaged boxes (backward compat) ──────────────────────────────────────
    face_boxes = []
    for i in range(max_person_count):
        samples = slot_data[i]
        if not samples:
            continue
        face_boxes.append({
            "x": sum(s["x"] for s in samples) / len(samples),
            "y": sum(s["y"] for s in samples) / len(samples),
            "w": sum(s["w"] for s in samples) / len(samples),
            "h": sum(s["h"] for s in samples) / len(samples),
        })

    return {"face_count": max_person_count, "face_boxes": face_boxes, "frames": per_frame}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    args = parser.parse_args()
    try:
        result = detect_with_opencv(args.frames_dir)
    except Exception as e:
        print(f"[face_detect] Error: {e}", file=sys.stderr)
        result = {"face_count": 0, "face_boxes": [], "frames": []}
    print(json.dumps(result))
