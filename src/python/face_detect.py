"""
Subject detection on a directory of JPEG frames.

Per-frame pipeline (in order):
  1. MediaPipe FaceDetector (BlazeFace short-range, Apache-2.0) on the whole frame and on square
     tiles of it: the model sees a 128×128 image, so on a wide frame the tiles keep faces big
     enough to find. Catches side-on and tilted faces the Haar cascade missed.
  2. Haar frontal-face cascade  — only when MediaPipe finds nothing, or can't be imported
  3. Frame differencing (motion) — position fallback for subjects with no detectable face
Faces that never change from frame to frame (a poster or photo on the wall) are dropped: the
detectors find them as readily as people, and they would pull the crop to a split layout.

Output: JSON to stdout:
  {
    "face_count": N,             # max person count seen across all frames
    "face_boxes": [{x,y,w,h}],  # averaged positions per slot (for backward compat)
    "frames": [
      {
        "frame_index": 0,
        "person_count": N,       # how many distinct people are in this frame
        "faces": [{x,y,w,h,score,lip}]  # positions of detected subjects, left→right
      }, ...
    ]
  }
  All coordinates are fractions of the frame (0.0–1.0). score is the detection confidence
  (MediaPipe), 1.0 for Haar faces and 0.0 for motion boxes. With --lips, "lip" is how open the
  mouth is (inner lip gap / face height), given only on frames with 2+ faces.
"""
import argparse
import json
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_PATH = os.path.join(SCRIPT_DIR, 'haarcascade_frontalface_default.xml')
BLAZEFACE_PATH = os.path.join(SCRIPT_DIR, 'blaze_face_short_range.tflite')
LANDMARKER_PATH = os.path.join(SCRIPT_DIR, 'face_landmarker.task')

MIN_SCORE = 0.5        # ignore less confident MediaPipe faces
MIN_FACE_AREA = 0.02   # ignore faces smaller than 2% of the frame (background people)


def _normalize(x, y, w, h, w_px, h_px, score=1.0):
    return {"x": x / w_px, "y": y / h_px, "w": w / w_px, "h": h / h_px,
            "cx": (x + w / 2) / w_px, "area": (w / w_px) * (h / h_px), "score": score}


def _filter_by_size(boxes, min_rel=0.4):
    """Drop boxes smaller than min_rel * largest box area (removes false positives)."""
    if not boxes:
        return []
    max_area = max(b["area"] for b in boxes)
    return [b for b in boxes if b["area"] >= min_rel * max_area]


def _iou(a, b):
    ix = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0.0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    inter = ix * iy
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes, iou=0.3):
    """The same face found in the whole frame and in a tile: keep the most confident one."""
    kept = []
    for b in sorted(boxes, key=lambda b: -b["score"]):
        if all(_iou(b, k) <= iou for k in kept):
            kept.append(b)
    return kept


class _MediaPipe:
    def __init__(self):
        os.environ.setdefault("MPLBACKEND", "Agg")  # mediapipe's drawing helpers import matplotlib
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        import mediapipe as mp
        self._mp = mp
        self._detector = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=BLAZEFACE_PATH),
            min_detection_confidence=MIN_SCORE,
        ))

    def _run(self, rgb):
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        return self._detector.detect(image).detections

    def detect(self, bgr):
        """Faces in the whole frame plus square tiles across it (for wide frames)"""
        rgb = bgr[:, :, ::-1]
        h_px, w_px = rgb.shape[:2]
        views = [(0, rgb)]
        if w_px > h_px * 1.2:
            n = int(np.ceil(w_px / h_px)) + 1          # 16:9 → 3 overlapping squares
            for i in range(n):
                x0 = round(i * (w_px - h_px) / (n - 1))
                views.append((x0, rgb[:, x0:x0 + h_px]))
        boxes = []
        for x0, view in views:
            for d in self._run(view):
                bb = d.bounding_box
                score = float(d.categories[0].score) if d.categories else 0.0
                x = max(0, bb.origin_x) + x0
                y = max(0, bb.origin_y)
                w = min(bb.width, w_px - x)
                h = min(bb.height, h_px - y)
                if w <= 0 or h <= 0:
                    continue
                boxes.append(_normalize(x, y, w, h, w_px, h_px, score))
        boxes = [b for b in _nms(boxes) if b["score"] >= MIN_SCORE and b["area"] >= MIN_FACE_AREA]
        boxes.sort(key=lambda b: -b["area"])
        return _filter_by_size(boxes)


class _Lips:
    """
    How open each face's mouth is (MediaPipe FaceLandmarker, Apache-2.0): the gap between the
    inner upper and lower lip divided by the face's height. Its change over time shows who is
    talking. Runs on a crop around each detected face, so small faces in a wide shot still get
    enough pixels.
    """
    UPPER, LOWER, TOP, CHIN = 13, 14, 10, 152

    def __init__(self):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        import mediapipe as mp
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=LANDMARKER_PATH), num_faces=1,
        ))

    def gap(self, bgr, face):
        import cv2
        h_px, w_px = bgr.shape[:2]
        # The face box plus a margin, as a square
        side = max(face["w"] * w_px, face["h"] * h_px) * 1.6
        cx, cy = (face["x"] + face["w"] / 2) * w_px, (face["y"] + face["h"] / 2) * h_px
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x1, y1 = int(min(w_px, cx + side / 2)), int(min(h_px, cy + side / 2))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        crop = cv2.resize(bgr[y0:y1, x0:x1], (256, 256), interpolation=cv2.INTER_LINEAR)
        rgb = np.ascontiguousarray(crop[:, :, ::-1])
        res = self._landmarker.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks:
            return None
        lm = res.face_landmarks[0]
        dist = lambda a, b: float(np.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y))
        height = dist(self.TOP, self.CHIN)
        return round(dist(self.UPPER, self.LOWER) / height, 4) if height > 0 else None


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


STATIC_RATIO = 0.85     # a face changing less than this × its surroundings is a picture, not a person
# …or less than its surroundings at all while MediaPipe is unsure it is a face: posters and photos
# score 0.81–0.86 on the samples, live faces 0.90–0.98 (a motionless video-call face: 1.04, 0.965)
STATIC_RATIO_UNSURE = 1.0
UNSURE_SCORE = 0.9
STATIC_MIN_FRAMES = 8   # judged only on faces seen for at least this many frames (2 s at 4 fps)


def _change_ratio(prev_gray, gray, b, margin=0.6):
    """
    How much a face box changed since the previous frame, relative to the ring of picture around
    it. A live face (blinking, talking, breathing) changes more than the wall behind it, even on
    a very still video call; a poster or photo changes no more than its surroundings, even when
    someone's arm moves in front of it.
    """
    h, w = gray.shape[:2]
    x0, y0 = int(b["x"] * w), int(b["y"] * h)
    x1, y1 = max(x0 + 1, int((b["x"] + b["w"]) * w)), max(y0 + 1, int((b["y"] + b["h"]) * h))
    mx, my = int(b["w"] * w * margin), int(b["h"] * h * margin)
    X0, Y0, X1, Y1 = max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)
    d = np.abs(gray[Y0:Y1, X0:X1].astype(np.int16) - prev_gray[Y0:Y1, X0:X1].astype(np.int16)).astype(np.float32)
    inner = d[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
    ring = (d.sum() - inner.sum()) / max(1, d.size - inner.size)
    return float(inner.mean() / (ring + 0.05))


def track_faces(frames: list[list[dict]], iou=0.3, resets: set[int] | None = None) -> list[list[int]]:
    """
    Gives every face a track id that stays the same while it is the same person: a face joins
    the track of the box it overlaps most (IoU > iou) in the previous frame. Tracks end at
    frame indexes in `resets` (camera cuts). Returns track ids per frame, parallel to `frames`.
    """
    ids: list[list[int]] = []
    next_id = 0
    for fi, faces in enumerate(frames):
        prev = frames[fi - 1] if fi > 0 and not (resets and fi in resets) else []
        prev_ids = ids[fi - 1] if prev else []
        taken: set[int] = set()
        row = []
        for f in faces:
            best, best_iou = None, iou
            for pi, p in enumerate(prev):
                v = _iou(f, p)
                if v > best_iou and prev_ids[pi] not in taken:
                    best, best_iou = prev_ids[pi], v
            if best is None:
                best = next_id
                next_id += 1
            taken.add(best)
            row.append(best)
        ids.append(row)
    return ids


def detect_subjects(frames_dir: str, lips: bool = False) -> dict:
    import cv2

    cascade = None
    if os.path.exists(CASCADE_PATH):
        cascade = cv2.CascadeClassifier(CASCADE_PATH)

    mediapipe = None
    try:
        mediapipe = _MediaPipe()
    except Exception as e:  # not installed, or the model file is missing: Haar + motion only
        print(f"[detect] MediaPipe unavailable, using Haar: {e}", file=sys.stderr)

    lip_reader = None
    if lips and mediapipe is not None:
        try:
            lip_reader = _Lips()
        except Exception as e:
            print(f"[detect] FaceLandmarker unavailable, no speaker tracking: {e}", file=sys.stderr)

    frame_files = sorted(
        f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    if not frame_files:
        return {"face_count": 0, "face_boxes": [], "frames": []}

    # ── Pass 1: faces per frame, how much each one changed since the previous frame, and
    #    where motion is (the fallback when no face is left) ─────────────────────────────
    raw: list[list[dict]] = []
    paths: list[str] = []
    motion_cx: list[float | None] = []
    methods: list[str] = []
    prev_gray = None
    for fi, fname in enumerate(frame_files):
        img = cv2.imread(os.path.join(frames_dir, fname))
        if img is None:
            continue
        hp, wp = img.shape[:2]
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        subjects = []
        method = "none"

        # MediaPipe faces — the main source of truth for person COUNT
        if mediapipe is not None:
            try:
                subjects = mediapipe.detect(img)
                if subjects:
                    method = "mediapipe"
            except Exception as e:
                print(f"[detect] MediaPipe failed on frame {fi}: {e}", file=sys.stderr)

        # Haar frontal faces when MediaPipe found nobody
        if not subjects and cascade is not None:
            subjects = _detect_haar(cascade, gray, wp, hp)
            if subjects:
                method = "haar"

        same_shot = prev_gray is not None and prev_gray.shape == gray.shape
        for s in subjects:
            s["change"] = _change_ratio(prev_gray, gray, s) if same_shot else None
        # Frame-differencing against the PREVIOUS frame finds WHERE motion is happening right
        # now (scene-change safe: no global background model). A talking / moving person
        # registers as motion; a static backdrop does not.
        motion_cx.append(_motion_center(prev_gray, gray, wp, hp) if same_shot else None)
        prev_gray = gray
        raw.append(subjects)
        paths.append(os.path.join(frames_dir, fname))
        methods.append(method)

    # ── Pass 2: drop faces that never change — posters, photos, paused screens ─────────────
    track_ids = track_faces(raw)
    changes: dict[int, list[float]] = {}
    for faces, ids in zip(raw, track_ids):
        for f, tid in zip(faces, ids):
            if f["change"] is not None:
                changes.setdefault(tid, []).append(f["change"])
    scores: dict[int, list[float]] = {}
    for faces, ids in zip(raw, track_ids):
        for f, tid in zip(faces, ids):
            scores.setdefault(tid, []).append(f["score"])
    static = set()
    for tid, d in changes.items():
        if len(d) < STATIC_MIN_FRAMES:
            continue
        ratio, score = float(np.median(d)), float(np.median(scores[tid]))
        if ratio < STATIC_RATIO or (ratio < STATIC_RATIO_UNSURE and score < UNSURE_SCORE):
            static.add(tid)

    # ── Pass 3: per-frame people, with the layout filters ────────────────────────────────
    per_frame = []
    slot_data: list[list[dict]] = [[] for _ in range(4)]
    max_person_count = 0
    counts_by_method = {"mediapipe": 0, "haar": 0, "motion": 0, "none": 0}
    lip_frames = 0
    for fi, (faces, ids) in enumerate(zip(raw, track_ids)):
        subjects = [f for f, tid in zip(faces, ids) if tid not in static]
        method = methods[fi] if subjects else "none"

        # Position fallback — no (live) face: follow the motion, counted as 1 person
        if not subjects and motion_cx[fi] is not None:
            cx = motion_cx[fi]
            subjects = [{"x": max(0, cx - 0.1), "y": 0.0,
                         "w": 0.2, "h": 1.0,
                         "cx": cx, "area": 0.05, "score": 0.0}]
            method = "motion"
        counts_by_method[method] += 1

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

        # Mouth opening, only where there is more than one person to choose between
        lip_values = [None] * len(subjects)
        real = [s for s in subjects if s["score"] > 0]
        if lip_reader is not None and len(real) >= 2:
            img = cv2.imread(paths[fi])
            for i, s in enumerate(subjects):
                if s["score"] > 0:
                    try:
                        lip_values[i] = lip_reader.gap(img, s)
                    except Exception as e:
                        print(f"[detect] FaceLandmarker failed on frame {fi}: {e}", file=sys.stderr)
            lip_frames += 1

        person_count = len(subjects)
        max_person_count = max(max_person_count, person_count)

        per_frame.append({
            "frame_index": fi,
            "person_count": person_count,
            "faces": [{"x": s["x"], "y": s["y"], "w": s["w"], "h": s["h"], "score": round(s["score"], 3),
                       **({"lip": lip_values[i]} if lip_values[i] is not None else {})}
                      for i, s in enumerate(subjects[:4])],
        })

        for i, s in enumerate(subjects[:4]):
            slot_data[i].append({"x": s["x"], "y": s["y"], "w": s["w"], "h": s["h"]})

    counts = [f["person_count"] for f in per_frame] or [0]
    print(
        f"[detect] {len(per_frame)} frames, faces/frame min={min(counts)} "
        f"avg={sum(counts) / len(counts):.2f} max={max(counts)}, by method {counts_by_method}, "
        f"static faces dropped: {len(static)}, lips read on {lip_frames} frames",
        file=sys.stderr,
    )

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
    parser.add_argument("--lips", action="store_true", help="also measure mouth opening (speaker tracking)")
    args = parser.parse_args()
    try:
        result = detect_subjects(args.frames_dir, lips=args.lips)
    except Exception as e:
        print(f"[face_detect] Error: {e}", file=sys.stderr)
        result = {"face_count": 0, "face_boxes": [], "frames": []}
    print(json.dumps(result))
