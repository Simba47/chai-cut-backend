"""
Layout composition functions.
Each function takes the source frame (full-res numpy BGR) and one or more
BoxPos dicts, and returns a composed 9:16 output frame (numpy BGR).
"""
from __future__ import annotations

import cv2
import numpy as np
from interpolation import BoxPos


def _crop(frame: np.ndarray, pos: BoxPos) -> np.ndarray:
    """Crop a normalized region from the source frame."""
    h, w = frame.shape[:2]
    x1 = int(pos["x"] * w)
    y1 = int(pos["y"] * h)
    x2 = int((pos["x"] + pos["w"]) * w)
    y2 = int((pos["y"] + pos["h"]) * h)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return frame[y1:y2, x1:x2]


def compose_vertical(
    canvas: np.ndarray,
    frame: np.ndarray,
    positions: list[BoxPos],
) -> np.ndarray:
    """Single crop fills the full 9:16 canvas."""
    out_h, out_w = canvas.shape[:2]
    crop = _crop(frame, positions[0])
    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)


def compose_split(
    canvas: np.ndarray,
    frame: np.ndarray,
    positions: list[BoxPos],
) -> np.ndarray:
    """Two equal horizontal bands stacked vertically."""
    out_h, out_w = canvas.shape[:2]
    result = canvas.copy()
    slot_h = out_h // 2

    for i, pos in enumerate(positions[:2]):
        crop = _crop(frame, pos)
        resized = cv2.resize(crop, (out_w, slot_h), interpolation=cv2.INTER_LANCZOS4)
        y0 = i * slot_h
        result[y0 : y0 + slot_h, :] = resized

    return result


def compose_trio(
    canvas: np.ndarray,
    frame: np.ndarray,
    positions: list[BoxPos],
) -> np.ndarray:
    """Top half (index 0) + two bottom quarters (indices 1 & 2)."""
    out_h, out_w = canvas.shape[:2]
    result = canvas.copy()
    top_h = int(out_h * 0.55)
    bot_h = out_h - top_h
    half_w = out_w // 2

    if len(positions) > 0:
        crop = _crop(frame, positions[0])
        result[:top_h, :] = cv2.resize(crop, (out_w, top_h), interpolation=cv2.INTER_LANCZOS4)

    for i, pos in enumerate(positions[1:3]):
        crop = _crop(frame, pos)
        resized = cv2.resize(crop, (half_w, bot_h), interpolation=cv2.INTER_LANCZOS4)
        result[top_h:, i * half_w : (i + 1) * half_w] = resized

    return result


def compose_spotlight(
    canvas: np.ndarray,
    frame: np.ndarray,
    positions: list[BoxPos],
) -> np.ndarray:
    """Full-frame fill — same as vertical."""
    return compose_vertical(canvas, frame, positions)


def compose_centered(
    canvas: np.ndarray,
    frame: np.ndarray,
    positions: list[BoxPos],
) -> np.ndarray:
    """Crop centered with letterbox bars if aspect doesn't match."""
    out_h, out_w = canvas.shape[:2]
    result = canvas.copy()
    crop = _crop(frame, positions[0])
    if crop.size == 0:
        return result

    ch, cw = crop.shape[:2]
    scale = min(out_w / cw, out_h / ch)
    dw = int(cw * scale)
    dh = int(ch * scale)
    resized = cv2.resize(crop, (dw, dh), interpolation=cv2.INTER_LANCZOS4)
    x_off = (out_w - dw) // 2
    y_off = (out_h - dh) // 2
    result[y_off : y_off + dh, x_off : x_off + dw] = resized
    return result


def compose_horizontal(
    canvas: np.ndarray,
    frame: np.ndarray,
    positions: list[BoxPos],
) -> np.ndarray:
    """Horizontal crop letterboxed into vertical canvas."""
    return compose_centered(canvas, frame, positions)


LAYOUT_COMPOSERS = {
    "vertical": compose_vertical,
    "split": compose_split,
    "trio": compose_trio,
    "spotlight": compose_spotlight,
    "centered": compose_centered,
    "horizontal": compose_horizontal,
}


def compose(
    layout: str,
    canvas: np.ndarray,
    frame: np.ndarray,
    positions: list[BoxPos],
) -> np.ndarray:
    fn = LAYOUT_COMPOSERS.get(layout, compose_vertical)
    return fn(canvas, frame, positions)
