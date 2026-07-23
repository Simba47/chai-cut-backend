"""Linear keyframe interpolation — mirrors interpolation.ts exactly."""
from __future__ import annotations
from typing import TypedDict


class BoxPos(TypedDict):
    x: float
    y: float
    w: float
    h: float


def get_box_position_at(t_ms: int, keyframes: list[dict]) -> BoxPos:
    """
    t_ms       : current frame time in milliseconds
    keyframes  : list of {t_ms, x, y, w, h} dicts, may be unsorted
    Returns normalized (0-1) box position/size.
    """
    if not keyframes:
        return BoxPos(x=0.0, y=0.0, w=1.0, h=1.0)

    sorted_kf = sorted(keyframes, key=lambda k: k["t_ms"])

    # Clamp before first
    if t_ms <= sorted_kf[0]["t_ms"]:
        k = sorted_kf[0]
        return BoxPos(x=k["x"], y=k["y"], w=k["w"], h=k["h"])

    # Clamp after last
    if t_ms >= sorted_kf[-1]["t_ms"]:
        k = sorted_kf[-1]
        return BoxPos(x=k["x"], y=k["y"], w=k["w"], h=k["h"])

    # Find surrounding pair and interpolate
    for i in range(len(sorted_kf) - 1):
        lo = sorted_kf[i]
        hi = sorted_kf[i + 1]
        if lo["t_ms"] <= t_ms <= hi["t_ms"]:
            rng = hi["t_ms"] - lo["t_ms"]
            alpha = 0.0 if rng == 0 else (t_ms - lo["t_ms"]) / rng
            return BoxPos(
                x=lo["x"] + (hi["x"] - lo["x"]) * alpha,
                y=lo["y"] + (hi["y"] - lo["y"]) * alpha,
                w=lo["w"] + (hi["w"] - lo["w"]) * alpha,
                h=lo["h"] + (hi["h"] - lo["h"]) * alpha,
            )

    # Fallback (should not reach)
    k = sorted_kf[-1]
    return BoxPos(x=k["x"], y=k["y"], w=k["w"], h=k["h"])
