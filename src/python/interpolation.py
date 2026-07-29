"""Step keyframe interpolation — mirrors interpolation.ts exactly."""
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

    Step interpolation: returns the position from the last keyframe at or
    before t_ms — matching getBoxPositionAt in interpolation.ts exactly.
    Hard cuts only, no smooth panning between keyframes.
    """
    if not keyframes:
        return BoxPos(x=0.0, y=0.0, w=1.0, h=1.0)

    sorted_kf = sorted(keyframes, key=lambda k: k["t_ms"])

    active = sorted_kf[0]
    for kf in sorted_kf:
        if kf["t_ms"] <= t_ms:
            active = kf
        else:
            break

    return BoxPos(x=active["x"], y=active["y"], w=active["w"], h=active["h"])
