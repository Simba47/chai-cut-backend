"""
Frame layouts: the 9:16 reel divided top-to-bottom into media slots and an optional letterbox
band for text. Mirrors chai-cut-frontend/src/modules/editor/frames.ts — keep the numbers in step.
"""
from __future__ import annotations

# (kind, slot index or None, share of the frame height). "caption": a plain strip (no lane) where
# the clip's captions sit while the frame shows.
FRAME_TEMPLATES: dict[str, list[tuple[str, int | None, float]]] = {
    "frame_single":         [("band", None, 0.20), ("slot", 0, 0.80)],
    "frame_video_photo":    [("slot", 0, 0.42), ("band", None, 0.16), ("slot", 1, 0.42)],
    "frame_dual":           [("slot", 0, 0.50), ("slot", 1, 0.50)],
    "frame_dual_letterbox": [("slot", 0, 0.42), ("band", None, 0.16), ("slot", 1, 0.42)],
    "frame_triple":         [("slot", 0, 1 / 3), ("slot", 1, 1 / 3), ("slot", 2, 1 / 3)],
    "frame_title_caption":  [("band", None, 0.16), ("slot", 0, 0.68), ("caption", None, 0.16)],
    "frame_big_small":      [("slot", 0, 0.70), ("slot", 1, 0.30)],
    "frame_photo_story":    [("band", None, 0.16), ("slot", 0, 0.42), ("slot", 1, 0.42)],
    # Stacked frames added 2026-10 (their row options are in AREA_LOOKS below)
    "frame_gameplay":       [("slot", 0, 0.50), ("slot", 1, 0.50)],
    "frame_screenshot":     [("slot", 0, 0.42), ("slot", 1, 0.58)],
    "frame_letterbox_wide": [("band", None, 0.34), ("slot", 0, 0.32), ("caption", None, 0.34)],
    "frame_two_crops":      [("slot", 0, 0.50), ("slot", 1, 0.50)],
}

# Options of a stacked frame's rows, by row index: a slot showing its media whole ("contain") on a
# fill colour. Mirrors the rows' fit / fill in the editor's frames.ts.
ROW_LOOKS: dict[str, dict[int, dict]] = {
    "frame_screenshot": {0: {"fit": "contain", "fill": "#111111"}},
}

# Frames whose areas sit anywhere (side by side, a grid, a floating box): shares of the 1080×1920
# frame, drawn in this order (later ones on top). kind "slot" (slot), "band" (lane "band" or
# "band2": a text area), "caption" (a strip for the captions), "wave" (the sound wave). Options:
# fit "contain" (media shown whole), fill (a slot's colour where media doesn't cover it; "none":
# see-through), bg / color / size (a text area's colour when empty and new text's look), radius (px
# at 1080 wide, or "circle"), ticker (the text scrolls). Mirrors FRAME_TEMPLATES[...].areas in the
# editor's frames.ts — keep the numbers in step.
FREE_TEMPLATES: dict[str, list[dict]] = {
    "frame_blur_fill": [
        {"kind": "band", "lane": "band", "x": 0, "y": 0, "w": 1, "h": 0.22, "bg": "none"},
        {"kind": "slot", "slot": 0, "x": 0, "y": 0.22, "w": 1, "h": 0.56, "fit": "contain", "fill": "none"},
        {"kind": "caption", "x": 0, "y": 0.78, "w": 1, "h": 0.22, "bg": "none"},
    ],
    "frame_pip": [
        {"kind": "slot", "slot": 0, "x": 0, "y": 0, "w": 1, "h": 1},
        {"kind": "slot", "slot": 1, "x": 0.58, "y": 0.66, "w": 0.36, "h": 0.2025, "radius": 36},
    ],
    "frame_comment": [
        {"kind": "slot", "slot": 0, "x": 0, "y": 0, "w": 1, "h": 1},
        {"kind": "band", "lane": "band", "x": 0.07, "y": 0.1, "w": 0.86, "h": 0.13, "radius": 28, "bg": "#ffffff", "color": "#111111", "size": 46},
    ],
    # Trio with main: the main video over half the frame, two parts side by side in the other half
    "frame_trio_main": [
        {"kind": "slot", "slot": 0, "x": 0, "y": 0, "w": 1, "h": 0.5},
        {"kind": "slot", "slot": 1, "x": 0, "y": 0.5, "w": 0.5, "h": 0.5},
        {"kind": "slot", "slot": 2, "x": 0.5, "y": 0.5, "w": 0.5, "h": 0.5},
    ],
    "frame_side_by_side": [
        {"kind": "band", "lane": "band", "x": 0, "y": 0, "w": 1, "h": 0.18},
        {"kind": "slot", "slot": 0, "x": 0, "y": 0.18, "w": 0.5, "h": 0.64},
        {"kind": "slot", "slot": 1, "x": 0.5, "y": 0.18, "w": 0.5, "h": 0.64},
        {"kind": "caption", "x": 0, "y": 0.82, "w": 1, "h": 0.18},
    ],
    "frame_name_tags": [
        {"kind": "slot", "slot": 0, "x": 0, "y": 0, "w": 1, "h": 0.5},
        {"kind": "slot", "slot": 1, "x": 0, "y": 0.5, "w": 1, "h": 0.5},
        {"kind": "band", "lane": "band", "x": 0.04, "y": 0.41, "w": 0.56, "h": 0.065, "radius": 14, "bg": "#111111", "size": 44},
        {"kind": "band", "lane": "band2", "x": 0.04, "y": 0.91, "w": 0.56, "h": 0.065, "radius": 14, "bg": "#111111", "size": 44},
    ],
    "frame_brand_card": [
        {"kind": "band", "lane": "band", "x": 0.06, "y": 0.05, "w": 0.88, "h": 0.15, "bg": "none"},
        {"kind": "slot", "slot": 0, "x": 0.06, "y": 0.21, "w": 0.88, "h": 0.6, "radius": 40},
        {"kind": "band", "lane": "band2", "x": 0.06, "y": 0.83, "w": 0.88, "h": 0.1, "bg": "none", "size": 40},
    ],
    "frame_top_list": [
        {"kind": "band", "lane": "band", "x": 0, "y": 0, "w": 1, "h": 0.16},
        {"kind": "slot", "slot": 0, "x": 0, "y": 0.16, "w": 1, "h": 0.84},
        {"kind": "band", "lane": "band2", "x": 0.05, "y": 0.19, "w": 0.2, "h": 0.1, "radius": 24, "bg": "#c8ff00", "color": "#000000", "size": 96},
    ],
    "frame_screen_face": [
        # main_crop: the main video put here (Same video) gets its own view, cropped; other media shows whole
        {"kind": "slot", "slot": 0, "x": 0, "y": 0, "w": 1, "h": 0.62, "fit": "contain", "fill": "#000000", "main_crop": True},
        # The caption strip is see-through (the Screen can reach the bottom); the face cam last: on top
        {"kind": "caption", "x": 0, "y": 0.89, "w": 1, "h": 0.11, "bg": "none"},
        {"kind": "slot", "slot": 1, "x": 0.3, "y": 0.66, "w": 0.4, "h": 0.225, "radius": "circle"},
    ],
    "frame_audiogram": [
        {"kind": "band", "lane": "band", "x": 0, "y": 0, "w": 1, "h": 0.16},
        {"kind": "slot", "slot": 0, "x": 0.15, "y": 0.18, "w": 0.7, "h": 0.39375, "radius": 32},
        {"kind": "wave", "x": 0.1, "y": 0.6, "w": 0.8, "h": 0.12, "color": "#c8ff00"},
        {"kind": "caption", "x": 0, "y": 0.76, "w": 1, "h": 0.2},
    ],
    "frame_grid": [
        {"kind": "slot", "slot": 0, "x": 0, "y": 0, "w": 0.5, "h": 0.5},
        {"kind": "slot", "slot": 1, "x": 0.5, "y": 0, "w": 0.5, "h": 0.5},
        {"kind": "slot", "slot": 2, "x": 0, "y": 0.5, "w": 0.5, "h": 0.5},
        {"kind": "slot", "slot": 3, "x": 0.5, "y": 0.5, "w": 0.5, "h": 0.5},
    ],
    "frame_news": [
        {"kind": "slot", "slot": 0, "x": 0, "y": 0, "w": 1, "h": 1},
        {"kind": "band", "lane": "band", "x": 0, "y": 0.7, "w": 1, "h": 0.09, "bg": "#dc2626", "color": "#ffffff", "size": 56},
        {"kind": "band", "lane": "band2", "x": 0, "y": 0.79, "w": 1, "h": 0.05, "bg": "#111111", "color": "#ffffff", "size": 38, "ticker": True},
    ],
}

# Under a frame's areas: a blurred copy of the main video, or the band's colour (default black)
FRAME_BASE: dict[str, str] = {"frame_blur_fill": "blur", "frame_brand_card": "band"}

MOTIONS = {"none", "zoom_in", "zoom_out", "pan_left", "pan_right"}

# Rounded corners on frame media from the editor's Corners slider (0-100), in px at 1080 wide at
# 100%: (black border around the media, corner radius). Mirrors cornerGeometry() in
# chai-cut-frontend/src/modules/editor/frames.ts ('s'/'m'/'l' from the first version map onto it).
CORNER_MAX = (44, 140)  # (inset, radius) px at 1080 wide: as the editor's CORNER_MAX (radius 96 until 2026-10-17)


def corner_geometry(v) -> tuple[float, float] | None:
    """(border, radius) in px at 1080 wide for a corners value, or None for square corners."""
    n = {"s": 35, "m": 60, "l": 100}.get(v, v) if isinstance(v, str) else v
    try:
        k = max(0.0, min(100.0, float(n or 0))) / 100.0
    except (TypeError, ValueError):
        return None
    return (CORNER_MAX[0] * k, CORNER_MAX[1] * k) if k > 0 else None


def is_frame(layout: str | None) -> bool:
    return bool(layout) and (layout in FRAME_TEMPLATES or layout in FREE_TEMPLATES)


def _even(v: float) -> int:
    return int(round(v / 2)) * 2


# A free frame's bar: the line between two of its areas, dragged to resize them (editor: the
# template's `bar`). dir "v": a vertical line (left | right), "h": a horizontal one (top / bottom);
# at: where the template has it; min / max: how far it can go. frame.split: where it was dragged.
# swap: it has a swap (frame.flip): the areas on either side trade places — the frame mirrored that way.
FREE_BARS = {
    "frame_side_by_side": {"dir": "v", "at": 0.5, "min": 0.2, "max": 0.8},
    "frame_screen_face": {"dir": "h", "at": 0.62, "min": 0.3, "max": 1.0},  # down to the bottom: the Screen fills the frame
    "frame_trio_main": {"dir": "h", "at": 0.5, "min": 0.2, "max": 0.8, "swap": True},  # Main / the pair
}
# A free frame's second bar (frame.split2; its swap frame.flip2). Editor: the template's `bar2`.
FREE_BARS2 = {
    "frame_trio_main": {"dir": "v", "at": 0.5, "min": 0.2, "max": 0.8, "swap": True},  # Left | Right
}
# A free frame's area that can be moved and resized anywhere in the frame, keeping its shape (the
# face cam's circle), even over the others (it's drawn on top). frame.face_box: {x, y, w} once placed;
# until then it follows the bar, keeping its gap. Editor: the template's `movable`.
FREE_MOVABLE = {"frame_screen_face": 1}
MIN_MOVABLE_W = 0.12


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def adjust_free_areas(layout: str, areas: list[dict], frame: dict | None) -> list[dict]:
    """A free frame's areas (shares of the frame) as its bars and moved area set them. Mirrors adjustFreeAreas() in the editor."""
    frame = frame or {}
    # A Branded card with its own look (frame.card): its areas from that look
    if layout == "frame_brand_card":
        look = card_look(frame)
        if look:
            return card_areas(look)
    out = [dict(a) for a in areas]
    bar, bar2, mv = FREE_BARS.get(layout), FREE_BARS2.get(layout), FREE_MOVABLE.get(layout)

    def shift(b: dict, split: float, moving=None) -> None:
        s = min(b["max"], max(b["min"], split))
        at, d = b["at"], s - b["at"]
        pos, size = ("x", "w") if b["dir"] == "v" else ("y", "h")
        for a in out:
            end = a[pos] + a[size]
            if moving is not None and a["kind"] == "slot" and a.get("slot") == moving:
                a[pos] = min(1 - a[size], max(0.0, a[pos] + d))   # follows the bar, inside the frame
            elif abs(end - at) < 1e-6:
                a[size] = s - a[pos]
            elif abs(a[pos] - at) < 1e-6:
                a[pos], a[size] = s, end - s

    split, split2 = _num(frame.get("split")), _num(frame.get("split2"))
    if bar and split is not None:
        shift(bar, split, mv)
    if bar2 and split2 is not None:
        shift(bar2, split2)
    # Swapped: the frame mirrored across the bar's way (before the moved area: placed, it stays put)
    for b, key in ((bar, "flip"), (bar2, "flip2")):
        if b and b.get("swap") and frame.get(key) is True:
            for a in out:
                if b["dir"] == "v":
                    a["x"] = 1 - a["x"] - a["w"]
                else:
                    a["y"] = 1 - a["y"] - a["h"]
    box = frame.get("face_box")
    if mv is not None and isinstance(box, dict):
        for a in out:
            if a["kind"] != "slot" or a.get("slot") != mv:
                continue
            ratio = a["h"] / a["w"]   # its shape: a circle stays a circle
            w = min(1.0, max(MIN_MOVABLE_W, _num(box.get("w")) or a["w"]))
            h = w * ratio
            if h > 1:
                h, w = 1.0, 1 / ratio
            x, y = _num(box.get("x")), _num(box.get("y"))
            a["x"] = min(1 - w, max(0.0, a["x"] if x is None else x))
            a["y"] = min(1 - h, max(0.0, a["y"] if y is None else y))
            a["w"], a["h"] = w, h
    return out


# ── The Branded card's look (frame.card) ─────────────────────────────────────
# Mirrors the editor's lib/shared/brandCard.ts: cardAreas ↔ card_areas, cardPatternHit /
# cardBackgroundPixels ↔ card_bg_filter. Keep them identical. Shares of the frame; pattern sizes
# px at 1080 wide.
_CARD_PATTERNS = ("dots", "grid", "stripes", "checker", "waves")
_CARD_VIDEO_RATIO = 0.6 / 0.88
_CARD_TITLE_H, _CARD_HANDLE_H, _CARD_TITLE_GAP, _CARD_HANDLE_GAP, _CARD_STACK_GAP = 0.15, 0.1, 0.01, 0.02, 0.005


def _card_num(v, lo: float, hi: float, fallback: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return fallback
    return min(hi, max(lo, f)) if f == f and abs(f) != float("inf") else fallback


def clean_background(raw, default_color: str = "#000000") -> dict:
    """A background (frame.bg, or a card look's) made safe (the editor's cleanFrameBackground)"""
    bg = raw if isinstance(raw, dict) else {}
    col = lambda c, f: c.lower() if isinstance(c, str) and len(c) == 7 and c.startswith("#") else f
    return {
        "style": bg.get("style") if bg.get("style") in ("solid", "gradient", "pattern") else "solid",
        "color": col(bg.get("color"), default_color),
        "color2": col(bg.get("color2"), "#2e2e3a"),
        "direction": "diagonal" if bg.get("direction") == "diagonal" else "down",
        "pattern": bg.get("pattern") if bg.get("pattern") in _CARD_PATTERNS else "dots",
        "scale": round(_card_num(bg.get("scale"), 16, 160, 48)),
    }


def frame_background(layout: str, frame: dict | None) -> dict | None:
    """
    The background a frame section shows when it has its own (the editor's frameBackgroundOf): its
    own (frame.bg), else a Branded card's look's; None = as the template has it (band colour, black, blur)
    """
    if layout == "frame_blur_fill":
        return None
    f = frame or {}
    if isinstance(f.get("bg"), dict):
        return clean_background(f["bg"])
    look = card_look(f) if layout == "frame_brand_card" else None
    return look["bg"] if look else None


def card_look(frame: dict | None) -> dict | None:
    """A Branded card section's look (frame.card) made safe; None when it has none (the template's own look)"""
    raw = (frame or {}).get("card")
    if not isinstance(raw, dict):
        return None
    v = raw.get("video") if isinstance(raw.get("video"), dict) else {}
    return {
        "bg": clean_background(raw.get("bg"), "#1c1c22"),
        "video": {
            "width": round(_card_num(v.get("width"), 0.5, 1, 0.88) * 1000) / 1000,
            "y": round(_card_num(v.get("y"), 0, 1, 0.51) * 1000) / 1000,
            "corners": round(_card_num(v.get("corners"), 0, 100, 50)),
        },
        "title_at": "bottom" if raw.get("title_at") == "bottom" else "top",
        "handle_at": "top" if raw.get("handle_at") == "top" else "bottom",
    }


def card_areas(look: dict) -> list[dict]:
    """Where the card's title, video and @handle go (the editor's cardAreas: the same numbers)"""
    vw = min(1.0, max(0.5, look["video"]["width"]))
    vh = vw * _CARD_VIDEO_RATIO
    vx = (1 - vw) / 2
    cy = min(1 - vh / 2, max(vh / 2, look["video"]["y"]))
    vy = cy - vh / 2
    inside = lambda y, h: min(1 - h, max(0.0, y))
    ty = inside(vy - _CARD_TITLE_GAP - _CARD_TITLE_H if look["title_at"] == "top" else vy + vh + _CARD_TITLE_GAP, _CARD_TITLE_H)
    if look["handle_at"] == look["title_at"]:
        hy = ty - _CARD_STACK_GAP - _CARD_HANDLE_H if look["handle_at"] == "top" else ty + _CARD_TITLE_H + _CARD_STACK_GAP
    else:
        hy = vy - _CARD_HANDLE_GAP - _CARD_HANDLE_H if look["handle_at"] == "top" else vy + vh + _CARD_HANDLE_GAP
    return [
        {"kind": "band", "lane": "band", "x": 0.06, "y": ty, "w": 0.88, "h": _CARD_TITLE_H, "bg": "none"},
        {"kind": "slot", "slot": 0, "x": vx, "y": vy, "w": vw, "h": vh, "radius": round(look["video"]["corners"] * 0.8)},
        {"kind": "band", "lane": "band2", "x": 0.06, "y": inside(hy, _CARD_HANDLE_H), "w": 0.88, "h": _CARD_HANDLE_H, "bg": "none", "size": 40},
    ]


def card_bg_filter(look: dict, w: int, h: int, dur_s: float, lbl: str) -> str | None:
    """A Branded card's background (bg_filter of its look's)"""
    return bg_filter(look["bg"], w, h, dur_s, lbl)


def bg_filter(bg: dict, w: int, h: int, dur_s: float, lbl: str) -> str:
    """
    A frame background (clean_background), w × h for dur_s, as one ffmpeg filter chain ending in
    `lbl`: a plain colour, or a gradient / pattern drawn once (geq on a single frame, then held).
    The editor's backgroundPixels: the same formulas.
    """
    if bg["style"] == "solid":
        return f"color=c=0x{bg['color'][1:]}:s={w}x{h}:d={dur_s:.3f}:r=30,format=yuv420p,setsar=1{lbl}"
    a = [int(bg["color"][i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(bg["color2"][i:i + 2], 16) for i in (1, 3, 5)]
    u, v = "(X*1080/W)", "(Y*1080/W)"
    if bg["style"] == "gradient":
        s = "((X/W+Y/H)/2)" if bg["direction"] == "diagonal" else "(Y/H)"
        chans = [f"{a[i]}+({b[i] - a[i]})*{s}" for i in range(3)]
    else:
        p = float(bg["scale"])
        hit = {
            "dots": f"lt(hypot(mod({u},{p})-{p / 2},mod({v},{p})-{p / 2}),{p * 0.18})",
            "grid": f"gt(lt(mod({u},{p}),{max(2.0, p * 0.06)})+lt(mod({v},{p}),{max(2.0, p * 0.06)}),0)",
            "stripes": f"lt(mod({u}+{v},{p}),{p * 0.3})",
            "checker": f"gte(mod(floor({u}/{p})+floor({v}/{p}),2),1)",
            "waves": f"lt(mod({v}+sin({u}/{p}*2*PI)*{p * 0.25},{p}),{max(2.0, p * 0.12)})",
        }[bg["pattern"]]
        chans = [f"if({hit},{b[i]},{a[i]})" for i in range(3)]
    # (commas inside the expressions are escaped for the filtergraph)
    return (f"color=c=black:s={w}x{h}:d=0.04:r=25,format=rgb24,"
            f"geq=r='{chans[0]}':g='{chans[1]}':b='{chans[2]}',"
            f"loop=loop=-1:size=1:start=0,trim=duration={dur_s:.3f},setpts=PTS-STARTPTS,fps=30,format=yuv420p,setsar=1{lbl}")


def frame_areas(layout: str, out_w: int, out_h: int, show_band: bool = True, custom: list[float] | None = None,
                order: list[int] | None = None, frame: dict | None = None) -> list[dict]:
    """
    Every area of a frame in drawing order, in even pixels: {kind, slot / lane, x, y, w, h, r (corner
    radius px), and the area's options}. A stacked frame's rows are full-width areas (their heights
    from frame_rows, so they add up exactly); a free frame's areas are placed where the template says.
    Mirrors frameAreas() in the editor. `order`: the rows top to bottom when swapped (row_order(seg));
    `frame`: the frame's settings (a free frame's bar and moved area: adjust_free_areas).
    """
    if layout in FREE_TEMPLATES:
        out: list[dict] = []
        k = out_w / 1080
        for a in adjust_free_areas(layout, FREE_TEMPLATES[layout], frame):
            if a["kind"] == "band" and not show_band:
                continue
            x, y = _even(a["x"] * out_w), _even(a["y"] * out_h)
            w = min(out_w - x, max(2, _even((a["x"] + a["w"]) * out_w) - x))
            h = min(out_h - y, max(2, _even((a["y"] + a["h"]) * out_h) - y))
            rad = a.get("radius")
            r = min(w, h) // 2 if rad == "circle" else int(round(float(rad or 0) * k))
            out.append({**a, "x": x, "y": y, "w": w, "h": h, "r": min(r, w // 2, h // 2)})
        return out
    looks = ROW_LOOKS.get(layout, {})
    tpl_rows = FRAME_TEMPLATES[layout]
    # The rows that are shown, top to bottom, with the index each has in the template (for its options)
    seq = order if order and sorted(order) == list(range(len(tpl_rows))) else list(range(len(tpl_rows)))
    shown = [i for i in seq if show_band or tpl_rows[i][0] != "band"]
    out, y = [], 0
    for i, (kind, slot, h) in zip(shown, frame_rows(layout, out_h, show_band, custom, order)):
        area = {**looks.get(i, {}), "kind": kind, "x": 0, "y": y, "w": out_w, "h": h, "r": 0}
        if kind == "slot":
            area["slot"] = slot
        elif kind == "band":
            area["lane"] = "band"
        out.append(area)
        y += h
    return out


def frame_base(layout: str | None) -> str | None:
    """What's under a frame's areas: 'blur' (a blurred copy of the main video), 'band' (the band's colour) or None (black)."""
    return FRAME_BASE.get(layout or "")


def shown_main_slots(state: dict) -> set:
    """The slots the main video is drawn in: its slots, less the ones its lane hid (main_hidden; the
    editor's lane eye). A hidden slot shows what's under it; its sound is still heard unless muted."""
    hidden = state.get("main_hidden") or {}
    return {s for s in (state.get("main_slots") or []) if not hidden.get(str(s))}


# The slot whose picture, blurred, fills a 'blur' frame's background: the main video, or a photo /
# video put in that slot while it shows (editor: BLUR_SOURCE_SLOT)
BLUR_SOURCE_SLOT = 0


def row_heights(seg: dict) -> list[float] | None:
    """A frame's resized row heights (frame.row_h), if they fit its template; else None. Mirrors validRowHeights() in the editor."""
    custom = (seg.get("frame") or {}).get("row_h")
    if (seg.get("layout") or "") in FREE_TEMPLATES:
        return None  # a free frame's areas don't resize
    tpl = FRAME_TEMPLATES.get(seg.get("layout") or "")
    if not tpl or not isinstance(custom, list) or len(custom) != len(tpl):
        return None
    try:
        hs = [float(h) for h in custom]
    except (TypeError, ValueError):
        return None
    return hs if all(h >= 0.03 for h in hs) else None


def row_order(seg: dict) -> list[int] | None:
    """A stacked frame's rows top to bottom when swapped (frame.row_order: template row numbers, each
    once), if it fits the template; else None. Mirrors validRowOrder() in the editor."""
    order = (seg.get("frame") or {}).get("row_order")
    tpl = FRAME_TEMPLATES.get(seg.get("layout") or "")
    if (seg.get("layout") or "") in FREE_TEMPLATES or not tpl or not isinstance(order, list) or len(order) != len(tpl):
        return None
    try:
        out = [int(i) for i in order]
    except (TypeError, ValueError):
        return None
    return out if sorted(out) == list(range(len(tpl))) and out != list(range(len(tpl))) else None


def frame_rows(layout: str, out_h: int, show_band: bool = True, custom: list[float] | None = None,
               order: list[int] | None = None) -> list[tuple[str, int | None, int]]:
    """
    Rows with integer, even pixel heights (yuv420p) that add up to exactly out_h. The text band
    only exists once the frame has text on it (show_band); without it the slots share the height.
    `custom`: the frame's resized row heights (row_heights(seg)), one per template row.
    `order`: the rows top to bottom when swapped with the editor's bar (row_order(seg)); each keeps its height.
    """
    tpl = FRAME_TEMPLATES[layout]
    if custom and len(custom) == len(tpl):
        tpl = [(kind, slot, h) for (kind, slot, _), h in zip(tpl, custom)]
    if order and sorted(order) == list(range(len(tpl))):
        tpl = [tpl[i] for i in order]
    rows = [r for r in tpl if show_band or r[0] != "band"]
    total = sum(share for _, _, share in rows) or 1.0
    heights: list[int] = []
    used = 0
    for i, (_, _, share) in enumerate(rows):
        h = out_h - used if i == len(rows) - 1 else max(2, int(round(share / total * out_h / 2)) * 2)
        heights.append(h)
        used += h
    return [(kind, slot, h) for (kind, slot, _), h in zip(rows, heights)]


def wrap_band_text(text: str, width_px: int, size_px: int) -> list[str]:
    """Greedy word wrap by character count (≈0.56 em per character for a bold sans)."""
    max_chars = max(8, int(width_px * 0.88 / max(1.0, size_px * 0.56)))
    lines: list[str] = []
    for para in (text or "").splitlines() or [""]:
        line = ""
        for word in para.split():
            probe = f"{line} {word}" if line else word
            if line and len(probe) > max_chars:
                lines.append(line)
                line = word
            else:
                line = probe
        lines.append(line)
    while lines and not lines[-1]:
        lines.pop()
    return lines


def photo_motion_filter(motion: str | None, w: int, h: int, dur_s: float) -> str:
    """
    Filters turning a looped still (already scaled/cropped to 2w×2h) into a w×h clip with the chosen
    motion. Matches the editor preview: zoom 1→1.15 (or back), or pan across at 1.15 zoom.
    """
    n = max(1, int(round(dur_s * 30)))
    p = f"min(on/{n}\\,1)"
    if motion == "zoom_in":
        z, x = f"1+0.15*{p}", "iw/2-(iw/zoom/2)"
    elif motion == "zoom_out":
        z, x = f"1.15-0.15*{p}", "iw/2-(iw/zoom/2)"
    elif motion == "pan_left":
        z, x = "1.15", f"(iw-iw/zoom)*(1-{p})"
    elif motion == "pan_right":
        z, x = "1.15", f"(iw-iw/zoom)*{p}"
    else:
        return ""
    return f"zoompan=z='{z}':x='{x}':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps=30"


def frame_state(seg: dict) -> dict:
    """
    A frame's settings with lane items. Frames saved before lanes existed kept one photo or other
    video per slot for the whole format (on its crop box) and one band text; those become
    full-length items. Mirrors frameOf() in the editor.
    """
    f = seg.get("frame") or {}
    if f.get("items") is not None:
        return {**f, "main_slots": f.get("main_slots") if f.get("main_slots") is not None else [0]}
    s0, s1 = int(seg.get("start_ms") or 0), int(seg.get("end_ms") or 0)
    items: list[dict] = []
    main: list[int] = []
    main_volume, main_muted, saw_main = 1.0, False, False
    for b in seg.get("crop_boxes") or []:
        span = {"start_ms": s0, "end_ms": s1, "lane": int(b.get("slot_index") or 0)}
        legacy_id = f"legacy-{b.get('id') or span['lane']}"
        if b.get("image_path"):
            items.append({"id": legacy_id, "kind": "photo", **span,
                          "image_path": b["image_path"], "motion": b.get("image_motion") or "none"})
        elif b.get("source_video_id"):
            items.append({"id": legacy_id, "kind": "video", **span,
                          "source_video_id": b["source_video_id"], "source_offset_ms": int(b.get("source_offset_ms") or 0),
                          "volume": b.get("volume") if b.get("volume") is not None else 1.0, "muted": bool(b.get("muted"))})
        else:
            main.append(span["lane"])
            if not saw_main:
                saw_main = True
                main_volume = b.get("volume") if b.get("volume") is not None else 1.0
                main_muted = bool(b.get("muted"))
    band = f.get("band") or {}
    if (band.get("text") or "").strip() and any(a.get("lane") == "band" for a in frame_areas(seg.get("layout") or "", 1080, 1920)):
        items.append({"id": "legacy-band", "kind": "text", "lane": "band", "start_ms": s0, "end_ms": s1,
                      "text": band["text"], "bg": band.get("bg"), "color": band.get("color"), "size": band.get("size")})
    return {"band": {**band, "text": ""}, "main_slots": main, "main_volume": main_volume,
            "main_muted": main_muted, "items": items}


def lane_items(seg: dict, state: dict, lane) -> list[dict]:
    """
    One lane's items in time order, cut to the format's range. Each gets rel_s (seconds from the
    format's start), dur_s, and for videos the source position at its visible start (off_ms).
    """
    s0, s1 = int(seg["start_ms"]), int(seg["end_ms"])
    out: list[dict] = []
    for it in state.get("items") or []:
        if it.get("lane") != lane:
            continue
        a, b = max(int(it.get("start_ms") or 0), s0), min(int(it.get("end_ms") or 0), s1)
        if b - a < 50:
            continue
        off = int(it.get("source_offset_ms") or 0) + (a - int(it.get("start_ms") or 0))
        out.append({**it, "rel_s": (a - s0) / 1000.0, "dur_s": (b - a) / 1000.0, "off_ms": max(0, off)})
    out.sort(key=lambda x: x["rel_s"])
    return out


def frame_band_shown(seg: dict) -> bool:
    """Frames whose template has a band always show it: a solid space for text (editor: frameBandShown)."""
    layout = seg.get("layout")
    return is_frame(layout) and any(a["kind"] == "band" for a in frame_areas(layout, 1080, 1920))


def caption_band_zones(segments: list[dict], out_h: int) -> list[tuple[int, int, int]]:
    """
    Where captions move into a frame's text band: (start ms, end ms, centre y px), in clip time.
    """
    zones: list[tuple[int, int, int]] = []
    for seg in segments:
        layout = seg.get("layout")
        if not is_frame(layout):
            continue
        centre = None
        strip = None
        for a in frame_areas(layout, 1080, out_h, True, row_heights(seg), row_order(seg), seg.get("frame")):
            if a["kind"] == "band" and a.get("lane") == "band":
                centre = a["y"] + a["h"] // 2
            elif a["kind"] == "caption":
                strip = a["y"] + a["h"] // 2
        s0 = int(seg["start_ms"])
        if centre is not None:
            for it in lane_items(seg, frame_state(seg), "band"):
                if it.get("kind") == "text" and it.get("captions") and not it.get("hidden"):
                    a = s0 + int(round(it["rel_s"] * 1000))
                    zones.append((a, a + int(round(it["dur_s"] * 1000)), centre))
        # A frame with a caption strip: the captions sit there for the whole frame
        if strip is not None:
            zones.append((s0, int(seg["end_ms"]), strip))
    return zones


def main_slot_sound(state: dict, slot: int) -> tuple[float, bool]:
    """
    (volume, muted) of the main video in one slot. Each slot has its own setting; one without it:
    the first main slot uses the frame's older main setting, any other is muted. Mirrors
    mainSlotSound() in the editor.
    """
    slots = state.get("main_slots") or [0]
    first = min(slots) if slots else 0
    key = str(slot)
    vols, mutes = state.get("main_volumes") or {}, state.get("main_mutes") or {}
    legacy_vol = state.get("main_volume") if state.get("main_volume") is not None else 1.0
    volume = float(vols[key]) if vols.get(key) is not None else (float(legacy_vol) if slot == first else 1.0)
    muted = bool(mutes[key]) if mutes.get(key) is not None else (bool(state.get("main_muted")) if slot == first else True)
    return volume, muted


def main_audio_volume(state: dict) -> float:
    """How loud the main video plays in a frame: the sum of every audible slot showing it."""
    # Heard under a frame no slot shows it in (Photo Story)
    if state.get("main_under") and not (state.get("main_slots") or []):
        if state.get("main_muted"):
            return 0.0
        return float(state.get("main_volume") if state.get("main_volume") is not None else 1.0)
    total = 0.0
    for slot in state.get("main_slots") or []:
        v, m = main_slot_sound(state, int(slot))
        total += 0.0 if m else v
    return total


def frame_audio_plan(seg: dict, secondary_videos: dict[str, str]) -> list[tuple[str | None, float, float, float, float]]:
    """
    Sound of one frame format as (source video id or None for the main video, source start in s,
    volume, delay from the format's start in s, duration in s). The main video plays throughout
    at the frame's main volume; each video item adds its own sound while it's on screen.
    """
    state = frame_state(seg)
    dur_s = (int(seg["end_ms"]) - int(seg["start_ms"])) / 1000.0
    out: list[tuple[str | None, float, float, float, float]] = []
    # Every slot showing the main video adds its own sound (the same audio, so volumes add up)
    vol = main_audio_volume(state)
    if vol > 0:
        if seg.get("video_offset_ms") is not None:
            off_ms = int(seg["video_offset_ms"])
        else:
            boxes = sorted(seg.get("crop_boxes") or [], key=lambda b: b.get("slot_index", 0))
            main_box = next((b for b in boxes if not b.get("source_video_id") and not b.get("image_path")), None)
            off_ms = int(main_box["source_offset_ms"]) if main_box and main_box.get("source_offset_ms") is not None else int(seg["start_ms"])
        out.append((None, off_ms / 1000.0, vol, 0.0, dur_s))
    lanes = sorted({it.get("lane") for it in state.get("items") or [] if isinstance(it.get("lane"), int)})
    for lane in lanes:
        for it in lane_items(seg, state, lane):
            if it.get("kind") != "video" or it.get("muted"):
                continue
            v = float(it.get("volume") if it.get("volume") is not None else 1.0)
            vid = it.get("source_video_id")
            if v <= 0 or vid not in secondary_videos:
                continue
            out.append((vid, it["off_ms"] / 1000.0, v, it["rel_s"], it["dur_s"]))
    return out
