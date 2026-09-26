"""
Frame layouts: the 9:16 reel divided top-to-bottom into media slots and an optional letterbox
band for text. Mirrors chai-cut-frontend/src/modules/editor/frames.ts — keep the numbers in step.
"""
from __future__ import annotations

# (kind, slot index or None, share of the frame height)
FRAME_TEMPLATES: dict[str, list[tuple[str, int | None, float]]] = {
    "frame_single":         [("band", None, 0.20), ("slot", 0, 0.80)],
    "frame_video_photo":    [("slot", 0, 0.42), ("band", None, 0.16), ("slot", 1, 0.42)],
    "frame_dual":           [("slot", 0, 0.50), ("slot", 1, 0.50)],
    "frame_dual_letterbox": [("slot", 0, 0.42), ("band", None, 0.16), ("slot", 1, 0.42)],
    "frame_triple":         [("slot", 0, 1 / 3), ("slot", 1, 1 / 3), ("slot", 2, 1 / 3)],
}

MOTIONS = {"none", "zoom_in", "zoom_out", "pan_left", "pan_right"}


def is_frame(layout: str | None) -> bool:
    return bool(layout) and layout in FRAME_TEMPLATES


def frame_rows(layout: str, out_h: int, show_band: bool = True) -> list[tuple[str, int | None, int]]:
    """
    Rows with integer, even pixel heights (yuv420p) that add up to exactly out_h. The text band
    only exists once the frame has text on it (show_band); without it the slots share the height.
    """
    rows = [r for r in FRAME_TEMPLATES[layout] if show_band or r[0] != "band"]
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
    if (band.get("text") or "").strip():
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
    """The text band shows only while the frame has text on it (editor: frameBandShown)."""
    layout = seg.get("layout")
    if not is_frame(layout) or not any(kind == "band" for kind, _, _ in FRAME_TEMPLATES[layout]):
        return False
    return bool(lane_items(seg, frame_state(seg), "band"))


def caption_band_zones(segments: list[dict], out_h: int) -> list[tuple[int, int, int]]:
    """
    Where captions move into a frame's text band: (start ms, end ms, centre y px), in clip time.
    """
    zones: list[tuple[int, int, int]] = []
    for seg in segments:
        layout = seg.get("layout")
        if not is_frame(layout):
            continue
        y = 0
        centre = None
        for kind, _, h in frame_rows(layout, out_h):
            if kind == "band":
                centre = y + h // 2
            y += h
        if centre is None:
            continue
        s0 = int(seg["start_ms"])
        for it in lane_items(seg, frame_state(seg), "band"):
            if it.get("kind") == "text" and it.get("captions"):
                a = s0 + int(round(it["rel_s"] * 1000))
                zones.append((a, a + int(round(it["dur_s"] * 1000)), centre))
    return zones


def frame_audio_plan(seg: dict, secondary_videos: dict[str, str]) -> list[tuple[str | None, float, float, float, float]]:
    """
    Sound of one frame format as (source video id or None for the main video, source start in s,
    volume, delay from the format's start in s, duration in s). The main video plays throughout
    at the frame's main volume; each video item adds its own sound while it's on screen.
    """
    state = frame_state(seg)
    dur_s = (int(seg["end_ms"]) - int(seg["start_ms"])) / 1000.0
    out: list[tuple[str | None, float, float, float, float]] = []
    vol = float(state.get("main_volume") if state.get("main_volume") is not None else 1.0)
    if not state.get("main_muted") and vol > 0:
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
