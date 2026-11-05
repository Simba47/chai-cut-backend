"""
After the export: is the main video there wherever it should be? On QA (2026-11-05) an export came
out with the speaker black from one part on to the end (the sound fine, FFmpeg reporting success)
while the same clip exported here was right. So each part that shows the main video is looked at
once in the finished file; black where it should be (and isn't black in the video itself) means
the export is run again, and if it's still wrong the export fails rather than hand over a broken video.
"""
import subprocess

from frames import (is_frame, frame_state, frame_areas, frame_band_shown, row_heights, row_order,
                    shown_main_slots, lane_items)

# The picture is looked at this small (grey): enough to tell black from a picture, quick to read
GRAB_W, GRAB_H = 216, 384


def _main_box(rect, aw: int, ah: int):
    """The main video's own box in a slot it was resized in (x, y, w, h as parts of the slot), else None"""
    if not isinstance(rect, dict):
        return None
    try:
        w, h = max(0.15, min(1.0, float(rect["w"]))), max(0.15, min(1.0, float(rect["h"])))
        x, y = max(0.0, min(1.0 - w, float(rect["x"]))), max(0.0, min(1.0 - h, float(rect["y"])))
    except (KeyError, TypeError, ValueError):
        return None
    return (x * aw, y * ah, w * aw, h * ah)


def _inner(x: float, y: float, w: float, h: float, keep: float = 0.5) -> tuple[float, float, float, float]:
    """The middle `keep` of a box (clear of its margins and rounded corners)"""
    return (x + w * (1 - keep) / 2, y + h * (1 - keep) / 2, w * keep, h * keep)


def main_picture_probes(segments: list[dict], out_w: int, out_h: int, blank_ranges: list[tuple[int, int]],
                        is_main: "callable") -> list[dict]:
    """
    Where and when the finished video must show the main video: one look in the middle of each part
    that shows it (and one near the end of the last), as {t: s in the export, src_t: s in the main
    video input, region: (x, y, w, h) px, seg: index}. `is_main(seg)`: the part shows the main video
    (not a B-roll or a video put in between). Parts at another speed, hidden stretches and parts
    under 1.5 s aren't looked at.
    """
    out: list[dict] = []
    for si, seg in enumerate(segments):
        s0, s1 = int(seg["start_ms"]), int(seg["end_ms"])
        dur = s1 - s0
        if dur < 1500 or seg.get("hidden") or seg.get("inserted") or not is_main(seg):
            continue
        layout = seg.get("layout") or "vertical"
        if not is_frame(layout) and float(seg.get("speed") or 1) != 1:
            continue
        rels = [dur // 2] + ([max(dur // 2, dur - 1000)] if si == len(segments) - 1 and dur >= 3000 else [])
        boxes = sorted(seg.get("crop_boxes") or [], key=lambda b: b.get("slot_index", 0))

        def src_ms(box) -> int:
            if seg.get("video_offset_ms") is not None:
                return int(seg["video_offset_ms"])
            if box and box.get("source_offset_ms") is not None:
                return int(box["source_offset_ms"])
            return s0

        for rel in rels:
            if any(a <= s0 + rel < b for a, b in blank_ranges):
                continue
            if not is_frame(layout):
                main_box = next((b for b in boxes if not b.get("source_video_id")), None)
                out.append({"t": (s0 + rel) / 1000.0, "src_t": (src_ms(main_box) + rel) / 1000.0,
                            "region": _inner(0, 0, out_w, out_h, 0.6), "seg": si})
                continue
            st = frame_state(seg)
            main = shown_main_slots(st)
            for a in frame_areas(layout, out_w, out_h, frame_band_shown(seg), row_heights(seg), row_order(seg), seg.get("frame")):
                if a.get("kind") != "slot" or a.get("slot") not in main:
                    continue
                slot = a["slot"]
                # (a photo or video put in this slot covers the main video while it shows)
                if any(not it.get("hidden") and it.get("kind") in ("photo", "video")
                       and it["rel_s"] * 1000 <= rel < (it["rel_s"] + it["dur_s"]) * 1000
                       for it in lane_items(seg, st, slot)):
                    continue
                box = _main_box((st.get("main_rects") or {}).get(str(slot)), a["w"], a["h"])
                bx, by, bw, bh = box if box else (0, 0, a["w"], a["h"])
                box_of_slot = next((b for b in boxes if b.get("slot_index") == slot), None)
                out.append({"t": (s0 + rel) / 1000.0, "src_t": (src_ms(box_of_slot) + rel) / 1000.0,
                            "region": _inner(a["x"] + bx, a["y"] + by, bw, bh), "seg": si, "slot": slot})
    return out


def grab_grey(path: str, t: float, input_opts: list[str] | None = None) -> bytes | None:
    """One picture of a video at t s, GRAB_W×GRAB_H grey (None: nothing there)"""
    cmd = ["ffmpeg", "-v", "error", *(input_opts or []), "-ss", f"{max(0.0, t):.3f}", "-i", path,
           "-frames:v", "1", "-vf", f"scale={GRAB_W}:{GRAB_H},format=gray", "-f", "rawvideo", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None
    return r.stdout if r.returncode == 0 and len(r.stdout) >= GRAB_W * GRAB_H else None


def flat_dark(grey: bytes, region: tuple[float, float, float, float] | None, frame_w: int, frame_h: int) -> bool:
    """
    Whether a stretch of a grey picture is one flat dark colour (black, or the dark grey of an empty
    slot): no picture there. A real picture, even a dark one, varies more than this.
    """
    x, y, w, h = region or (0, 0, frame_w, frame_h)
    kx, ky = GRAB_W / frame_w, GRAB_H / frame_h
    x0, y0 = max(0, int(x * kx)), max(0, int(y * ky))
    x1, y1 = min(GRAB_W, max(x0 + 1, int((x + w) * kx))), min(GRAB_H, max(y0 + 1, int((y + h) * ky)))
    px = [grey[r * GRAB_W + c] for r in range(y0, y1) for c in range(x0, x1)]
    if not px:
        return False
    mean = sum(px) / len(px)
    sd = (sum((p - mean) ** 2 for p in px) / len(px)) ** 0.5
    return mean < 40 and sd < 3.0


def missing_main_picture(rendered: str, probes: list[dict], out_w: int, out_h: int, video_path: str,
                         clip_start_s: float, video_opts: list[str] | None = None) -> list[dict]:
    """
    The probes where the finished video is flat dark although the main video, at that moment, isn't
    (a stretch that is black in the video itself is fine).
    """
    bad: list[dict] = []
    grabbed: dict[float, bytes | None] = {}
    for p in probes:
        if p["t"] not in grabbed:
            grabbed[p["t"]] = grab_grey(rendered, p["t"])
        g = grabbed[p["t"]]
        if g is None or not flat_dark(g, p["region"], out_w, out_h):
            continue
        src = grab_grey(video_path, clip_start_s + p["src_t"], video_opts)
        # (the video itself black there — a fade, a black screen — or unreadable: not a fault)
        if src is None or flat_dark(src, None, GRAB_W, GRAB_H):
            continue
        bad.append(p)
    return bad


def at(t: float) -> str:
    """m:ss.s"""
    return f"{int(t // 60)}:{t % 60:04.1f}"
