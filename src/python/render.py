#!/usr/bin/env python3
"""
FFmpeg-native video compositor.

Replaces the frame-by-frame Python/OpenCV render loop with a single
FFmpeg filter_complex invocation. Clip render time drops from 10-12 min
to 1-2 min on the same CPU hardware. Output quality is identical.

CLI contract is unchanged:
  python3 render.py --video src.mp4 --spec spec.json --output out.mp4
  [--secondary-videos '{"video_id":"local_path"}']
  [--overlay-images   '{"storage_path":"local_path"}']
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audio import build_ffmpeg_audio_args, build_segment_audio_args, extract_speech_ranges
from frames import is_frame, frame_rows, wrap_band_text, photo_motion_filter, frame_state, frame_band_shown, lane_items, caption_band_zones

# ── Quality presets (identical to previous version) ───────────────────────────
_QUALITY: dict[int, dict] = {
    480:  dict(crf=20, maxrate="3M",  bufsize="6M",  preset="veryfast", audio_br="128k"),
    720:  dict(crf=18, maxrate="6M",  bufsize="12M", preset="veryfast", audio_br="192k"),
    1080: dict(crf=18, maxrate="10M", bufsize="20M", preset="veryfast", audio_br="192k"),
    2160: dict(crf=16, maxrate="20M", bufsize="40M", preset="veryfast", audio_br="256k"),
}

# ── Font lookup ────────────────────────────────────────────────────────────────
_FONTS_DIR = str(Path(__file__).parent / "fonts")
_FONT_FILES = {
    "noto-sans-telugu":     "NotoSansTelugu-Regular.ttf",
    "noto-sans-devanagari": "NotoSansDevanagari-Regular.ttf",
    "roboto":               "Roboto-Regular.ttf",
    "montserrat-bold":      "Montserrat-Bold.ttf",
}
_FONT_NAMES = {
    "noto-sans-telugu":     "Noto Sans Telugu",
    "noto-sans-devanagari": "Noto Sans Devanagari",
    "roboto":               "Roboto",
    "montserrat-bold":      "Montserrat Bold",
}


def _find_font_path(font_id: str) -> str:
    filename = _FONT_FILES.get(font_id, _FONT_FILES["roboto"])
    for d in [os.environ.get("FONTS_DIR", ""), _FONTS_DIR, "/usr/share/fonts", "/System/Library/Fonts"]:
        if not d:
            continue
        for root, _, files in os.walk(d):
            if filename in files:
                return os.path.join(root, filename)
    return ""


# ── ASS subtitle generation ────────────────────────────────────────────────────

def _hex_to_ass(hex_color: str, alpha: int = 0) -> str:
    """#RRGGBB → ASS &HAABBGGRR  (alpha 0 = fully opaque)."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def _ms_to_ass_ts(ms: int) -> str:
    cs = (ms // 10) % 100
    s  = (ms // 1000) % 60
    m  = (ms // 60_000) % 60
    h  =  ms // 3_600_000
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _word_display(w: dict, roman: bool = False) -> str:
    """A caption word in the letters the user picked: English letters (word_roman) or its own script."""
    return (w.get("word_roman") if roman else None) or w.get("word", "")


def _is_sentence_end(word: str) -> bool:
    stripped = word.rstrip()
    return bool(stripped) and stripped[-1] in ".!?।"


_PHRASE_GAP_MS   = 300  # silence gap longer than this → new subtitle line
_MAX_PHRASE_WORDS = 5   # also break at this many words even with no gap


def _group_sentences(words: list[dict]) -> list[list[dict]]:
    """Split words into subtitle phrases using gap + word-count, not punctuation.

    Punctuation-only splitting silently merges entire clips into one event when
    the transcript has no .!? marks (common for Telugu lyrics / dubbed dialogue).
    Gap-based splitting mirrors the editor preview (CAPTION_GAP_MS / MAX_WORDS).
    """
    sentences: list[list[dict]] = []
    current: list[dict] = []
    for i, w in enumerate(words):
        current.append(w)
        is_last = i == len(words) - 1
        if is_last:
            sentences.append(current)
            break
        gap = words[i + 1]["start_ms"] - w["end_ms"]
        # A different speaker always starts a new line (never mix two people's words)
        speaker_change = (
            w.get("speaker_id") is not None
            and words[i + 1].get("speaker_id") is not None
            and words[i + 1]["speaker_id"] != w["speaker_id"]
        )
        if (
            _is_sentence_end(w.get("word", ""))
            or speaker_change
            or gap > _PHRASE_GAP_MS
            or len(current) >= _MAX_PHRASE_WORDS
        ):
            sentences.append(current)
            current = []
    return sentences


def _write_ass(
    words: list[dict],
    style: dict,
    clip_start_ms: int,
    out_w: int,
    out_h: int,
    path: str,
    band_zones: list[tuple[int, int, int]] | None = None,
) -> None:
    font_id   = style.get("font") or "noto-sans-telugu"
    font_name = _FONT_NAMES.get(font_id, "Roboto")
    font_size = int(style.get("size") or 52)
    color_hex = style.get("color") or "#ffffff"
    pos_y_frac = float(style.get("position_y") or 0.84)
    # The editor's Caption language: 'roman' = English letters, anything else = the spoken script
    roman     = style.get("language") == "roman"

    primary = _hex_to_ass(color_hex, 0)
    shadow  = "&H80000000"

    # Editor preview shows a word at video time = word time + timing_offset_ms
    offset_ms = int(style.get("timing_offset_ms") or 0)

    # Rebase word timestamps to clip-relative (0 = first frame of clip)
    clip_words = [
        {**w, "start_ms": w["start_ms"] - clip_start_ms + offset_ms,
               "end_ms":   w["end_ms"]   - clip_start_ms + offset_ms}
        for w in words
    ]

    pos_x = out_w // 2
    pos_y = int(pos_y_frac * out_h)

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {out_w}",
        f"PlayResY: {out_h}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour,"
        " BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle,"
        " BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font_name},{font_size},{primary},&H00FFFFFF,&H00000000,{shadow},"
        "0,0,0,0,100,100,0,0,1,3,2,5,10,10,10,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    sentences = [s for s in _group_sentences(clip_words) if s]
    # max(): transcription occasionally returns a word whose end is before its start
    spans = [[s[0]["start_ms"], max(s[0]["start_ms"], s[-1]["end_ms"])] for s in sentences]

    # Transcription sometimes stamps several lines with the same start time
    # (zero-length words). Spread each such run evenly up to the next line.
    i = 0
    while i < len(spans):
        j = i
        while j + 1 < len(spans) and spans[j + 1][0] <= spans[i][0]:
            j += 1
        if j > i:
            run_start = spans[i][0]
            run_end = spans[j + 1][0] if j + 1 < len(spans) else run_start + 1500 * (j - i + 1)
            run_end = max(run_end, max(e for _, e in spans[i:j + 1]))
            slot = (run_end - run_start) / (j - i + 1)
            for k in range(i, j + 1):
                spans[k] = [round(run_start + (k - i) * slot), round(run_start + (k - i + 1) * slot)]
        i = j + 1

    for i, sentence in enumerate(sentences):
        start_ms = max(0, spans[i][0])
        end_ms   = spans[i][1] + 200
        # Only one line on screen at a time (all lines share the same \pos, so any
        # overlap draws them on top of each other). Like the editor preview, bridge
        # sub-500ms gaps to the next line and never run past its start.
        if i + 1 < len(spans):
            next_start = spans[i + 1][0]
            if next_start - spans[i][1] < 500:
                end_ms = next_start
            end_ms = min(end_ms, next_start)
        if end_ms <= start_ms:
            continue
        text     = " ".join(_word_display(w, roman) for w in sentence if _word_display(w, roman))
        if not text.strip():
            continue
        # Alignment=5 (center of screen); \pos pins the anchor to exact coordinates. Where a
        # frame shows captions in its text band, that part of the line is centred in the band.
        for a, b, y in _split_by_zones(start_ms, end_ms, band_zones or [], pos_y):
            tag = f"{{\\pos({pos_x},{y})}}"
            lines.append(
                f"Dialogue: 0,{_ms_to_ass_ts(a)},{_ms_to_ass_ts(b)},"
                f"Default,,0,0,0,,{tag}{text}"
            )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _split_by_zones(start_ms: int, end_ms: int, zones: list[tuple[int, int, int]], default_y: int) -> list[tuple[int, int, int]]:
    """Cut [start, end) where caption band zones begin/end: (start, end, y) pieces."""
    cuts = {start_ms, end_ms}
    for a, b, _ in zones:
        if start_ms < a < end_ms:
            cuts.add(a)
        if start_ms < b < end_ms:
            cuts.add(b)
    pts = sorted(cuts)
    out: list[tuple[int, int, int]] = []
    for a, b in zip(pts, pts[1:]):
        mid = (a + b) / 2
        y = next((zy for za, zb, zy in zones if za <= mid < zb), default_y)
        if out and out[-1][2] == y and out[-1][1] == a:
            out[-1] = (out[-1][0], b, y)
        else:
            out.append((a, b, y))
    return out


# ── Frames ────────────────────────────────────────────────────────────────────

_probe_cache: dict[str, tuple[float, bool]] = {}


def _probe(path: str) -> tuple[float, bool]:
    """(duration in s or 0 if unknown, has an audio stream) of a media file."""
    if path in _probe_cache:
        return _probe_cache[path]
    dur, has_audio = 0.0, True
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", path],
            capture_output=True, text=True, timeout=30,
        )
        info = json.loads(r.stdout or "{}")
        dur = float((info.get("format") or {}).get("duration") or 0)
        has_audio = any(st.get("codec_type") == "audio" for st in info.get("streams") or [])
    except Exception:
        pass
    _probe_cache[path] = (dur, has_audio)
    return _probe_cache[path]


def _hex6(v: str | None, fallback: str) -> str:
    h = (v or "").lstrip("#")
    return h[:6] if len(h) >= 6 and all(c in "0123456789abcdefABCDEF" for c in h[:6]) else fallback


def _frame_text_font(text: str) -> str:
    """Montserrat Bold for Latin text; Noto for Indian scripts Montserrat can't draw."""
    if any("\u0c00" <= c <= "\u0c7f" for c in text):
        path = _find_font_path("noto-sans-telugu")
    elif any("\u0900" <= c <= "\u097f" for c in text):
        path = _find_font_path("noto-sans-devanagari")
    else:
        path = _find_font_path("montserrat-bold")
    path = path or _find_font_path("roboto")
    # Quoted in the filter; inside the quotes ':' (Windows drive letters) still needs escaping
    return path.replace("\\", "/").replace(":", "\\:") if path else ""


def _frame_text_chain(text: str, bg: str, color: str, size_1080: int, w: int, h: int, dur_s: float) -> str:
    """A solid w×h card with centred, wrapped text (the band or a text card in a slot)."""
    chain = f"color=c=0x{bg}:s={w}x{h}:d={dur_s:.3f}:r=30,format=yuv420p"
    text = (text or "").strip()
    font = _frame_text_font(text) if text else ""
    if text and font:
        size = max(8, int(round(size_1080 * w / 1080)))
        # Wrap at 1080-wide scale, like the editor preview, so lines break in the same places
        lines = wrap_band_text(text, 1080, size_1080)
        lh = int(size * 1.2)
        top = (h - lh * len(lines)) / 2
        for li, ln in enumerate(lines):
            if not ln:
                continue
            y = int(top + li * lh + (lh - size) / 2)
            chain += (f",drawtext=fontfile='{font}':text='{_escape_drawtext(ln)}':fontsize={size}"
                      f":fontcolor=0x{color}:x=(w-tw)/2:y={y}")
    return f"{chain},setsar=1"


# ── FFmpeg crop expression builder ────────────────────────────────────────────

_MAX_KF_PER_ATTR = 20  # FFmpeg expression depth limit — lower = faster eval


def _rdp_simplify(kfs: list[dict], tol: float = 0.005) -> list[dict]:
    """
    Ramer-Douglas-Peucker simplification over all crop attributes simultaneously.
    Removes keyframes whose x/y/w/h values are within `tol` of linear interpolation
    between their neighbours. tol=0.005 = 0.5% of normalized [0,1] range (< 6px at 1080p).
    """
    if len(kfs) <= 2:
        return kfs[:]

    first, last = kfs[0], kfs[-1]
    t0, t1 = first["t_ms"], last["t_ms"]
    if t1 == t0:
        return [first, last]

    max_dev, max_i = 0.0, 0
    for i in range(1, len(kfs) - 1):
        alpha = (kfs[i]["t_ms"] - t0) / (t1 - t0)
        dev = max(
            abs(kfs[i][a] - (first[a] + alpha * (last[a] - first[a])))
            for a in ("x", "y", "w", "h")
        )
        if dev > max_dev:
            max_dev, max_i = dev, i

    if max_dev <= tol:
        return [first, last]

    left  = _rdp_simplify(kfs[:max_i + 1], tol)
    right = _rdp_simplify(kfs[max_i:], tol)
    return left[:-1] + right

def _step_expr(kf_list: list[dict], attr: str, seg_start_ms: int) -> str:
    """Step-interpolated crop coordinate (holds value until next keyframe fires)."""
    dim = "iw" if attr in ("x", "w") else "ih"
    defaults = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}

    if not kf_list:
        return f"{dim}*{defaults[attr]:.6f}"

    sk = _rdp_simplify(sorted(kf_list, key=lambda k: k["t_ms"]))

    deduped: list[dict] = [sk[0]]
    for kf in sk[1:]:
        if abs(kf[attr] - deduped[-1][attr]) > 1e-6:
            deduped.append(kf)
    sk = deduped

    if len(sk) > _MAX_KF_PER_ATTR:
        step = (len(sk) - 1) / (_MAX_KF_PER_ATTR - 1)
        sk = [sk[round(i * step)] for i in range(_MAX_KF_PER_ATTR)]

    if len(sk) == 1:
        return f"{dim}*{sk[0][attr]:.6f}"

    result = f"{dim}*{sk[-1][attr]:.6f}"
    for i in range(len(sk) - 2, -1, -1):
        t_switch = (sk[i + 1]["t_ms"] - seg_start_ms) / 1000.0
        result = f"if(lt(t,{t_switch:.3f}),{dim}*{sk[i][attr]:.6f},{result})"

    return result


def _linear_expr(kf_list: list[dict], attr: str, seg_start_ms: int) -> str:
    """
    Linear-interpolated crop coordinate — smoothly pans between keyframes.
    Used for motion-tracked crops so the crop follows the subject continuously
    instead of jumping every N seconds.
    """
    dim = "iw" if attr in ("x", "w") else "ih"
    defaults = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}

    if not kf_list:
        return f"{dim}*{defaults[attr]:.6f}"

    sk = _rdp_simplify(sorted(kf_list, key=lambda k: k["t_ms"]))

    # Deduplicate unchanged values to shrink the expression
    deduped: list[dict] = [sk[0]]
    for kf in sk[1:]:
        if abs(kf[attr] - deduped[-1][attr]) > 1e-4:
            deduped.append(kf)
    sk = deduped

    if len(sk) > _MAX_KF_PER_ATTR:
        step = (len(sk) - 1) / (_MAX_KF_PER_ATTR - 1)
        sk = [sk[round(i * step)] for i in range(_MAX_KF_PER_ATTR)]

    if len(sk) == 1:
        return f"{dim}*{sk[0][attr]:.6f}"

    # Build right-to-left: each segment linearly interpolates from kf[i] to kf[i+1]
    result = f"{dim}*{sk[-1][attr]:.6f}"
    for i in range(len(sk) - 2, -1, -1):
        t0 = (sk[i]["t_ms"] - seg_start_ms) / 1000.0
        t1 = (sk[i + 1]["t_ms"] - seg_start_ms) / 1000.0
        v0 = sk[i][attr]
        v1 = sk[i + 1][attr]
        dt = max(t1 - t0, 0.001)
        # lerp: v0 + (v1-v0) * (t-t0) / dt
        lerp = f"{dim}*({v0:.6f}+({v1:.6f}-{v0:.6f})*(t-{t0:.3f})/{dt:.3f})"
        result = f"if(lt(t,{t1:.3f}),{lerp},{result})"

    return result


def _crop_filter(box: dict | None, seg_start_ms: int) -> str:
    kf = box.get("box_keyframes", []) if box else []
    # Use linear interpolation when there are multiple keyframes (motion tracking)
    # so the crop smoothly follows the subject instead of jumping at each keyframe.
    expr = _linear_expr if len(kf) > 1 else _step_expr
    w = _esc_expr(f"max(2,{expr(kf, 'w', seg_start_ms)})")
    h = _esc_expr(f"max(2,{expr(kf, 'h', seg_start_ms)})")
    x = _esc_expr(f"min(iw-2,{expr(kf, 'x', seg_start_ms)})")
    y = _esc_expr(f"min(ih-2,{expr(kf, 'y', seg_start_ms)})")
    return f"crop=w={w}:h={h}:x={x}:y={y}"


def _scale_cover(w: int, h: int) -> str:
    """Scale to fill w×h (cover crop — no black bars, excess is cropped center)."""
    return (
        f"scale=w={w}:h={h}:flags=lanczos:force_original_aspect_ratio=increase,"
        f"crop={w}:{h}"
    )


def _scale_fit(w: int, h: int) -> str:
    """Scale to fit inside w×h (letterbox — preserves AR, pads with black)."""
    return (
        f"scale=w={w}:h={h}:flags=lanczos:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black"
    )


def _escape_drawtext(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:").replace("%", "\\%")


def _esc_expr(s: str) -> str:
    """Escape an expression for FFmpeg filter_complex option values.

    The filter_complex parser splits on ',' to find filter chain boundaries.
    It does NOT track parenthesis depth, so commas inside if()/max()/min()/lt()
    must be backslash-escaped so they aren't mistaken for filter separators.
    """
    return s.replace("\\", "\\\\").replace(",", "\\,")


# ── Main compositor ────────────────────────────────────────────────────────────

def main(
    video_path: str,
    spec_path: str,
    output_path: str,
    secondary_videos: dict[str, str] | None = None,
    overlay_images: dict[str, str] | None = None,
    overlay_videos: dict[str, str] | None = None,
    watermark: bool = False,
    frame_images: dict[str, str] | None = None,
) -> None:
    secondary_videos = secondary_videos or {}
    overlay_images   = overlay_images   or {}
    overlay_videos   = overlay_videos   or {}
    frame_images     = frame_images     or {}

    spec           = json.load(open(spec_path))
    clip_start_ms  = int(spec["start_ms"])
    clip_end_ms    = int(spec["end_ms"])
    clip_dur_ms    = clip_end_ms - clip_start_ms
    out_w          = int(spec.get("output_width",  1080))
    out_h          = int(spec.get("output_height", 1920))
    # Sort by (start_ms, sort_order) — sort_order breaks ties so the "primary" segment
    # (lower sort_order) always wins when two segments share the same start_ms.
    segments = sorted(spec["segments"], key=lambda s: (s["start_ms"], s.get("sort_order", 0)))
    # Mirror the preview's "first match wins" rule: where a segment starts before the
    # previous one ended, only its overlapped part is hidden. Trim its start to the
    # previous end (shifting its source position by the same amount) instead of dropping
    # the whole segment — otherwise e.g. a main-video segment that starts under a B-roll
    # insert loses everything after the insert, even though the preview plays it.
    _filtered: list[dict] = []
    _next_start = -1
    for _s in segments:
        if _s["start_ms"] < _next_start:
            if _s["end_ms"] <= _next_start:
                print(f"[render] SKIP hidden seg: start={_s['start_ms']}ms end={_s['end_ms']}ms sort_order={_s.get('sort_order')}", flush=True)
                continue
            delta = _next_start - _s["start_ms"]
            print(f"[render] TRIM overlapping seg: start={_s['start_ms']}ms → {_next_start}ms (end={_s['end_ms']}ms)", flush=True)
            _s = {**_s, "start_ms": _next_start}
            if _s.get("video_offset_ms") is not None:
                _s["video_offset_ms"] = int(_s["video_offset_ms"]) + delta
            _s["crop_boxes"] = [
                {**b, "source_offset_ms": int(b["source_offset_ms"]) + delta} if b.get("source_offset_ms") is not None else b
                for b in (_s.get("crop_boxes") or [])
            ]
        _filtered.append(_s)
        _next_start = _s["end_ms"]
    segments = _filtered
    # Formats are independent in the editor, so parts of the clip can have no format. Those parts
    # use the default framing: a full-frame box, which the cover scale crops to a centred 9:16
    # (the same default the editor preview shows). Filling them keeps video, audio and captions
    # the same length and in sync.
    _filled: list[dict] = []
    _cursor = 0
    for _s in segments + [None]:
        _gap_end = clip_dur_ms if _s is None else int(_s["start_ms"])
        if _gap_end - _cursor >= 50:
            print(f"[render] default framing for uncovered {_cursor}ms–{_gap_end}ms", flush=True)
            _filled.append({
                "start_ms": _cursor, "end_ms": _gap_end, "layout": "vertical", "sort_order": 0,
                "crop_boxes": [{
                    "slot_index": 0, "source_video_id": None, "source_offset_ms": _cursor,
                    "box_keyframes": [{"t_ms": _cursor, "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}],
                }],
            })
        if _s is not None:
            _filled.append(_s)
            _cursor = max(_cursor, int(_s["end_ms"]))
    segments = _filled
    for _s in segments:
        broll = bool((_s.get("crop_boxes") or [{}])[0].get("source_video_id"))
        print(f"[render] seg: start={_s['start_ms']}ms end={_s['end_ms']}ms video_offset={_s.get('video_offset_ms')} broll={broll}", flush=True)
    words          = spec.get("words", [])
    caption_styles = spec.get("caption_styles", [])
    text_overlays  = spec.get("text_overlays", [])
    audio_tracks   = spec.get("audio_tracks", [])
    filters_cfg    = spec.get("filters", {})
    img_overlays   = [o for o in spec.get("overlays", []) if o.get("type") == "image"]
    vid_overlays   = [o for o in spec.get("overlays", []) if o.get("type") == "video"]

    for ov in img_overlays:
        ov["local_path"] = overlay_images.get(ov.get("storage_path", ""), "")

    for ov in vid_overlays:
        ov["local_path"] = overlay_videos.get(ov.get("source_video_id", ""), "")

    clip_words    = [w for w in words if w["end_ms"] >= clip_start_ms and w["start_ms"] <= clip_end_ms]
    caption_style = caption_styles[0] if caption_styles else {}
    qs            = _QUALITY.get(out_w, _QUALITY[1080])

    print(f"[render] Clip {clip_start_ms}ms–{clip_end_ms}ms → {out_w}x{out_h}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:

        # ── Audio ──────────────────────────────────────────────────────────────
        audio_path    = os.path.join(tmp, "audio.aac")
        speech_ranges = extract_speech_ranges(clip_words)
        ar = subprocess.run(
            build_segment_audio_args(
                video_path, clip_start_ms, clip_end_ms,
                segments, secondary_videos, audio_tracks, speech_ranges, audio_path,
            ),
            capture_output=True,
        )
        if ar.returncode != 0:
            raise RuntimeError(f"Audio extraction failed:\n{ar.stderr.decode()[-500:]}")

        # ── ASS captions ───────────────────────────────────────────────────────
        ass_path = None
        if clip_words and caption_style:
            ass_path = os.path.join(tmp, "captions.ass")
            _write_ass(clip_words, caption_style, clip_start_ms, out_w, out_h, ass_path,
                       band_zones=caption_band_zones(segments, out_h))

        # ── Count how many times each source video is needed ───────────────────
        # FFmpeg requires explicit split() when a stream is consumed more than once.
        slot_counts: dict[str | None, int] = {None: 0}
        for seg in segments:
            boxes   = sorted(seg.get("crop_boxes", []), key=lambda b: b.get("slot_index", 0))
            if is_frame(seg.get("layout")):
                # Only slots showing the main video read it; lane items are inputs of their own
                main_slots = set(frame_state(seg).get("main_slots") or [])
                for kind, slot, _ in frame_rows(seg["layout"], out_h, frame_band_shown(seg)):
                    if kind == "slot" and slot in main_slots:
                        slot_counts[None] = slot_counts.get(None, 0) + 1
                continue
            n_slots = {"split": 2, "trio": 3}.get(seg.get("layout", "vertical"), 1)
            # For B-roll INSERT segments, remember the primary (slot-0) source so empty
            # extra slots fall back to it instead of the main video.
            primary_vid = (boxes[0].get("source_video_id") if boxes else None)
            primary_vid = primary_vid if (primary_vid and primary_vid in secondary_videos) else None
            for i in range(n_slots):
                box    = boxes[i] if i < len(boxes) else None
                vid_id = box.get("source_video_id") if box else None
                if not (vid_id and vid_id in secondary_videos) and i > 0 and primary_vid:
                    vid_id = primary_vid
                if vid_id and vid_id in secondary_videos:
                    slot_counts[vid_id] = slot_counts.get(vid_id, 0) + 1
                else:
                    slot_counts[None] = slot_counts.get(None, 0) + 1

        # ── FFmpeg inputs ──────────────────────────────────────────────────────
        clip_start_s = clip_start_ms / 1000.0
        clip_dur_s   = clip_dur_ms   / 1000.0
        inputs = [
            "ffmpeg", "-y",
            "-ss", f"{clip_start_s:.3f}",
            "-t",  f"{clip_dur_s:.3f}",
            "-i",  video_path,    # index 0
            "-i",  audio_path,    # index 1
        ]

        sec_input_idx: dict[str, int] = {}
        for i, (vid_id, path) in enumerate(secondary_videos.items()):
            sec_input_idx[vid_id] = i + 2
            inputs += ["-stream_loop", "-1", "-i", path]

        img_base = 2 + len(secondary_videos)
        valid_img: list[tuple[dict, str]] = []
        for ov in img_overlays:
            lp = ov.get("local_path", "")
            if lp and os.path.exists(lp):
                inputs += ["-i", lp]
                valid_img.append((ov, f"[{img_base + len(valid_img)}:v]"))

        vid_base = img_base + len(valid_img)
        valid_vid: list[tuple[dict, str]] = []
        for ov in vid_overlays:
            lp = ov.get("local_path", "")
            if lp and os.path.exists(lp):
                inputs += ["-i", lp]
                valid_vid.append((ov, f"[{vid_base + len(valid_vid)}:v]"))

        # Frame lane items: every video and photo is an input of its own, opened at the point it
        # starts, so nothing has to be buffered until the item appears
        next_input = vid_base + len(valid_vid)
        frame_item_in: dict[tuple[int, str], str] = {}
        for si, seg in enumerate(segments):
            if not is_frame(seg.get("layout")):
                continue
            st = frame_state(seg)
            for kind, slot, _ in frame_rows(seg["layout"], out_h, frame_band_shown(seg)):
                if kind != "slot":
                    continue
                for it in lane_items(seg, st, slot):
                    if it.get("kind") == "video" and it.get("source_video_id") in secondary_videos:
                        path = secondary_videos[it["source_video_id"]]
                        off = it["off_ms"] / 1000.0
                        length, _ = _probe(path)
                        if length > 0:
                            off = off % length  # videos loop: a start past the end wraps around
                        inputs += ["-stream_loop", "-1", "-ss", f"{off:.3f}", "-i", path]
                    elif it.get("kind") == "photo" and os.path.exists(frame_images.get(it.get("image_path") or "", "")):
                        inputs += ["-loop", "1", "-framerate", "30", "-t", f"{it['dur_s']:.3f}", "-i", frame_images[it["image_path"]]]
                    else:
                        continue
                    frame_item_in[(si, it["id"])] = f"[{next_input}:v]"
                    next_input += 1

        # ── Build filter_complex ───────────────────────────────────────────────
        fp: list[str] = []

        # Pre-allocate split pools so each slot gets its own named stream copy.
        # Primary video always gets setpts to normalise PTS after the -ss seek.
        src_pool: dict[str | None, list[str]] = {}
        for src_key, count in slot_counts.items():
            if count == 0:
                continue  # source never used — don't create an unconnected pad
            if src_key is None:
                if count == 1:
                    fp.append("[0:v]setpts=PTS-STARTPTS[p0]")
                    src_pool[None] = ["[p0]"]
                else:
                    labels = [f"[p{i}]" for i in range(count)]
                    fp.append(f"[0:v]setpts=PTS-STARTPTS,split={count}{''.join(labels)}")
                    src_pool[None] = labels
            else:
                idx  = sec_input_idx[src_key]
                safe = src_key.replace("-", "_").replace(":", "_")
                if count == 1:
                    src_pool[src_key] = [f"[{idx}:v]"]  # consumed once — no split needed
                else:
                    labels = [f"[s{safe}{i}]" for i in range(count)]
                    fp.append(f"[{idx}:v]split={count}{''.join(labels)}")
                    src_pool[src_key] = labels

        def pop_src(vid_id: str | None) -> str:
            return src_pool[vid_id].pop(0)

        # ── Segment filters ────────────────────────────────────────────────────
        seg_labels: list[str] = []

        for si, seg in enumerate(segments):
            boxes   = sorted(seg.get("crop_boxes", []), key=lambda b: b.get("slot_index", 0))
            layout  = seg.get("layout", "vertical")
            smss    = int(seg["start_ms"])   # timeline ms (may differ from video ms after INSERT)
            emss    = int(seg["end_ms"])
            dur_ms  = emss - smss
            # video_offset_ms is set when this segment was pushed forward by a true INSERT.
            # For non-pushed main-video segments, fall back to crop_box[0].source_offset_ms,
            # which stores the correct video position (e.g. after Part B, the continuation
            # starts at the video position where Part B left off, not at start_ms).
            _primary_box = boxes[0] if boxes else None
            _primary_box_vid_id = _primary_box.get("source_video_id") if _primary_box else None
            _is_main_video = not (_primary_box_vid_id and _primary_box_vid_id in secondary_videos)
            if seg.get("video_offset_ms") is not None:
                vid_start_ms = int(seg["video_offset_ms"])
            elif _is_main_video and _primary_box and _primary_box.get("source_offset_ms") is not None:
                vid_start_ms = int(_primary_box["source_offset_ms"])
            else:
                vid_start_ms = smss
            start_s = vid_start_ms / 1000.0
            end_s   = (vid_start_ms + dur_ms) / 1000.0
            out_lbl = f"[seg{si}]"

            if is_frame(layout):
                dur_s = dur_ms / 1000.0
                st = frame_state(seg)
                main_slots = set(st.get("main_slots") or [])
                band = st.get("band") or {}
                band_bg = _hex6(band.get("bg"), "000000")
                row_lbls: list[str] = []
                for ri, (kind, slot, rh) in enumerate(frame_rows(layout, out_h, frame_band_shown(seg))):
                    lbl = f"[fr{si}r{ri}]"
                    cur_row = f"[fr{si}r{ri}b]"
                    # Underneath: the band colour, the main video (framed by this slot's crop box), or an empty dark slot
                    if kind == "band":
                        fp.append(f"color=c=0x{band_bg}:s={out_w}x{rh}:d={dur_s:.3f}:r=30,format=yuv420p,setsar=1{cur_row}")
                    elif slot in main_slots:
                        box = next((b for b in boxes if b.get("slot_index") == slot), None)
                        off_ms = seg.get("video_offset_ms")
                        if off_ms is None:
                            off_ms = box.get("source_offset_ms") if box and box.get("source_offset_ms") is not None else smss
                        ts = int(off_ms) / 1000.0
                        fp.append(f"{pop_src(None)}trim=start={ts:.3f}:end={ts + dur_s:.3f},setpts=PTS-STARTPTS,"
                                  f"{_crop_filter(box, smss)},{_scale_cover(out_w, rh)},setsar=1,format=yuv420p{cur_row}")
                    else:
                        fp.append(f"color=c=0x111111:s={out_w}x{rh}:d={dur_s:.3f}:r=30,format=yuv420p,setsar=1{cur_row}")

                    # On top: this lane's items, each only during its own time
                    for ii, it in enumerate(lane_items(seg, st, "band" if kind == "band" else slot)):
                        d, rel = it["dur_s"], it["rel_s"]
                        ik = it.get("kind")
                        if ik == "text":
                            if it.get("captions"):
                                continue  # the captions themselves are moved into the band (see _write_ass)
                            chain = _frame_text_chain(
                                it.get("text") or "", _hex6(it.get("bg"), band_bg if kind == "band" else "000000"),
                                _hex6(it.get("color"), "ffffff"), int(it.get("size") or 64), out_w, rh, d)
                        elif ik == "photo" and (si, it["id"]) in frame_item_in:
                            src = frame_item_in[(si, it["id"])]
                            zp = photo_motion_filter(it.get("motion") or "none", out_w, rh, d)
                            if zp:
                                chain = (f"{src}scale={out_w * 2}:{rh * 2}:force_original_aspect_ratio=increase,crop={out_w * 2}:{rh * 2},"
                                         f"{zp},setsar=1,format=yuv420p,trim=duration={d:.3f},setpts=PTS-STARTPTS")
                            else:
                                chain = f"{src}{_scale_cover(out_w, rh)},setsar=1,format=yuv420p,trim=duration={d:.3f},setpts=PTS-STARTPTS"
                        elif ik == "video" and (si, it["id"]) in frame_item_in:
                            src = frame_item_in[(si, it["id"])]
                            chain = f"{src}setpts=PTS-STARTPTS,trim=duration={d:.3f},{_scale_cover(out_w, rh)},setsar=1,format=yuv420p"
                        else:
                            print(f"[render] frame item skipped (media missing): {ik} {it.get('id')}", flush=True)
                            continue
                        il, ol = f"[fr{si}r{ri}i{ii}]", f"[fr{si}r{ri}o{ii}]"
                        fp.append(f"{chain},setpts=PTS+{rel:.3f}/TB{il}")
                        fp.append(f"{cur_row}{il}overlay=0:0:eof_action=pass:enable='between(t,{rel:.3f},{rel + d:.3f})'{ol}")
                        cur_row = ol
                    fp.append(f"{cur_row}null{lbl}")
                    row_lbls.append(lbl)
                # One row (e.g. Single with no text yet) is the whole frame; vstack needs two or more
                if len(row_lbls) == 1:
                    fp.append(f"{row_lbls[0]}null{out_lbl}")
                else:
                    fp.append(f"{''.join(row_lbls)}vstack=inputs={len(row_lbls)}{out_lbl}")
                seg_labels.append(out_lbl)
                continue

            # Primary (slot-0) B-roll source for this segment, used as fallback for empty slots.
            _primary_broll = _primary_box_vid_id if (_primary_box_vid_id and _primary_box_vid_id in secondary_videos) else None

            def trim_slot(slot_i: int, dst_w: int, dst_h: int, fit: bool, lbl: str) -> None:
                box    = boxes[slot_i] if slot_i < len(boxes) else None
                vid_id = box.get("source_video_id") if box else None

                # If this extra slot has no B-roll source but the primary slot does, inherit
                # the primary B-roll so the split renders the INSERT video everywhere (matching
                # the editor preview, which paints brollVid for all slots).
                if not (vid_id and vid_id in secondary_videos) and slot_i > 0 and _primary_broll:
                    box    = boxes[0]
                    vid_id = _primary_broll

                if vid_id and vid_id in secondary_videos:
                    # Secondary: trim from source_offset_ms for segment duration
                    off_ms = int(box.get("source_offset_ms", 0))
                    src    = pop_src(vid_id)
                    ts, te = off_ms / 1000.0, (off_ms + dur_ms) / 1000.0
                else:
                    # Primary: trim clip-relative segment window
                    src    = pop_src(None)
                    ts, te = start_s, end_s

                # Keyframes are stored at composite currentTimeMs (smss-relative for each segment),
                # so use smss as the base — not vid_start_ms, which differs for Part B segments.
                crop  = _crop_filter(box, smss)
                scale = _scale_fit(dst_w, dst_h) if fit else _scale_cover(dst_w, dst_h)
                fp.append(
                    f"{src}trim=start={ts:.3f}:end={te:.3f},setpts=PTS-STARTPTS,"
                    f"{crop},{scale},setsar=1{lbl}"
                )

            if layout in ("vertical", "spotlight", "centered"):
                trim_slot(0, out_w, out_h, False, out_lbl)
            elif layout == "horizontal":
                trim_slot(0, out_w, out_h, True, out_lbl)
            elif layout == "split":
                slot_h = out_h // 2
                trim_slot(0, out_w, slot_h, False, f"[sp{si}a]")
                trim_slot(1, out_w, slot_h, False, f"[sp{si}b]")
                fp.append(f"[sp{si}a][sp{si}b]vstack=inputs=2{out_lbl}")
            elif layout == "trio":
                slot_h = out_h // 3
                trim_slot(0, out_w, slot_h, False, f"[tr{si}a]")
                trim_slot(1, out_w, slot_h, False, f"[tr{si}b]")
                trim_slot(2, out_w, slot_h, False, f"[tr{si}c]")
                fp.append(f"[tr{si}a][tr{si}b][tr{si}c]vstack=inputs=3{out_lbl}")
            else:
                trim_slot(0, out_w, out_h, False, out_lbl)

            seg_labels.append(out_lbl)

        # ── Concatenate segments ───────────────────────────────────────────────
        if len(seg_labels) == 1:
            cur = seg_labels[0]
        else:
            n = len(seg_labels)
            fp.append(f"{''.join(seg_labels)}concat=n={n}:v=1:a=0[vmain]")
            cur = "[vmain]"

        # ── Colour filters (eq) ────────────────────────────────────────────────
        br = float(filters_cfg.get("brightness", 100))
        co = float(filters_cfg.get("contrast",   100))
        sa = float(filters_cfg.get("saturation", 100))
        if br != 100 or co != 100 or sa != 100:
            # FFmpeg eq: brightness ∈ [-1,1], contrast ∈ [0,∞], saturation ∈ [0,∞]
            fp.append(
                f"{cur}eq=brightness={(br-100)/100:.4f}"
                f":contrast={co/100:.4f}:saturation={sa/100:.4f}[veq]"
            )
            cur = "[veq]"

        # ── Subtitles (ASS captions) ───────────────────────────────────────────
        if ass_path:
            # Escape path for FFmpeg filter option: quoted, forward slashes, and inside the quotes
            # ':' (Windows drive letters) still needs escaping
            def _esc_path(p: str) -> str:
                return "'" + p.replace("\\", "/").replace(":", "\\:").replace("'", "") + "'"
            esc_ass   = _esc_path(ass_path)
            esc_fonts = _esc_path(_FONTS_DIR)
            fp.append(f"{cur}subtitles=filename={esc_ass}:fontsdir={esc_fonts}[vcap]")
            cur = "[vcap]"

        # ── Text overlays (drawtext) ───────────────────────────────────────────
        for oi, ov in enumerate(text_overlays):
            text = _escape_drawtext(ov.get("text") or "")
            if not text:
                continue
            font_path = _find_font_path(ov.get("font") or "roboto")
            if not font_path:
                continue
            sz   = int(ov.get("size") or 48)
            hx   = (ov.get("color") or "#ffffff").lstrip("#")
            r, g, b2 = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
            sx   = int((ov.get("x") or 0.1) * out_w)
            sy   = int((ov.get("y") or 0.1) * out_h)
            t0   = ov.get("start_ms", 0) / 1000.0
            t1   = ov.get("end_ms", clip_dur_ms) / 1000.0
            olbl = f"[vdt{oi}]"
            fp.append(
                f"{cur}drawtext=fontfile={font_path}:text='{text}':fontsize={sz}"
                f":fontcolor=0x{r:02X}{g:02X}{b2:02X}:x={sx}:y={sy}"
                f":shadowx=2:shadowy=2:shadowcolor=black@0.7"
                f":enable='between(t,{t0:.3f},{t1:.3f})'{olbl}"
            )
            cur = olbl

        # ── Image overlays ─────────────────────────────────────────────────────
        for oi, (ov, img_lbl) in enumerate(valid_img):
            iw   = max(1, int(ov.get("w", 0.2) * out_w))
            ih   = max(1, int(ov.get("h", 0.2) * out_h))
            ix   = int(ov.get("x", 0) * out_w)
            iy   = int(ov.get("y", 0) * out_h)
            t0   = ov.get("start_ms", 0) / 1000.0
            t1   = ov.get("end_ms", clip_dur_ms) / 1000.0
            slbl = f"[img{oi}s]"
            olbl = f"[vov{oi}]"
            fp.append(f"{img_lbl}scale={iw}:{ih}:flags=lanczos{slbl}")
            fp.append(
                f"{cur}{slbl}overlay=x={ix}:y={iy}"
                f":enable='between(t,{t0:.3f},{t1:.3f})'{olbl}"
            )
            cur = olbl

        # ── Video (PiP) overlays ───────────────────────────────────────────────
        for vi, (ov, vid_lbl) in enumerate(valid_vid):
            vw     = max(1, int(ov.get("w", 0.25) * out_w))
            vh     = max(1, int(ov.get("h", 0.25) * out_h))
            vx     = int(ov.get("x", 0.05) * out_w)
            vy     = int(ov.get("y", 0.05) * out_h)
            t0     = ov.get("start_ms", 0) / 1000.0
            t1     = ov.get("end_ms", clip_dur_ms) / 1000.0
            off_s  = ov.get("source_offset_ms", 0) / 1000.0
            pip_dur = t1 - t0
            tslbl  = f"[pip{vi}t]"
            sclbl  = f"[pip{vi}s]"
            olbl   = f"[vpip{vi}]"
            # Trim overlay video to the display window starting at source_offset_ms
            fp.append(
                f"{vid_lbl}trim=start={off_s:.3f}:duration={pip_dur:.3f},"
                f"setpts=PTS-STARTPTS{tslbl}"
            )
            fp.append(f"{tslbl}scale={vw}:{vh}:flags=lanczos{sclbl}")
            fp.append(
                f"{cur}{sclbl}overlay=x={vx}:y={vy}"
                f":enable='between(t,{t0:.3f},{t1:.3f})'{olbl}"
            )
            cur = olbl

        # ── Watermark (free plan only) ─────────────────────────────────────────
        if watermark:
            wm_font = _find_font_path("roboto")
            wm_text = _escape_drawtext("Chai Cut")
            if wm_font:
                fp.append(
                    f"{cur}drawtext=fontfile={wm_font}:text='{wm_text}'"
                    f":fontsize={max(24, out_w // 36)}:fontcolor=white@0.55"
                    f":x=w-tw-{max(16, out_w // 60)}:y=h-th-{max(16, out_h // 120)}"
                    f":shadowx=1:shadowy=1:shadowcolor=black@0.5[vwm]"
                )
                cur = "[vwm]"

        # Rename final label to [vout]
        fp.append(f"{cur}copy[vout]")

        # ── Assemble and run ───────────────────────────────────────────────────
        cmd = inputs + [
            "-filter_complex", ";".join(fp),
            "-map", "[vout]",
            "-map", "1:a",
            "-c:v", "libx264",
            # x264 defaults to ~1.5 threads per CPU core; on large Railway hosts that
            # makes encoder startup allocate GBs and fail ("Error while opening encoder")
            "-threads", "8",
            "-pix_fmt", "yuv420p",
            "-profile:v", "main",
            "-level", "4.0",
            "-crf", str(qs["crf"]),
            "-preset", qs["preset"],
            "-maxrate", qs["maxrate"],
            "-bufsize", qs["bufsize"],
            "-c:a", "aac",
            "-b:a", qs["audio_br"],
            "-shortest",
            output_path,
        ]

        fc_str = ";".join(fp)
        print(f"[render] filter_complex ({len(fc_str)} chars): {fc_str[:800]}", flush=True)
        print(f"[render] Running FFmpeg (crf={qs['crf']}, {out_w}x{out_h}, {len(segments)} seg(s)) ...", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[render] FFmpeg stderr:\n{result.stderr[-4000:]}", flush=True)
            raise RuntimeError(f"FFmpeg render failed (exit {result.returncode})")

    print(f"[render] Done → {output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",            required=True)
    parser.add_argument("--spec",             required=True)
    parser.add_argument("--output",           required=True)
    parser.add_argument("--secondary-videos", default="{}", help="JSON: video_id → local_path")
    parser.add_argument("--overlay-images",   default="{}", help="JSON: storage_path → local_path")
    parser.add_argument("--overlay-videos",   default="{}", help="JSON: source_video_id → local_path")
    parser.add_argument("--watermark",        action="store_true", help="Burn in Chai Cut watermark")
    parser.add_argument("--frame-images",     default="{}", help="JSON: frame slot image_path → local_path")
    args = parser.parse_args()
    main(
        args.video, args.spec, args.output,
        secondary_videos=json.loads(args.secondary_videos),
        overlay_images=json.loads(args.overlay_images),
        overlay_videos=json.loads(args.overlay_videos),
        watermark=args.watermark,
        frame_images=json.loads(args.frame_images),
    )
