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
from audio import build_ffmpeg_audio_args, build_segment_audio_args, extract_speech_ranges, heard_speech_ranges
import emoji as emoji_art   # colour emoji (Twemoji 17), drawn locally
from frames import is_frame, frame_rows, row_heights, wrap_band_text, photo_motion_filter, frame_state, frame_band_shown, lane_items, caption_band_zones, corner_geometry

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
    "noto-sans-tamil":      "NotoSansTamil-Regular.ttf",
    "noto-sans-kannada":    "NotoSansKannada-Regular.ttf",
    "noto-sans-malayalam":  "NotoSansMalayalam-Regular.ttf",
    "noto-sans-bengali":    "NotoSansBengali-Regular.ttf",
    "roboto":               "Roboto-Regular.ttf",
    "montserrat-bold":      "Montserrat-Bold.ttf",
}
_FONT_NAMES = {
    "noto-sans-telugu":     "Noto Sans Telugu",
    "noto-sans-devanagari": "Noto Sans Devanagari",
    "noto-sans-tamil":      "Noto Sans Tamil",
    "noto-sans-kannada":    "Noto Sans Kannada",
    "noto-sans-malayalam":  "Noto Sans Malayalam",
    "noto-sans-bengali":    "Noto Sans Bengali",
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


def _group_sentences(words: list[dict], max_words: int = _MAX_PHRASE_WORDS) -> list[list[dict]]:
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
            or len(current) >= max_words
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
               "end_ms":   w["end_ms"]   - clip_start_ms + offset_ms,
               "_src_start": w["start_ms"]}
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
        *_preset_styles(style, font_name, font_size),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    if style.get("animation") in _PRESETS:
        lines += _preset_events(clip_words, style, out_w, out_h, band_zones or [])
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(emoji_art.colour_emoji_ass(lines)) + "\n")
        return

    # Karaoke: the spoken word in the highlight colour, lit until the next word starts (the
    # editor's drawCaptions does the same)
    karaoke = (style.get("animation") or "karaoke") == "karaoke"
    rest_c = _ass_color(color_hex)
    live_c = _ass_color(_karaoke_live(color_hex, style.get("highlight_color") or "#FFE700"))

    sentences = [s for s in _group_sentences(clip_words) if s]
    for i, (sentence, (start_ms, end_ms)) in enumerate(zip(sentences, _line_times(sentences))):
        if end_ms <= start_ms:
            continue
        shown = [(w, _word_display(w, roman)) for w in sentence]
        shown = [(w, t) for w, t in shown if t]
        text     = " ".join(t for _, t in shown)
        if not text.strip():
            continue
        # Alignment=5 (center of screen); \pos pins the anchor to exact coordinates. Where a
        # frame shows captions in its text band, that part of the line is centred in the band.
        for a, b, y in _split_by_zones(start_ms, end_ms, band_zones or [], pos_y):
            tag = f"{{\\pos({pos_x},{y})}}"
            body = text
            if karaoke:
                runs = []
                for j, (w, t) in enumerate(shown):
                    on = w["start_ms"] - a
                    off = (shown[j + 1][0]["start_ms"] if j + 1 < len(shown) else end_ms) - a
                    tags = f"\\1c{live_c if on <= 0 < off else rest_c}"
                    if on > 0:
                        tags += f"\\t({on},{on + 1},\\1c{live_c})"
                    if 0 < off < b - a:
                        tags += f"\\t({off},{off + 1},\\1c{rest_c})"
                    runs.append(f"{{{tags}}}{t}")
                body = " ".join(runs)
            lines.append(
                f"Dialogue: 0,{_ms_to_ass_ts(a)},{_ms_to_ass_ts(b)},"
                f"Default,,0,0,0,,{tag}{body}"
            )

    # Emoji in full colour: stacked colour layers on top of the line (emoji.colour_emoji_ass)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(emoji_art.colour_emoji_ass(lines)) + "\n")


def _line_times(sentences: list[list[dict]]) -> list[tuple[int, int]]:
    """When each caption line is on screen: (start, end) per line, clip-relative ms."""
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

    out: list[tuple[int, int]] = []
    for i in range(len(spans)):
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
        out.append((start_ms, end_ms))
    return out


# ── Animated caption presets ──────────────────────────────────────────────────
#
# Mirrored by drawPresetCaptions() in the editor (VideoPreview.tsx): same lines, same timing,
# same colours. The older animations (karaoke / fade / none) keep the plain path above.
#   pop       — each word appears when spoken, scaling 80% → 110% → 100% in 150 ms
#   highlight — 3 words a line; a highlight_color box behind the spoken word
#   bounce    — the line slides up into place and fades in; the spoken word in highlight_color
#   word      — one big word at a time, centre screen
#   hormozi   — 3 capitalised words a line, thick outline; the spoken word in highlight_color, 115%
#   box       — the line on a dark box, no outline; the spoken word in highlight_color
#   glow      — words glow in highlight_color; the spoken word bright, the others dimmed
# Emphasised words (caption_styles.emphasis, keyed by the word's start_ms in the video) are drawn
# in highlight_color and 15% larger.

_PRESETS = {"pop", "highlight", "bounce", "word", "hormozi", "box", "glow"}
_POP_MS = 150
_BOUNCE_MS = 180
_BOX_PAD = 14          # highlight box padding (px at 1080 wide)
_WORD_SCALE = 150      # 'word' preset: % of the caption size
_EMPHASIS_SCALE = 115
_HORMOZI_SCALE = 115   # the spoken word
_HORMOZI_STROKE = 2    # added to the outline
_LINE_BOX = "&H60000000"   # box preset: black, ~62% opaque
_GLOW_BLUR = 6
_GLOW_DIM = "&H59&"        # the words not being spoken (~65% opaque)


def _karaoke_live(color: str, hl: str) -> str:
    """The karaoke colour of the spoken word: the highlight colour, unless it is the text's own"""
    if hl.lower() != color.lower():
        return hl
    return "#FFE700" if color.lower() == "#ffffff" else "#FFFFFF"


def _ass_color(hex_color: str) -> str:
    """#RRGGBB → inline ASS colour &HBBGGRR&"""
    return _hex_to_ass(hex_color, 0).replace("&H00", "&H", 1) + "&"


def _text_on(hex_color: str) -> str:
    """Black or white, whichever reads on a box of this colour"""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return "#000000" if 0.299 * r + 0.587 * g + 0.114 * b > 150 else "#FFFFFF"


def _preset_styles(style: dict, font_name: str, font_size: int) -> list[str]:
    if style.get("animation") not in _PRESETS:
        return []
    # The Noto Indic fonts have no Latin letters: Roman-letter captions use Roboto (the editor too)
    if style.get("language") == "roman":
        font_name = _FONT_NAMES["roboto"]
    stroke = int(style.get("stroke_width") if style.get("stroke_width") is not None else 4)
    hl = _hex_to_ass(style.get("highlight_color") or "#FFE700", 0)
    primary = _hex_to_ass(style.get("color") or "#ffffff", 0)
    return [
        f"Style: Preset,{font_name},{font_size},{primary},&H00FFFFFF,&H00000000,&H80000000,"
        f"1,0,0,0,100,100,0,0,1,{stroke},2,5,10,10,10,1",
        # Opaque box (BorderStyle 3) in the highlight colour; its text is never drawn
        f"Style: PresetBox,{font_name},{font_size},&HFF000000,&H00FFFFFF,{hl},&H00000000,"
        f"1,0,0,0,100,100,0,0,3,{_BOX_PAD},0,5,10,10,10,1",
        # box preset: one dark box round the whole line; its text is never drawn
        f"Style: PresetLine,{font_name},{font_size},&HFF000000,&H00FFFFFF,{_LINE_BOX},&H00000000,"
        f"1,0,0,0,100,100,0,0,3,{_BOX_PAD},0,5,10,10,10,1",
    ]


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def _preset_events(words: list[dict], style: dict, out_w: int, out_h: int, zones: list[tuple[int, int, int]]) -> list[str]:
    anim = style["animation"]
    roman = style.get("language") == "roman"
    upper = bool(style.get("uppercase")) or anim == "hormozi"
    color = style.get("color") or "#ffffff"
    hl = style.get("highlight_color") or "#FFE700"
    emphasis = style.get("emphasis") or {}
    if isinstance(emphasis, str):
        try:
            emphasis = json.loads(emphasis)
        except ValueError:
            emphasis = {}
    stroke = int(style.get("stroke_width") if style.get("stroke_width") is not None else 4)
    if anim == "hormozi":
        stroke += _HORMOZI_STROKE
    if anim == "box":
        stroke = 0
    per_line = 1 if anim == "word" else int(style.get("words_per_line") or (3 if anim in ("highlight", "hormozi") else _MAX_PHRASE_WORDS))
    pos_y_frac = style.get("position_y")
    pos_y = int(float(pos_y_frac if pos_y_frac is not None else (0.5 if anim == "word" else 0.84)) * out_h)
    pos_x = out_w // 2
    base_scale = _WORD_SCALE if anim == "word" else 100

    events: list[str] = []
    sentences = [s for s in _group_sentences(words, max(1, per_line)) if s]
    for sentence, (start_ms, end_ms) in zip(sentences, _line_times(sentences)):
        if end_ms <= start_ms:
            continue
        shown = [(w, _word_display(w, roman)) for w in sentence]
        shown = [(w, (t.upper() if upper else t)) for w, t in shown if t]
        if not shown:
            continue
        for a, b, y in _split_by_zones(start_ms, end_ms, zones, pos_y):
            text_runs, box_runs = [], []
            for j, (w, t) in enumerate(shown):
                on = w["start_ms"] - a                      # spoken from (piece-relative)
                off = (shown[j + 1][0]["start_ms"] if j + 1 < len(shown) else end_ms) - a
                emph = bool(emphasis.get(str(w.get("_src_start", w["start_ms"]))))
                sc = round(base_scale * (_EMPHASIS_SCALE if emph else 100) / 100)
                rest = _ass_color(hl if emph else color)
                live = _ass_color(_text_on(hl)) if anim == "highlight" else _ass_color(hl)
                spoken = on <= 0 < off
                tags = f"\\fscx{sc}\\fscy{sc}"
                if anim in ("pop", "word"):
                    if on <= 0:
                        tags += "\\alpha&H00&"
                    else:
                        lo, hi = round(sc * 0.8), round(sc * 1.1)
                        tags = (f"\\alpha&HFF&\\fscx{lo}\\fscy{lo}\\t({on},{on + 1},\\alpha&H00&)"
                                f"\\t({on},{on + _POP_MS // 2},\\fscx{hi}\\fscy{hi})"
                                f"\\t({on + _POP_MS // 2},{on + _POP_MS},\\fscx{sc}\\fscy{sc})")
                if anim == "hormozi":
                    big = round(sc * _HORMOZI_SCALE / 100)
                    if spoken:
                        tags = f"\\fscx{big}\\fscy{big}"
                    if on > 0:
                        tags += f"\\t({on},{on + 1},\\fscx{big}\\fscy{big})"
                    if 0 < off < b - a:
                        tags += f"\\t({off},{off + 1},\\fscx{sc}\\fscy{sc})"
                if anim == "box":
                    tags += "\\bord0\\shad0"
                if anim == "glow":
                    # Text in the caption colour with a soft outline in the highlight colour
                    dim = "&H00&" if spoken else _GLOW_DIM
                    tags += f"\\1c{rest}\\3c{_ass_color(hl)}\\bord3\\shad0\\blur{_GLOW_BLUR}\\alpha{dim}"
                    if on > 0:
                        tags += f"\\t({on},{on + 1},\\alpha&H00&)"
                    if 0 < off < b - a:
                        tags += f"\\t({off},{off + 1},\\alpha{_GLOW_DIM})"
                elif anim == "word":
                    tags += f"\\1c{rest}"
                else:
                    tags += f"\\1c{live if spoken else rest}"
                    if on > 0:
                        tags += f"\\t({on},{on + 1},\\1c{live})"
                    if 0 < off < b - a:
                        tags += f"\\t({off},{off + 1},\\1c{rest})"
                if anim == "highlight":
                    # No outline inside the box; the box itself is drawn by the PresetBox layer
                    tags += f"\\bord{0 if spoken else stroke}\\shad{0 if spoken else 2}"
                    if on > 0:
                        tags += f"\\t({on},{on + 1},\\bord0\\shad0)"
                    if 0 < off < b - a:
                        tags += f"\\t({off},{off + 1},\\bord{stroke}\\shad2)"
                    box = f"\\fscx{sc}\\fscy{sc}\\3a{'&H00&' if spoken else '&HFF&'}"
                    if on > 0:
                        box += f"\\t({on},{on + 1},\\3a&H00&)"
                    if 0 < off < b - a:
                        box += f"\\t({off},{off + 1},\\3a&HFF&)"
                    box_runs.append(f"{{{box}}}{_esc(t)}")
                text_runs.append(f"{{{tags}}}{_esc(t)}")
            if anim == "bounce" and a == start_ms:
                dy = round(out_h * 0.02)
                head = f"{{\\move({pos_x},{y + dy},{pos_x},{y},0,{_BOUNCE_MS})\\fad(80,0)}}"
            else:
                head = f"{{\\pos({pos_x},{y})}}"
            if anim == "box":
                plain = " ".join(_esc(t) for _, t in shown)
                events.append(f"Dialogue: 0,{_ms_to_ass_ts(a)},{_ms_to_ass_ts(b)},PresetLine,,0,0,0,,{head}{plain}")
            if box_runs:
                events.append(f"Dialogue: 0,{_ms_to_ass_ts(a)},{_ms_to_ass_ts(b)},PresetBox,,0,0,0,,{head}{' '.join(box_runs)}")
            events.append(f"Dialogue: 1,{_ms_to_ass_ts(a)},{_ms_to_ass_ts(b)},Preset,,0,0,0,,{head}{' '.join(text_runs)}")
    return events


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


def _filter_path(p: str) -> str:
    """A file path as a quoted filter option (forward slashes; ':' still escaped inside the quotes)."""
    return "'" + p.replace("\\", "/").replace(":", "\\:").replace("'", "") + "'"


def _box_mask(tmp: str, w: int, h: int, x: int, y: int, bw: int, bh: int, r: int) -> str:
    """
    A w×h RGBA image that is black except for a transparent rounded box at (x, y) sized bw×bh with
    corner radius r: laid over media padded with black, it gives the media rounded corners
    wherever its box sits in the slot. Drawn at twice the size and scaled down; made once per shape.
    """
    path = os.path.join(tmp, f"box_{w}x{h}_{x}_{y}_{bw}x{bh}_{r}.png")
    if os.path.exists(path):
        return path
    W, H, X0, Y0, X1, Y1, R = w * 2, h * 2, x * 2, y * 2, (x + bw) * 2, (y + bh) * 2, r * 2
    dx = f"max(max({X0 + R}-X\\,X-{X1 - 1 - R})\\,0)"
    dy = f"max(max({Y0 + R}-Y\\,Y-{Y1 - 1 - R})\\,0)"
    inside = (f"lte({dx}*{dx}+{dy}*{dy}\\,{R * R})*gte(X\\,{X0})*lt(X\\,{X1})*gte(Y\\,{Y0})*lt(Y\\,{Y1})")
    vf = f"format=rgba,geq=r=0:g=0:b=0:a=255*(1-{inside}),scale={w}:{h}:flags=area"
    res = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:d=1",
                          "-vf", vf, "-frames:v", "1", path], capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(path):
        raise RuntimeError(f"box mask failed: {res.stderr[-400:]}")
    return path


def _main_box_px(rect, w: int, h: int) -> tuple[int, int, int, int] | None:
    """The main video's resized box in a w×h slot as even pixels (x, y, bw, bh); None = the whole slot. Mirrors mainRect() in the editor."""
    if not isinstance(rect, dict):
        return None
    try:
        rw, rh = max(0.15, min(1.0, float(rect["w"]))), max(0.15, min(1.0, float(rect["h"])))
        rx, ry = max(0.0, min(1.0 - rw, float(rect["x"]))), max(0.0, min(1.0 - rh, float(rect["y"])))
    except (KeyError, TypeError, ValueError):
        return None
    if rw > 0.999 and rh > 0.999:
        return None
    ev = lambda v: int(round(v / 2)) * 2
    bw, bh = max(8, ev(rw * w)), max(8, ev(rh * h))
    x, y = min(ev(rx * w), w - bw), min(ev(ry * h), h - bh)
    return x, y, bw, bh


def _corner_mask(tmp: str, w: int, h: int, m: int, r: int) -> str:
    """
    A w×h RGBA image that is black except for a transparent rounded box inset by m with corner
    radius r. Laid over media padded with black, it gives the media rounded corners and a black
    border. Drawn at twice the size and scaled down for smooth corners; made once per size.
    """
    path = os.path.join(tmp, f"corners_{w}x{h}_{m}_{r}.png")
    if os.path.exists(path):
        return path
    W, H, M, R = w * 2, h * 2, m * 2, r * 2
    dx = f"max(max({M + R}-X\\,X-{W - M - 1 - R})\\,0)"
    dy = f"max(max({M + R}-Y\\,Y-{H - M - 1 - R})\\,0)"
    inside = (f"lte({dx}*{dx}+{dy}*{dy}\\,{R * R})*gte(X\\,{M})*lt(X\\,{W - M})*gte(Y\\,{M})*lt(Y\\,{H - M})")
    vf = f"format=rgba,geq=r=0:g=0:b=0:a=255*(1-{inside}),scale={w}:{h}:flags=area"
    res = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:d=1",
                          "-vf", vf, "-frames:v", "1", path], capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(path):
        raise RuntimeError(f"corner mask failed: {res.stderr[-400:]}")
    return path


def _frame_text_font(text: str) -> str:
    """Montserrat Bold for Latin text; the Noto font of the Indian script Montserrat can't draw."""
    scripts = (("\u0c00", "\u0c7f", "noto-sans-telugu"), ("\u0900", "\u097f", "noto-sans-devanagari"),
               ("\u0b80", "\u0bff", "noto-sans-tamil"), ("\u0c80", "\u0cff", "noto-sans-kannada"),
               ("\u0d00", "\u0d7f", "noto-sans-malayalam"), ("\u0980", "\u09ff", "noto-sans-bengali"))
    font_id = next((fid for a, b, fid in scripts if any(a <= c <= b for c in text)), "montserrat-bold")
    path = _find_font_path(font_id)
    path = path or _find_font_path("roboto")
    # Quoted in the filter; inside the quotes ':' (Windows drive letters) still needs escaping
    return path.replace("\\", "/").replace(":", "\\:") if path else ""


# Line height of a text overlay, in em of its font size (the editor's TEXT_LINE_EM)
_TEXT_LINE_EM = 1.15


def _wrap_text_lines(text: str, max_w: float, measure) -> list[str]:
    """The lines a text overlay is drawn in: each line typed, and with a box width (`max_w` > 0)
    words wrapped onto the next line when the line would be wider than the box; a word wider than
    the box broken across lines. The editor breaks lines by exactly this rule (textStyle.ts
    wrapTextLines), measuring with the same font file."""
    lines: list[str] = []
    for para in text.split("\n"):
        if not max_w or max_w <= 0:
            lines.append(para)
            continue
        words = para.split(" ")
        line = ""
        for i, word in enumerate(words):
            probe = word if i == 0 else f"{line} {word}"
            if i > 0 and line and measure(probe) > max_w:
                lines.append(line)
                line = word
            else:
                line = probe
            # A word wider than the box on its own line: broken across lines, as many letters as
            # fit on each (the editor does the same: textStyle.ts wrapTextLines)
            if line == word and measure(word) > max_w:
                piece = ""
                for ch in word:
                    if piece and measure(piece + ch) > max_w:
                        lines.append(piece)
                        piece = ch
                    else:
                        piece += ch
                line = piece
        lines.append(line)
    return lines


def _rotated_text_png(line_runs: list[list[tuple[bool, str]]], widths: list[list[float]], font_path: str, size: int,
                      color: tuple[int, int, int], rotation: float, block_w: float, block_h: float,
                      shift_y: float, line_h: float, above: int) -> tuple[str, int]:
    """
    A text overlay turned by `rotation` degrees (clockwise, as the editor turns it), as a PNG: its
    lines drawn as the drawtext path draws them (left-aligned, one baseline per line, the same 2 px
    shadow), in a block block_w × block_h px, then turned round the block's middle. Returns the
    PNG and the margin added round the block (the picture's middle is the block's middle).
    """
    from PIL import Image, ImageDraw, ImageFont
    pad = int(size * 0.5) + 4
    img = Image.new("RGBA", (int(block_w + 2 * pad + 0.5), int(block_h + 2 * pad + 0.5)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, size)
    fill = (*color, 255)
    for li, runs in enumerate(line_runs):
        cursor = float(pad)
        base = pad + shift_y + li * line_h + above
        for (is_emoji, piece), w in zip(runs, widths[li]):
            if piece:
                if is_emoji:
                    png = emoji_art.emoji_png(piece, int(size))
                    if png:
                        em = Image.open(png).convert("RGBA")
                        img.alpha_composite(em, (int(cursor + size * emoji_art.SIDE_EM), int(base - size * emoji_art.ABOVE_EM)))
                else:
                    draw.text((cursor + 2, base + 2), piece, font=font, fill=(0, 0, 0, 178), anchor="ls")
                    draw.text((cursor, base), piece, font=font, fill=fill, anchor="ls")
            cursor += w
    # PIL turns anticlockwise for a positive angle; the editor (CSS / canvas) turns clockwise
    img = img.rotate(-rotation, resample=Image.BICUBIC, expand=True)
    fd, path = tempfile.mkstemp(suffix=".png", prefix="rtext_")
    os.close(fd)
    img.save(path)
    return path, pad


def _line_baseline(pil_font, runs: list[tuple[bool, str]], size: int) -> int:
    """
    How far below a line's y its baseline is when the line is drawn in pieces (text runs and emoji
    pictures). drawtext's y is the top of the tallest glyph it draws, so pieces with different
    letters would each sit at a different height; every piece is drawn on this one baseline
    instead (y_align=baseline): the line looks exactly as one drawtext of it would.
    """
    plain = "".join(t for is_e, t in runs if not is_e)
    above = -pil_font.getbbox(plain, anchor="ls")[1] if plain.strip() else 0
    if any(is_e for is_e, _ in runs):
        above = max(above, round(size * emoji_art.ABOVE_EM))
    return above


def _frame_text_chain(fp: list[str], inputs: list[str], next_input: int, key: str,
                      text: str, bg: str, color: str, size_1080: int, w: int, h: int, dur_s: float,
                      cx: float = 0.5, cy: float = 0.5) -> tuple[str, int]:
    """A solid w×h card with wrapped text centred on (cx, cy) — shares of the card, set by dragging
    it in the editor. Appends its filter statements onto `fp` (and any emoji PNG inputs onto
    `inputs`, same scheme as the main text_overlays loop) and returns the finished node's label
    plus the updated next_input counter."""
    base_lbl = f"[{key}base]"
    fp.append(f"color=c=0x{bg}:s={w}x{h}:d={dur_s:.3f}:r=30,format=yuv420p{base_lbl}")
    cur = base_lbl
    text = (text or "").strip()
    font = _frame_text_font(text) if text else ""
    ridx = 0
    if text and font:
        size = max(8, int(round(size_1080 * w / 1080)))
        # Wrap at 1080-wide scale, like the editor preview, so lines break in the same places
        lines = wrap_band_text(text, 1080, size_1080)
        lh = int(size * 1.2)
        top = cy * h - lh * len(lines) / 2
        for li, ln in enumerate(lines):
            if not ln:
                continue
            y = int(top + li * lh + (lh - size) / 2)
            runs = emoji_art.split_emoji(ln)
            if not any(is_e for is_e, _ in runs):
                olbl = f"[{key}t{ridx}]"
                fp.append(f"{cur}drawtext=fontfile='{font}':{_text_opt(ln)}:fontsize={size}"
                          f":fontcolor=0x{color}:x={cx * w:.1f}-tw/2:y={y}{olbl}")
                cur = olbl
                ridx += 1
                continue
            # Has emoji: same per-run drawtext/overlay composite as the main text_overlays loop,
            # just centred on this line's own width instead of ffmpeg's (w-text_w)/2 expression
            from PIL import ImageFont
            # `font` is escaped for the filter (Windows drive letters): Pillow needs the plain path
            pil_font = ImageFont.truetype(font.replace("\:", ":"), size)
            # An emoji takes the Chai Emoji font's room (1.1 em, its square 0.05 em in from each side)
            # and sits on the baseline (0.82 em above, 0.18 em below) — as the editor's preview draws it
            run_widths = [size * emoji_art.ADVANCE_EM if is_e else pil_font.getlength(s) for is_e, s in runs]
            base = y + _line_baseline(pil_font, runs, size)
            cursor = cx * w - sum(run_widths) / 2
            for (is_emoji, s), rw in zip(runs, run_widths):
                if not s:
                    continue
                if is_emoji:
                    png = emoji_art.emoji_png(s, int(size))
                    if png:
                        inputs += ["-i", png]
                        src_lbl = f"[{next_input}:v]"
                        next_input += 1
                        slbl, olbl = f"[{key}e{ridx}s]", f"[{key}e{ridx}]"
                        fp.append(f"{src_lbl}format=rgba{slbl}")
                        fp.append(f"{cur}{slbl}overlay=x={int(cursor + size * emoji_art.SIDE_EM)}"
                                  f":y={int(base - size * emoji_art.ABOVE_EM)}{olbl}")
                        cur = olbl
                        ridx += 1
                else:
                    olbl = f"[{key}t{ridx}]"
                    fp.append(f"{cur}drawtext=fontfile='{font}':{_text_opt(s)}:fontsize={size}"
                              f":fontcolor=0x{color}:x={int(cursor)}:y_align=baseline:y={base}{olbl}")
                    cur = olbl
                    ridx += 1
                cursor += rw
    final_lbl = f"[{key}fin]"
    fp.append(f"{cur}setsar=1{final_lbl}")
    return final_lbl, next_input


# ── FFmpeg crop expression builder ────────────────────────────────────────────

_MAX_KF_PER_ATTR = 400  # points per crop after simplification (a balanced if() tree: ~9 levels)


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

_ZOOM_STEP_S = 0.1  # a zoom that glides is re-sent this often (10 steps a second)
_MAX_ZOOM_CMDS = 1500
_crop_seq = 0       # unique crop@name per filter so sendcmd hits the right one


def _view_points(kf_list: list[dict], seg_start_ms: int) -> list[dict]:
    """
    Keyframes as the editor preview plays them: straight lines between keyframes, so a
    "cut" (a hold keyframe 1ms before the change) switches views instantly and Motion
    glides. Nothing is dropped by value — a hold carries the previous value on purpose.
    """
    pts = [
        {"t": (k["t_ms"] - seg_start_ms) / 1000.0, "x": k["x"], "y": k["y"], "w": k["w"], "h": k["h"]}
        for k in sorted(kf_list, key=lambda k: k["t_ms"])
    ]
    # Recorded Motion can hold thousands of points; thin only what a straight line already
    # describes (RDP keeps cuts: the hold sits far off the line to the next view)
    tol = 0.002
    simplified = _rdp_simplify([{**p, "t_ms": p["t"]} for p in pts], tol)
    while len(simplified) > _MAX_KF_PER_ATTR:
        tol *= 2
        simplified = _rdp_simplify([{**p, "t_ms": p["t"]} for p in pts], tol)
    return simplified


def _value_at(pts: list[dict], attr: str, t: float) -> float:
    if t <= pts[0]["t"]:
        return pts[0][attr]
    for a, b in zip(pts, pts[1:]):
        if t < b["t"]:
            span = b["t"] - a["t"]
            return a[attr] if span <= 0 else a[attr] + (b[attr] - a[attr]) * (t - a["t"]) / span
    return pts[-1][attr]


def _piecewise_expr(pts: list[dict], attr: str) -> str:
    """
    Per-frame expression for one attribute: a balanced if() tree over the keyframe
    intervals (depth log2(n), so hundreds of view changes stay cheap to evaluate).
    """
    dim = "iw" if attr in ("x", "w") else "ih"
    if len(pts) == 1 or all(abs(p[attr] - pts[0][attr]) < 1e-6 for p in pts):
        return f"{dim}*{pts[0][attr]:.6f}"

    def span(i: int) -> str:
        a, b = pts[i], pts[i + 1]
        if abs(b[attr] - a[attr]) < 1e-6:
            return f"{dim}*{a[attr]:.6f}"
        dt = max(b["t"] - a["t"], 0.0005)
        return f"{dim}*({a[attr]:.6f}+({b[attr] - a[attr]:.6f})*clip((t-{a['t']:.4f})/{dt:.4f},0,1))"

    def tree(lo: int, hi: int) -> str:  # spans lo..hi-1; span i covers [t_i, t_i+1)
        if hi - lo == 1:
            return span(lo)
        mid = (lo + hi) // 2
        return f"if(lt(t,{pts[mid]['t']:.4f}),{tree(lo, mid)},{tree(mid, hi)})"

    return tree(0, len(pts) - 1)


def _zoom_commands(pts: list[dict], name: str) -> list[str]:
    """
    FFmpeg's crop reads w/h once when the filter starts, so a view that zooms later would
    keep the first size. Send the new size at every change: once at a cut, in small steps
    across a glide.
    """
    times: list[float] = []
    for a, b in zip(pts, pts[1:]):
        if abs(b["w"] - a["w"]) < 1e-6 and abs(b["h"] - a["h"]) < 1e-6:
            continue
        if b["t"] - a["t"] <= 0.05:
            times.append(b["t"])
        else:
            n = max(1, int((b["t"] - a["t"]) / _ZOOM_STEP_S))
            times.extend(a["t"] + (b["t"] - a["t"]) * i / n for i in range(1, n + 1))
    if len(times) > _MAX_ZOOM_CMDS:
        step = len(times) / _MAX_ZOOM_CMDS
        times = [times[int(i * step)] for i in range(_MAX_ZOOM_CMDS)] + [times[-1]]

    cmds: list[str] = []
    last = (_value_at(pts, "w", 0.0), _value_at(pts, "h", 0.0))
    for t in sorted(set(round(max(t, 0.0), 3) for t in times)):
        w, h = _value_at(pts, "w", t), _value_at(pts, "h", t)
        if abs(w - last[0]) < 1e-6 and abs(h - last[1]) < 1e-6:
            continue
        last = (w, h)
        # No commas in the arguments: inside sendcmd a ',' separates commands
        cmds.append(f"{t:.3f} crop@{name} w iw*{w:.6f}\\, crop@{name} h ih*{h:.6f}")
    return cmds


def _crop_filter(box: dict | None, seg_start_ms: int) -> str:
    """
    Crop that plays each view exactly like the editor preview: a view holds until the next
    change, cuts switch on the frame, Motion glides, and zoom changes apply mid-segment.
    """
    global _crop_seq
    kf = box.get("box_keyframes", []) if box else []
    if not kf:
        return "crop=w=iw:h=ih:x=0:y=0"
    pts = _view_points(kf, seg_start_ms)

    _crop_seq += 1
    name = f"c{_crop_seq}"
    w0, h0 = _value_at(pts, "w", 0.0), _value_at(pts, "h", 0.0)
    w = _esc_expr(f"max(2,iw*{w0:.6f})")
    h = _esc_expr(f"max(2,ih*{h0:.6f})")
    x = _esc_expr(f"min(iw-2,{_piecewise_expr(pts, 'x')})")
    y = _esc_expr(f"min(ih-2,{_piecewise_expr(pts, 'y')})")
    crop = f"crop@{name}=w={w}:h={h}:x={x}:y={y}"

    cmds = _zoom_commands(pts, name)
    if not cmds:
        return crop
    return f"sendcmd=c='{';'.join(cmds)}',{crop}"


def _is_full_frame(box: dict | None) -> bool:
    """Every keyframe of the box frames the whole picture"""
    kfs = (box or {}).get("box_keyframes") or []
    return bool(kfs) and all(abs(float(k.get("x", 0))) < 0.005 and abs(float(k.get("y", 0))) < 0.005
                             and float(k.get("w", 0)) > 0.995 and float(k.get("h", 0)) > 0.995 for k in kfs)


def _even_rows(total: int, n: int) -> list[int]:
    """Heights of n stacked rows that add up to exactly `total`, each even (yuv420p rounds an odd
    height down: at 480p two 427 px halves came out 852 px tall, the other sections 854, and
    FFmpeg refused to join them). The last row takes what's left: 854 → 426 + 428."""
    base = (total // n) // 2 * 2
    return [base] * (n - 1) + [total - base * (n - 1)]


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


def _fit_font_size(text: str, font_path: str, size: int, max_w: int) -> int:
    """The largest size up to `size` at which `text` is at most `max_w` px wide (min 28)."""
    try:
        from PIL import ImageFont  # installed with matplotlib
        width = ImageFont.truetype(font_path, size).getlength(text)
    except Exception:
        return size
    return size if width <= max_w else max(28, int(size * max_w / width))


def _escape_drawtext(s: str) -> str:
    """Every call site wraps this in text='...' and sets expansion=none, so % (e.g. "100%") is drawn
    as typed instead of starting a drawtext %{...} code ("Stray %" failed the whole export). FFmpeg's filtergraph quoting is POSIX-shell-style:
    backslash has no special meaning inside single quotes, so \\' does NOT escape a literal quote
    there — it corrupts quote-tracking for the rest of the filter graph (confirmed: a name with an
    apostrophe crashed every render downstream of it with "No option name near ..."). A literal
    quote has to break out of the string, escape itself, then reopen, exactly like 'it'\\''s' in a
    shell. Everything else (colon, percent, backslash) is already literal inside single quotes —
    escaping those would print a literal backslash in the text instead of protecting anything."""
    return s.replace("'", "'\\''")

# User text is handed to drawtext in a file (textfile=), never inline: FFmpeg parses filter text
# twice, so inline text needs two layers of escaping and a colon, apostrophe or % in it ("It's
# 100%: true") failed the whole export. From a file every character is drawn as typed
# (expansion=none: % isn't a drawtext code).
_TEXT_DIR: str | None = None


def _text_opt(s: str) -> str:
    """drawtext options that draw `s` exactly as typed"""
    global _TEXT_DIR
    if _TEXT_DIR is None:
        import atexit, shutil
        _TEXT_DIR = tempfile.mkdtemp(prefix="chai-text-")
        atexit.register(shutil.rmtree, _TEXT_DIR, True)
    fd, path = tempfile.mkstemp(suffix=".txt", dir=_TEXT_DIR)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(s)
    return f"textfile={_filter_path(path)}:expansion=none"



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

    # UTF-8 explicitly: Windows defaults to cp1252 and fails on Telugu/Hindi captions
    spec           = json.load(open(spec_path, encoding="utf-8"))
    # A crop box showing this id shows the clip's own video at another moment (a "borrowed"
    # reaction, from source_offset_ms): read through its own input, seeked straight there
    main_video_id  = spec.get("main_video_id")

    def is_borrowed(box) -> bool:
        return bool(box and main_video_id and box.get("source_video_id") == main_video_id
                    and main_video_id in secondary_videos)
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
        # Muted sections silence the picture's own sound (not the music) inside the mix (audio.py)
        mute_ranges = [(int(r["start_ms"]), int(r["end_ms"])) for r in spec.get("mute_ranges", [])
                       if int(r["end_ms"]) > int(r["start_ms"])]
        # The music dips only where the voice is actually heard (clip time; words are in video time)
        speech_ranges = heard_speech_ranges(
            extract_speech_ranges(clip_words), clip_start_ms, clip_end_ms - clip_start_ms,
            segments, secondary_videos, mute_ranges, audio_tracks,
            bool(spec.get("main_detached")), float(spec.get("main_volume", 1.0)),
        )
        ar = subprocess.run(
            build_segment_audio_args(
                video_path, clip_start_ms, clip_end_ms,
                segments, secondary_videos, audio_tracks, speech_ranges, audio_path,
                mute_ranges=mute_ranges, main_detached=bool(spec.get("main_detached")),
                main_volume=float(spec.get("main_volume", 1.0)),
            ),
            capture_output=True,
        )
        if ar.returncode != 0:
            raise RuntimeError(f"Audio extraction failed:\n{ar.stderr.decode()[-500:]}")
        print(f"[render] sound: {len(audio_tracks)} track(s), {len(mute_ranges)} muted part(s), levelled to -14 LUFS", flush=True)

        # ── Hidden added videos: the main video shows there (default framing). Their sound was
        #    mixed above by their own switch, so this only changes the picture. ──
        for _i, _s in enumerate(segments):
            _b = (_s.get("crop_boxes") or [{}])[0]
            if _b.get("hidden") and _b.get("source_video_id") and not is_frame(_s.get("layout")):
                _t = int(_s["start_ms"])
                segments[_i] = {**_s, "crop_boxes": [{
                    **_b, "source_video_id": None, "source_offset_ms": _t, "image_path": None, "hidden": False,
                    "box_keyframes": [{"t_ms": _t, "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}],
                }]}

        # ── ASS captions ───────────────────────────────────────────────────────
        ass_path = None
        # enabled False = the user turned captions off (styles saved before that field count as on)
        if clip_words and caption_style and caption_style.get("enabled") is not False:
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
                for kind, slot, _ in frame_rows(seg["layout"], out_h, frame_band_shown(seg), row_heights(seg)):
                    if kind == "slot" and slot in main_slots:
                        slot_counts[None] = slot_counts.get(None, 0) + 1
                continue
            n_slots = {"split": 2, "trio": 3}.get(seg.get("layout", "vertical"), 1)
            # For B-roll INSERT segments, remember the primary (slot-0) source so empty
            # extra slots fall back to it instead of the main video.
            primary_vid = (boxes[0].get("source_video_id") if boxes else None)
            primary_vid = primary_vid if (primary_vid and primary_vid in secondary_videos and not is_borrowed(boxes[0])) else None
            for i in range(n_slots):
                box    = boxes[i] if i < len(boxes) else None
                if is_borrowed(box):
                    continue  # its own input (below)
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
        # The clip's own video is only read by borrowed slots (their own seeked inputs, below)
        looped = [(vid_id, path) for vid_id, path in secondary_videos.items() if vid_id != main_video_id]
        for i, (vid_id, path) in enumerate(looped):
            sec_input_idx[vid_id] = i + 2
            inputs += ["-stream_loop", "-1", "-i", path]

        img_base = 2 + len(looped)
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
            for kind, slot, _ in frame_rows(seg["layout"], out_h, frame_band_shown(seg), row_heights(seg)):
                if kind != "slot":
                    continue
                for it in lane_items(seg, st, slot):
                    if it.get("hidden"):
                        continue  # kept only for its sound (the mix reads it on its own)
                    if it.get("kind") == "video" and it.get("source_video_id") in secondary_videos:
                        path = secondary_videos[it["source_video_id"]]
                        off = it["off_ms"] / 1000.0
                        length, _ = _probe(path)
                        if length > 0:
                            off = off % length  # videos loop: a start past the end wraps around
                        # thread_queue_size: without it, a looped input feeding a complex filter graph
                        # alongside other inputs can overflow ffmpeg's default demuxer->filter queue
                        # ("Failed to inject frame into filter network: Resource temporarily unavailable")
                        inputs += ["-thread_queue_size", "1024", "-stream_loop", "-1", "-ss", f"{off:.3f}", "-i", path]
                    elif it.get("kind") == "photo" and os.path.exists(frame_images.get(it.get("image_path") or "", "")):
                        inputs += ["-thread_queue_size", "1024", "-loop", "1", "-framerate", "30", "-t", f"{it['dur_s']:.3f}", "-i", frame_images[it["image_path"]]]
                    else:
                        continue
                    frame_item_in[(si, it["id"])] = f"[{next_input}:v]"
                    next_input += 1

        # Borrowed reaction slots: the clip's own video from another moment, one input each, opened
        # at that moment (the whole video is never decoded up to it)
        borrowed_in: dict[tuple[int, int], str] = {}
        for si, seg in enumerate(segments):
            if is_frame(seg.get("layout")):
                continue
            for box in seg.get("crop_boxes", []):
                if not is_borrowed(box):
                    continue
                off = max(0.0, int(box.get("source_offset_ms", 0)) / 1000.0)
                dur = (int(seg["end_ms"]) - int(seg["start_ms"])) / 1000.0
                inputs += ["-thread_queue_size", "1024", "-ss", f"{off:.3f}", "-t", f"{dur + 0.5:.3f}", "-i", secondary_videos[main_video_id]]
                borrowed_in[(si, int(box.get("slot_index", 0)))] = f"[{next_input}:v]"
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
            # Borrowed slots show another moment, so the segment's own position comes from a live slot
            _primary_box = next((b for b in boxes if not is_borrowed(b)), boxes[0] if boxes else None)
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

                def media_box(corners, rect, rh: int) -> tuple[int, int, int, int, int, bool]:
                    """
                    Where media goes in a slot of height rh: (width, height) it's scaled to, its top-left
                    (px, py), the corner radius r, and whether it's set on black at all (False = it
                    fills the slot edge to edge). `rect`: its box when resized / moved; `corners`: its
                    rounded border.
                    """
                    g = corner_geometry(corners)
                    m = max(2, int(round(g[0] * out_w / 1080 / 2)) * 2) if g else 0
                    r = max(2, int(round(g[1] * out_w / 1080))) if g else 0
                    px_box = _main_box_px(rect, out_w, rh)
                    bx, by, bw, bh = px_box or (0, 0, out_w, rh)
                    iw, ih = max(2, bw - 2 * m), max(2, bh - 2 * m)
                    return iw, ih, bx + m, by + m, min(r, iw // 2, ih // 2), bool(m or px_box)

                def placed(chain: str, rh: int, iw: int, ih: int, px: int, py: int, r: int, boxed: bool, key: str) -> str:
                    """Media (already iw×ih) set at its place in the slot on black, with rounded corners when it has them."""
                    if not boxed:
                        return chain
                    fp.append(f"{chain},pad={out_w}:{rh}:{px}:{py}:color=black,format=yuv420p[{key}p]")
                    if not r:
                        return f"[{key}p]setsar=1"
                    mask = _box_mask(tmp, out_w, rh, px, py, iw, ih, r)
                    fp.append(f"movie=filename={_filter_path(mask)},format=rgba[{key}k]")
                    return f"[{key}p][{key}k]overlay=0:0,format=yuv420p,setsar=1"

                main_slots = set(st.get("main_slots") or [])
                band = st.get("band") or {}
                band_bg = _hex6(band.get("bg"), "000000")
                row_lbls: list[str] = []
                for ri, (kind, slot, rh) in enumerate(frame_rows(layout, out_h, frame_band_shown(seg), row_heights(seg))):
                    lbl = f"[fr{si}r{ri}]"
                    cur_row = f"[fr{si}r{ri}b]"
                    # Underneath: the band colour, the main video (framed by this slot's crop box), or an empty dark slot
                    if kind in ("band", "caption"):
                        fp.append(f"color=c=0x{band_bg}:s={out_w}x{rh}:d={dur_s:.3f}:r=30,format=yuv420p,setsar=1{cur_row}")
                    elif slot in main_slots:
                        box = next((b for b in boxes if b.get("slot_index") == slot), None)
                        off_ms = seg.get("video_offset_ms")
                        if off_ms is None:
                            off_ms = box.get("source_offset_ms") if box and box.get("source_offset_ms") is not None else smss
                        ts = int(off_ms) / 1000.0
                        iw, ih, px, py, rr, boxed = media_box((st.get("main_corners") or {}).get(str(slot)),
                                                              (st.get("main_rects") or {}).get(str(slot)), rh)
                        chain = (f"{pop_src(None)}trim=start={ts:.3f}:end={ts + dur_s:.3f},setpts=PTS-STARTPTS,"
                                 f"{_crop_filter(box, smss)},{_scale_cover(iw, ih)},setsar=1,format=yuv420p")
                        fp.append(f"{placed(chain, rh, iw, ih, px, py, rr, boxed, f'fr{si}r{ri}m')}{cur_row}")
                    else:
                        fp.append(f"color=c=0x111111:s={out_w}x{rh}:d={dur_s:.3f}:r=30,format=yuv420p,setsar=1{cur_row}")

                    # On top: this lane's items, each only during its own time
                    for ii, it in enumerate([] if kind == "caption" else lane_items(seg, st, "band" if kind == "band" else slot)):
                        if it.get("hidden"):
                            continue  # hidden: what's under it shows (its sound, if on, is still mixed)
                        d, rel = it["dur_s"], it["rel_s"]
                        ik = it.get("kind")
                        if ik == "text":
                            if it.get("captions"):
                                continue  # the captions themselves are moved into the band (see _write_ass)
                            text_lbl, next_input = _frame_text_chain(
                                fp, inputs, next_input, f"fr{si}r{ri}i{ii}txt",
                                it.get("text") or "", _hex6(it.get("bg"), band_bg if kind == "band" else "000000"),
                                _hex6(it.get("color"), "ffffff"), int(it.get("size") or 64), out_w, rh, d,
                                float(it["x"]) if it.get("x") is not None else 0.5,
                                float(it["y"]) if it.get("y") is not None else 0.5)
                            chain = f"{text_lbl}null"  # passthrough: the shared epilogue below appends setpts
                        elif ik == "photo" and (si, it["id"]) in frame_item_in:
                            src = frame_item_in[(si, it["id"])]
                            iw, ih, px, py, rr, boxed = media_box(it.get("corners"), it.get("rect"), rh)
                            zp = photo_motion_filter(it.get("motion") or "none", iw, ih, d)
                            if zp:
                                chain = (f"{src}scale={iw * 2}:{ih * 2}:force_original_aspect_ratio=increase,crop={iw * 2}:{ih * 2},"
                                         f"{zp},setsar=1,format=yuv420p,trim=duration={d:.3f},setpts=PTS-STARTPTS")
                            else:
                                chain = f"{src}{_scale_cover(iw, ih)},setsar=1,format=yuv420p,trim=duration={d:.3f},setpts=PTS-STARTPTS"
                            chain = placed(chain, rh, iw, ih, px, py, rr, boxed, f"fr{si}r{ri}i{ii}c")
                        elif ik == "video" and (si, it["id"]) in frame_item_in:
                            src = frame_item_in[(si, it["id"])]
                            iw, ih, px, py, rr, boxed = media_box(it.get("corners"), it.get("rect"), rh)
                            chain = f"{src}setpts=PTS-STARTPTS,trim=duration={d:.3f},{_scale_cover(iw, ih)},setsar=1,format=yuv420p"
                            chain = placed(chain, rh, iw, ih, px, py, rr, boxed, f"fr{si}r{ri}i{ii}c")
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
                # A split/trio slot framing the whole picture shows it whole (a related visual):
                # fitted with bars, not cropped to fill the slot
                if not fit and layout in ("split", "trio") and _is_full_frame(box):
                    fit = True

                if (si, slot_i) in borrowed_in:
                    # Already opened at its moment: only the length and the crop
                    crop  = _crop_filter(box, smss)
                    scale = _scale_fit(dst_w, dst_h) if fit else _scale_cover(dst_w, dst_h)
                    fp.append(f"{borrowed_in[(si, slot_i)]}setpts=PTS-STARTPTS,trim=duration={dur_ms / 1000.0:.3f},{crop},{scale},setsar=1{lbl}")
                    return

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
                rows = _even_rows(out_h, 2)
                trim_slot(0, out_w, rows[0], False, f"[sp{si}a]")
                trim_slot(1, out_w, rows[1], False, f"[sp{si}b]")
                fp.append(f"[sp{si}a][sp{si}b]vstack=inputs=2{out_lbl}")
            elif layout == "trio":
                rows = _even_rows(out_h, 3)
                trim_slot(0, out_w, rows[0], False, f"[tr{si}a]")
                trim_slot(1, out_w, rows[1], False, f"[tr{si}b]")
                trim_slot(2, out_w, rows[2], False, f"[tr{si}c]")
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

        # ── Hidden main video: black there (captions, text and photos still go on top) ──
        blank_ranges = [(int(r["start_ms"]), int(r["end_ms"])) for r in spec.get("blank_ranges", [])
                        if int(r["end_ms"]) > int(r["start_ms"])]
        if blank_ranges:
            expr = "+".join(f"between(t,{a / 1000:.3f},{b / 1000:.3f})" for a, b in blank_ranges)
            fp.append(f"{cur}drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='{expr}'[vblank]")
            cur = "[vblank]"

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
            raw_text = ov.get("text") or ""
            if not raw_text.strip():
                continue
            font_path = _find_font_path(ov.get("font") or "roboto")
            if not font_path:
                continue
            # Sizes are px at 1080 wide (as the editor draws them): scaled to this export's width
            sz   = max(8, int(round((ov.get("size") or 48) * out_w / 1080)))
            hx   = (ov.get("color") or "#ffffff").lstrip("#")
            r, g, b2 = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
            centered = ov.get("x") is None
            # A black outline keeps it readable on any background (e.g. a white wall)
            outline = f":borderw={max(2, round(out_w * 4 / 1080))}:bordercolor=black" if centered else ""
            sy = int((ov.get("y") or 0.1) * out_h)
            t0 = ov.get("start_ms", 0) / 1000.0
            t1 = ov.get("end_ms", clip_dur_ms) / 1000.0
            enable = f"between(t,{t0:.3f},{t1:.3f})"

            box_w = float(ov.get("w") or 0) if not centered else 0.0
            box_h = float(ov.get("h") or 0) if not centered else 0.0
            # Turned on the preview (its rotate handle): drawn as a picture below, turned round its middle
            rotation = float(ov.get("rotation") or 0) % 360 if not centered else 0.0
            if not emoji_art.has_emoji(raw_text) and "\n" not in raw_text and box_w <= 0 and box_h <= 0 and not rotation:
                # ── One line, no emoji: a single drawtext call, exactly as before ──
                sz_fit = _fit_font_size(raw_text, font_path, sz, int(out_w * 0.9)) if centered else sz
                sx = "(w-text_w)/2" if centered else int((ov.get("x") or 0.1) * out_w)
                olbl = f"[vdt{oi}]"
                fp.append(
                    f"{cur}drawtext=fontfile={_filter_path(font_path)}:{_text_opt(raw_text)}:fontsize={sz_fit}"
                    f":fontcolor=0x{r:02X}{g:02X}{b2:02X}:x={sx}:y={sy}"
                    f":shadowx=2:shadowy=2:shadowcolor=black@0.7{outline}"
                    f":enable='{enable}'{olbl}"
                )
                cur = olbl
                continue

            # ── Several lines (Enter, or wrapped in its box) or emoji: each line is drawn in
            # pieces — drawtext for the plain-text runs, emoji composited as images — on one
            # baseline per line, lines TEXT_LINE_EM apart, as the editor draws them.
            # Emoji are treated as sz×sz boxes (Twemoji is drawn square). ──
            from PIL import ImageFont
            if box_w > 0:
                # Wrapped as the editor wraps it: the same rule, measured with the same font file
                wrap_font = ImageFont.truetype(font_path, sz)
                def _w(s: str) -> float:
                    return sum(sz * emoji_art.ADVANCE_EM if is_e else wrap_font.getlength(t) for is_e, t in emoji_art.split_emoji(s))
                text_lines = _wrap_text_lines(raw_text, box_w * out_w, _w)
            else:
                text_lines = raw_text.split("\n")
            line_runs = [emoji_art.split_emoji(ln) for ln in text_lines]

            def _measure(size: int) -> tuple[list[list[float]], float]:
                pil_font = ImageFont.truetype(font_path, size)
                # An emoji takes the Chai Emoji font's room (as the editor's preview draws it)
                widths = [[size * emoji_art.ADVANCE_EM if is_e else pil_font.getlength(s) for is_e, s in runs] for runs in line_runs]
                return widths, max((sum(ws) for ws in widths), default=0.0)

            widths, max_w = _measure(sz)
            if centered and max_w > out_w * 0.9 and max_w > 0:
                sz = max(28, int(sz * (out_w * 0.9) / max_w))
                widths, max_w = _measure(sz)

            line_h = sz * _TEXT_LINE_EM
            pil_font = ImageFont.truetype(font_path, sz)
            sx0 = int((ov.get("x") or 0.1) * out_w)
            # One distance from a line's top to its baseline for every line, so the lines are evenly spaced
            above = max((_line_baseline(pil_font, runs, sz) for runs in line_runs), default=0)
            if rotation:
                # The block the editor turns: as wide as its box (or its widest line), as tall as its
                # box (or its lines); the lines sit in its middle, up and down
                block_w = max(max_w, box_w * out_w)
                block_h = max(len(line_runs) * line_h, box_h * out_h)
                shift_y = max(0.0, (box_h * out_h - len(line_runs) * line_h) / 2)
                png, _pad = _rotated_text_png(line_runs, widths, font_path, sz, (r, g, b2), rotation,
                                              block_w, block_h, shift_y, line_h, above)
                inputs += ["-i", png]
                src_lbl = f"[{next_input}:v]"
                next_input += 1
                cx, cy = sx0 + block_w / 2, sy + block_h / 2
                olbl = f"[vrt{oi}]"
                fp.append(f"{src_lbl}format=rgba[vrt{oi}s]")
                fp.append(f"{cur}[vrt{oi}s]overlay=x={cx:.1f}-overlay_w/2:y={cy:.1f}-overlay_h/2:enable='{enable}'{olbl}")
                cur = olbl
                continue
            # A box with a height: the lines sit in its middle (as the editor draws them)
            sy = sy + max(0.0, (box_h * out_h - len(line_runs) * line_h) / 2)
            ridx = 0
            for li, runs in enumerate(line_runs):
                line_w = sum(widths[li])
                cursor = (out_w - line_w) / 2 if centered else sx0
                base = round(sy + li * line_h + above)
                for (is_emoji, s), w in zip(runs, widths[li]):
                    if not s:
                        continue
                    if is_emoji:
                        png = emoji_art.emoji_png(s, int(sz))
                        if png:
                            inputs += ["-i", png]
                            src_lbl = f"[{next_input}:v]"
                            next_input += 1
                            slbl = f"[vemo{oi}_{ridx}s]"
                            olbl = f"[vemo{oi}_{ridx}]"
                            fp.append(f"{src_lbl}format=rgba{slbl}")
                            fp.append(f"{cur}{slbl}overlay=x={int(cursor + sz * emoji_art.SIDE_EM)}"
                                      f":y={int(base - sz * emoji_art.ABOVE_EM)}:enable='{enable}'{olbl}")
                            cur = olbl
                            ridx += 1
                        # Reserve the width either way so trailing text doesn't overlap an emoji
                        # the font has no drawing for
                    else:
                        olbl = f"[vdt{oi}_{ridx}]"
                        fp.append(
                            f"{cur}drawtext=fontfile={_filter_path(font_path)}:{_text_opt(s)}:fontsize={sz}"
                            f":fontcolor=0x{r:02X}{g:02X}{b2:02X}:x={int(cursor)}:y_align=baseline:y={int(base)}"
                            f":shadowx=2:shadowy=2:shadowcolor=black@0.7{outline}"
                            f":enable='{enable}'{olbl}"
                        )
                        cur = olbl
                        ridx += 1
                    cursor += w

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
            # The whole photo inside its box, in its own shape (the preview shows it so: object-fit
            # contain), in the box's middle; turned round that middle when it was rotated
            rot = float(ov.get("rotation") or 0) % 360
            turn = ""
            if rot:
                rad = f"{rot * 3.141592653589793 / 180:.6f}"
                turn = f",format=rgba,rotate={rad}:c=none:ow=rotw({rad}):oh=roth({rad})"
            fp.append(f"{img_lbl}scale={iw}:{ih}:force_original_aspect_ratio=decrease:flags=lanczos{turn}{slbl}")
            fp.append(
                f"{cur}{slbl}overlay=x={ix + iw / 2:.1f}-overlay_w/2:y={iy + ih / 2:.1f}-overlay_h/2"
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
                    f"{cur}drawtext=fontfile={_filter_path(wm_font)}:text='{wm_text}'"
                    f":fontsize={max(24, out_w // 36)}:fontcolor=white@0.55"
                    f":x=w-tw-{max(16, out_w // 60)}:y=h-th-{max(16, out_h // 120)}"
                    f":shadowx=1:shadowy=1:shadowcolor=black@0.5[vwm]"
                )
                cur = "[vwm]"

        # Rename final label to [vout]
        fp.append(f"{cur}copy[vout]")

        # ── Assemble and run ───────────────────────────────────────────────────
        # The graph goes in a file: clips with many view changes build graphs larger than a
        # single command-line argument may be (128 KB on Linux, 32 KB total on Windows)
        fc_str = ";".join(fp)
        fc_path = os.path.join(tmp, "graph.txt")
        with open(fc_path, "w", encoding="utf-8") as f:
            f.write(fc_str)
        cmd = inputs + [
            "-/filter_complex", fc_path,
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
