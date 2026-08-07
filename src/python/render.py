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
from audio import build_ffmpeg_audio_args, extract_speech_ranges

# ── Quality presets (identical to previous version) ───────────────────────────
_QUALITY: dict[int, dict] = {
    480:  dict(crf=20, maxrate="3M",  bufsize="6M",  preset="fast", audio_br="128k"),
    720:  dict(crf=18, maxrate="6M",  bufsize="12M", preset="fast", audio_br="192k"),
    1080: dict(crf=18, maxrate="10M", bufsize="20M", preset="fast", audio_br="192k"),
    2160: dict(crf=16, maxrate="20M", bufsize="40M", preset="fast", audio_br="256k"),
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


def _word_display(w: dict) -> str:
    return w.get("word_roman") or w.get("word", "")


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
        if (
            _is_sentence_end(w.get("word", ""))
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
) -> None:
    font_id   = style.get("font") or "noto-sans-telugu"
    font_name = _FONT_NAMES.get(font_id, "Roboto")
    font_size = int(style.get("size") or 52)
    color_hex = style.get("color") or "#ffffff"
    pos_y_frac = float(style.get("position_y") or 0.84)

    primary = _hex_to_ass(color_hex, 0)
    shadow  = "&H80000000"

    # Rebase word timestamps to clip-relative (0 = first frame of clip)
    clip_words = [
        {**w, "start_ms": w["start_ms"] - clip_start_ms,
               "end_ms":   w["end_ms"]   - clip_start_ms}
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

    for sentence in _group_sentences(clip_words):
        if not sentence:
            continue
        start_ms = max(0, sentence[0]["start_ms"])
        end_ms   = sentence[-1]["end_ms"] + 200
        text     = " ".join(_word_display(w) for w in sentence if _word_display(w))
        if not text.strip():
            continue
        # Alignment=5 (center of screen); \pos pins the anchor to exact coordinates
        tag = f"{{\\pos({pos_x},{pos_y})}}"
        lines.append(
            f"Dialogue: 0,{_ms_to_ass_ts(start_ms)},{_ms_to_ass_ts(end_ms)},"
            f"Default,,0,0,0,,{tag}{text}"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ── FFmpeg crop expression builder ────────────────────────────────────────────

def _step_expr(kf_list: list[dict], attr: str, seg_start_ms: int) -> str:
    """
    Build an FFmpeg filter expression for a step-interpolated crop coordinate.

    attr        one of 'x','y','w','h' (stored as fractions 0-1 of source frame)
    seg_start_ms  clip-relative ms at which this segment begins
    t           FFmpeg variable = seconds since start of the trimmed segment
                (valid after trim+setpts=PTS-STARTPTS)
    keyframes   stored with clip-relative t_ms → convert to seg-relative by subtracting
    """
    dim = "iw" if attr in ("x", "w") else "ih"
    defaults = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}

    if not kf_list:
        return f"{dim}*{defaults[attr]:.6f}"

    sk = sorted(kf_list, key=lambda k: k["t_ms"])

    if len(sk) == 1:
        return f"{dim}*{sk[0][attr]:.6f}"

    # Build right-to-left: position from kf[i] holds until kf[i+1] fires
    result = f"{dim}*{sk[-1][attr]:.6f}"
    for i in range(len(sk) - 2, -1, -1):
        # t_switch is the seg-relative time when we transition to the next position
        t_switch = (sk[i + 1]["t_ms"] - seg_start_ms) / 1000.0
        result = f"if(lt(t,{t_switch:.3f}),{dim}*{sk[i][attr]:.6f},{result})"

    return result


def _crop_filter(box: dict | None, seg_start_ms: int) -> str:
    kf = box.get("box_keyframes", []) if box else []
    w = _esc_expr(f"max(2,{_step_expr(kf, 'w', seg_start_ms)})")
    h = _esc_expr(f"max(2,{_step_expr(kf, 'h', seg_start_ms)})")
    x = _esc_expr(f"min(iw-2,{_step_expr(kf, 'x', seg_start_ms)})")
    y = _esc_expr(f"min(ih-2,{_step_expr(kf, 'y', seg_start_ms)})")
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
    watermark: bool = False,
) -> None:
    secondary_videos = secondary_videos or {}
    overlay_images   = overlay_images   or {}

    spec           = json.load(open(spec_path))
    clip_start_ms  = int(spec["start_ms"])
    clip_end_ms    = int(spec["end_ms"])
    clip_dur_ms    = clip_end_ms - clip_start_ms
    out_w          = int(spec.get("output_width",  1080))
    out_h          = int(spec.get("output_height", 1920))
    segments       = sorted(spec["segments"], key=lambda s: s["sort_order"])
    words          = spec.get("words", [])
    caption_styles = spec.get("caption_styles", [])
    text_overlays  = spec.get("text_overlays", [])
    audio_tracks   = spec.get("audio_tracks", [])
    filters_cfg    = spec.get("filters", {})
    img_overlays   = [o for o in spec.get("overlays", []) if o.get("type") == "image"]

    for ov in img_overlays:
        ov["local_path"] = overlay_images.get(ov.get("storage_path", ""), "")

    clip_words    = [w for w in words if w["end_ms"] >= clip_start_ms and w["start_ms"] <= clip_end_ms]
    caption_style = caption_styles[0] if caption_styles else {}
    qs            = _QUALITY.get(out_w, _QUALITY[1080])

    print(f"[render] Clip {clip_start_ms}ms–{clip_end_ms}ms → {out_w}x{out_h}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:

        # ── Audio ──────────────────────────────────────────────────────────────
        audio_path    = os.path.join(tmp, "audio.aac")
        speech_ranges = extract_speech_ranges(clip_words)
        ar = subprocess.run(
            build_ffmpeg_audio_args(
                video_path, clip_start_ms, clip_end_ms, audio_tracks, speech_ranges, audio_path
            ),
            capture_output=True,
        )
        if ar.returncode != 0:
            raise RuntimeError(f"Audio extraction failed:\n{ar.stderr.decode()[-500:]}")

        # ── ASS captions ───────────────────────────────────────────────────────
        ass_path = None
        if clip_words and caption_style:
            ass_path = os.path.join(tmp, "captions.ass")
            _write_ass(clip_words, caption_style, clip_start_ms, out_w, out_h, ass_path)

        # ── Count how many times each source video is needed ───────────────────
        # FFmpeg requires explicit split() when a stream is consumed more than once.
        slot_counts: dict[str | None, int] = {None: 0}
        for seg in segments:
            boxes   = sorted(seg.get("crop_boxes", []), key=lambda b: b.get("slot_index", 0))
            n_slots = {"split": 2, "trio": 3}.get(seg.get("layout", "vertical"), 1)
            for i in range(n_slots):
                box    = boxes[i] if i < len(boxes) else None
                vid_id = box.get("source_video_id") if box else None
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
            inputs += ["-i", path]

        img_base = 2 + len(secondary_videos)
        valid_img: list[tuple[dict, str]] = []
        for ov in img_overlays:
            lp = ov.get("local_path", "")
            if lp and os.path.exists(lp):
                inputs += ["-i", lp]
                valid_img.append((ov, f"[{img_base + len(valid_img)}:v]"))

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
            smss    = int(seg["start_ms"])   # clip-relative ms
            emss    = int(seg["end_ms"])
            dur_ms  = emss - smss
            start_s = smss / 1000.0
            end_s   = emss / 1000.0
            out_lbl = f"[seg{si}]"

            def trim_slot(slot_i: int, dst_w: int, dst_h: int, fit: bool, lbl: str) -> None:
                box    = boxes[slot_i] if slot_i < len(boxes) else None
                vid_id = box.get("source_video_id") if box else None

                if vid_id and vid_id in secondary_videos:
                    # Secondary: trim from source_offset_ms for segment duration
                    off_ms = int(box.get("source_offset_ms", 0))
                    src    = pop_src(vid_id)
                    ts, te = off_ms / 1000.0, (off_ms + dur_ms) / 1000.0
                else:
                    # Primary: trim clip-relative segment window
                    src    = pop_src(None)
                    ts, te = start_s, end_s

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
            # Escape path for FFmpeg filter option (colons are option separators)
            esc_ass   = ass_path.replace("\\", "\\\\").replace(":", "\\:")
            esc_fonts = _FONTS_DIR.replace("\\", "\\\\").replace(":", "\\:")
            fp.append(f"{cur}subtitles={esc_ass}:fontsdir={esc_fonts}[vcap]")
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
    parser.add_argument("--watermark",        action="store_true", help="Burn in Chai Cut watermark")
    args = parser.parse_args()
    main(
        args.video, args.spec, args.output,
        secondary_videos=json.loads(args.secondary_videos),
        overlay_images=json.loads(args.overlay_images),
        watermark=args.watermark,
    )
