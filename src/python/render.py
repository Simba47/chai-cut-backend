#!/usr/bin/env python3
"""
Single-pass video compositor with multi-source support and image overlays.

Usage:
  python3 render.py --video <source.mp4> --spec <spec.json> --output <out.mp4>
  --secondary-videos <json_map>   optional JSON mapping video_id -> local_path
  --overlay-images   <json_map>   optional JSON mapping storage_path -> local_path
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from interpolation import get_box_position_at
from layouts import compose
from captions import burn_captions
from audio import build_ffmpeg_audio_args, extract_speech_ranges


def apply_filters(frame: np.ndarray, brightness: float, contrast: float, saturation: float) -> np.ndarray:
    b = brightness / 100.0
    c = contrast / 100.0
    s = saturation / 100.0
    frame = frame.astype(np.float32)
    frame = frame * b
    frame = (frame - 128.0) * c + 128.0
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    if s != 1.0:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s, 0, 255)
        frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return frame


def burn_text_overlay(frame: np.ndarray, t_ms: int, overlays: list[dict]) -> np.ndarray:
    from captions import _find_font, _hex_to_rgb, cv2_to_pil, pil_to_cv2
    from PIL import ImageDraw

    active = [o for o in overlays if o["start_ms"] <= t_ms < o["end_ms"]]
    if not active:
        return frame

    pil = Image.fromarray(cv2_to_pil(frame))
    draw = ImageDraw.Draw(pil)
    w, h = pil.size

    for o in active:
        font_id = o.get("font") or "roboto"
        size = int(o.get("size") or 48)
        color = _hex_to_rgb(o.get("color") or "#ffffff")
        font = _find_font(font_id, size)
        x = int((o.get("x") or 0.1) * w)
        y = int((o.get("y") or 0.1) * h)
        text = o.get("text") or ""
        draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 180))
        draw.text((x, y), text, font=font, fill=color)

    return pil_to_cv2(np.array(pil))


def burn_image_overlays(
    frame: np.ndarray,
    t_ms: int,
    overlays: list[dict],
    image_cache: dict[str, np.ndarray],
) -> np.ndarray:
    """Blend image overlays onto the frame."""
    active = [o for o in overlays if o.get("type") == "image"
              and o["start_ms"] <= t_ms < o["end_ms"]
              and o.get("local_path")]
    if not active:
        return frame

    out_h, out_w = frame.shape[:2]
    result = frame.copy()

    for o in sorted(active, key=lambda x: x.get("z_index", 1)):
        path = o["local_path"]
        if path not in image_cache:
            try:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                image_cache[path] = img
            except Exception:
                image_cache[path] = None

        img = image_cache.get(path)
        if img is None:
            continue

        x1 = int(o["x"] * out_w)
        y1 = int(o["y"] * out_h)
        x2 = int((o["x"] + o["w"]) * out_w)
        y2 = int((o["y"] + o["h"]) * out_h)
        dw, dh = max(1, x2 - x1), max(1, y2 - y1)

        resized = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_LINEAR)

        if resized.shape[2] == 4:
            # Alpha blending
            alpha = resized[:, :, 3:4].astype(np.float32) / 255.0
            rgb = resized[:, :, :3].astype(np.float32)
            roi = result[y1:y1+dh, x1:x1+dw].astype(np.float32)
            blended = rgb * alpha + roi * (1.0 - alpha)
            result[y1:y1+dh, x1:x1+dw] = blended.clip(0, 255).astype(np.uint8)
        else:
            result[y1:y1+dh, x1:x1+dw] = resized[:, :, :3]

    return result


def find_active_segment(segments: list[dict], t_ms: int) -> dict | None:
    for seg in segments:
        if seg["start_ms"] <= t_ms < seg["end_ms"]:
            return seg
    return None


def get_frame_from_cap(cap: cv2.VideoCapture, t_ms: int) -> np.ndarray | None:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        return None
    frame_idx = int((t_ms / 1000.0) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None


def main(video_path: str, spec_path: str, output_path: str,
         secondary_videos: dict[str, str] | None = None,
         overlay_images: dict[str, str] | None = None) -> None:
    with open(spec_path) as f:
        spec = json.load(f)

    clip_start_ms: int = spec["start_ms"]
    clip_end_ms: int   = spec["end_ms"]
    out_w: int         = spec.get("output_width", 1080)
    out_h: int         = spec.get("output_height", 1920)

    # Per-quality encode settings — maxrate kept conservative so output stays under 45MB
    # for typical 1–2 min clips (Supabase free plan has a 50MB object limit).
    # Higher bitrates compensate for the double-encode (OpenCV decode → Python
    # crop/resize → raw pipe → ffmpeg encode).  fast preset reduces blocking artefacts.
    _quality_settings = {
        480:  dict(crf=20, maxrate="3M",   bufsize="6M",   preset="fast", audio_br="128k"),
        720:  dict(crf=18, maxrate="6M",   bufsize="12M",  preset="fast", audio_br="192k"),
        1080: dict(crf=18, maxrate="10M",  bufsize="20M",  preset="fast", audio_br="192k"),
        2160: dict(crf=16, maxrate="20M",  bufsize="40M",  preset="fast", audio_br="256k"),
    }
    _qs = _quality_settings.get(out_w, _quality_settings[1080])
    segments: list     = sorted(spec["segments"], key=lambda s: s["sort_order"])
    words: list        = spec.get("words", [])
    caption_styles     = spec.get("caption_styles", [])
    text_overlays      = spec.get("text_overlays", [])
    audio_tracks       = spec.get("audio_tracks", [])
    filters            = spec.get("filters", {})
    brightness         = float(filters.get("brightness", 100))
    contrast           = float(filters.get("contrast", 100))
    saturation         = float(filters.get("saturation", 100))
    image_overlays_spec = spec.get("overlays", [])

    # Attach local paths to image overlay specs
    for ov in image_overlays_spec:
        if ov.get("type") == "image" and ov.get("storage_path") and overlay_images:
            ov["local_path"] = overlay_images.get(ov["storage_path"], "")

    clip_words = [
        w for w in words
        if w["end_ms"] >= clip_start_ms and w["start_ms"] <= clip_end_ms
    ]
    speech_ranges = extract_speech_ranges(clip_words)
    caption_style = caption_styles[0] if caption_styles else {}

    # Open primary video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[render] Source: {src_w}x{src_h} @ {fps:.2f}fps, {total_frames} frames", flush=True)
    print(f"[render] Clip: {clip_start_ms}ms – {clip_end_ms}ms → {out_w}x{out_h}", flush=True)

    # Open secondary video captures
    secondary_caps: dict[str, cv2.VideoCapture] = {}
    if secondary_videos:
        for vid_id, path in secondary_videos.items():
            c = cv2.VideoCapture(path)
            if c.isOpened():
                secondary_caps[vid_id] = c
                print(f"[render] Opened secondary video {vid_id}: {path}", flush=True)

    image_cache: dict[str, np.ndarray | None] = {}

    start_frame = int((clip_start_ms / 1000.0) * fps)
    end_frame   = int((clip_end_ms   / 1000.0) * fps)
    clip_frames = end_frame - start_frame

    import os
    tmp_audio = tempfile.NamedTemporaryFile(suffix=".aac", delete=False)
    tmp_audio.close()
    audio_args = build_ffmpeg_audio_args(
        video_path, clip_start_ms, clip_end_ms, audio_tracks, speech_ranges, tmp_audio.name
    )
    audio_proc = subprocess.run(audio_args, capture_output=True)
    if audio_proc.returncode != 0:
        print("[render] Audio ffmpeg stderr:", audio_proc.stderr.decode(), flush=True)
        raise RuntimeError("Audio extraction failed")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}",
        "-r", str(fps), "-i", "pipe:0",
        "-i", tmp_audio.name,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "main", "-level", "4.0",
        "-crf", str(_qs["crf"]), "-preset", _qs["preset"],
        "-maxrate", _qs["maxrate"], "-bufsize", _qs["bufsize"],
        "-c:a", "aac", "-b:a", _qs["audio_br"],
        "-shortest", output_path,
    ]

    ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    canvas_black = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    processed = 0

    try:
        for frame_idx in range(clip_frames):
            ret, primary_frame = cap.read()
            if not ret:
                break

            abs_frame = start_frame + frame_idx
            t_ms = int((abs_frame / fps) * 1000)
            # Segment start/end and keyframe t_ms are stored relative to clip start;
            # caption word timestamps are stored absolute — use rel_t_ms only for
            # segment lookup and keyframe interpolation.
            rel_t_ms = t_ms - clip_start_ms

            seg = find_active_segment(segments, rel_t_ms)
            canvas = canvas_black.copy()

            if seg:
                layout = seg["layout"]
                boxes = sorted(seg.get("crop_boxes", []), key=lambda b: b["slot_index"])
                positions = []

                for box in boxes:
                    kf = box.get("box_keyframes", [])
                    pos = get_box_position_at(rel_t_ms, kf)
                    positions.append(pos)

                # Determine source frames per slot
                slot_frames: list[np.ndarray] = []
                for box in boxes:
                    src_vid_id = box.get("source_video_id")
                    if src_vid_id and src_vid_id in secondary_caps:
                        src_offset = box.get("source_offset_ms", 0)
                        elapsed = t_ms - seg["start_ms"]
                        src_t_ms = src_offset + elapsed
                        frame = get_frame_from_cap(secondary_caps[src_vid_id], src_t_ms)
                        slot_frames.append(frame if frame is not None else primary_frame)
                    else:
                        slot_frames.append(primary_frame)

                if positions:
                    canvas = compose_multi_source(layout, canvas, slot_frames, positions, out_w, out_h)
                else:
                    canvas = _cover_crop(primary_frame, out_w, out_h)
            else:
                canvas = _cover_crop(primary_frame, out_w, out_h)

            if brightness != 100 or contrast != 100 or saturation != 100:
                canvas = apply_filters(canvas, brightness, contrast, saturation)

            if caption_style and clip_words:
                canvas = burn_captions(canvas, t_ms, clip_words, caption_style)

            if text_overlays:
                canvas = burn_text_overlay(canvas, t_ms, text_overlays)

            if image_overlays_spec:
                canvas = burn_image_overlays(canvas, t_ms, image_overlays_spec, image_cache)

            assert ffmpeg.stdin is not None
            ffmpeg.stdin.write(canvas.tobytes())
            processed += 1

            if processed % 100 == 0:
                pct = processed / max(clip_frames, 1) * 100
                print(f"[render] {processed}/{clip_frames} frames ({pct:.1f}%)", flush=True)

    finally:
        cap.release()
        for c in secondary_caps.values():
            c.release()
        if ffmpeg.stdin:
            ffmpeg.stdin.close()

    _, ffmpeg_err = ffmpeg.communicate()
    if ffmpeg.returncode != 0:
        print("[render] ffmpeg stderr:", ffmpeg_err.decode(), flush=True)
        raise RuntimeError(f"ffmpeg encode failed (exit {ffmpeg.returncode})")

    try:
        os.unlink(tmp_audio.name)
    except Exception:
        pass

    print(f"[render] Done — {processed} frames → {output_path}", flush=True)


def _cover_crop(src: np.ndarray, dst_w: int, dst_h: int) -> np.ndarray:
    """Scale src to cover (dst_w, dst_h) maintaining aspect ratio, center-crop excess."""
    src_h, src_w = src.shape[:2]
    if src_w == 0 or src_h == 0:
        return np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale = max(dst_w / src_w, dst_h / src_h)
    sw = max(int(src_w * scale), 1)
    sh = max(int(src_h * scale), 1)
    scaled = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_LINEAR)
    x0 = (sw - dst_w) // 2
    y0 = (sh - dst_h) // 2
    return scaled[y0:y0 + dst_h, x0:x0 + dst_w]


def _crop(frame: np.ndarray, pos: dict) -> np.ndarray:
    h, w = frame.shape[:2]
    x1 = max(0, int(pos["x"] * w))
    y1 = max(0, int(pos["y"] * h))
    x2 = min(w, int((pos["x"] + pos["w"]) * w))
    y2 = min(h, int((pos["y"] + pos["h"]) * h))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return frame[y1:y2, x1:x2]


def compose_multi_source(
    layout: str,
    canvas: np.ndarray,
    slot_frames: list[np.ndarray],
    positions: list[dict],
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """Like layouts.compose but each slot can have its own source frame."""
    if layout in ("vertical", "spotlight", "centered", "horizontal"):
        frame = slot_frames[0] if slot_frames else canvas
        crop = _crop(frame, positions[0])
        return _cover_crop(crop, out_w, out_h)
    elif layout == "split":
        result = canvas.copy()
        slot_h = out_h // 2
        for i, (pos, frame) in enumerate(zip(positions[:2], slot_frames[:2])):
            crop = _crop(frame, pos)
            result[i * slot_h : (i + 1) * slot_h, :] = _cover_crop(crop, out_w, slot_h)
        return result
    elif layout == "trio":
        result = canvas.copy()
        top_h = int(out_h * 0.55)
        bot_h = out_h - top_h
        half_w = out_w // 2
        for i, (pos, frame) in enumerate(zip(positions[:3], slot_frames[:3])):
            crop = _crop(frame, pos)
            if i == 0:
                result[:top_h, :] = _cover_crop(crop, out_w, top_h)
            else:
                result[top_h:, (i - 1) * half_w : i * half_w] = _cover_crop(crop, half_w, bot_h)
        return result
    else:
        if slot_frames:
            return _cover_crop(_crop(slot_frames[0], positions[0]), out_w, out_h)
        return canvas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  required=True)
    parser.add_argument("--spec",   required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--secondary-videos", default="{}", help="JSON: video_id -> local_path")
    parser.add_argument("--overlay-images",   default="{}", help="JSON: storage_path -> local_path")
    args = parser.parse_args()
    main(
        args.video, args.spec, args.output,
        secondary_videos=json.loads(args.secondary_videos),
        overlay_images=json.loads(args.overlay_images),
    )
