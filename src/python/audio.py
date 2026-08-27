"""
Audio mixing utilities.

Handles:
  - Extracting speech audio from source video (preserving original codec)
  - Mixing background music with optional speech ducking
  - Assembling final audio via ffmpeg
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


def build_ffmpeg_audio_args(
    source_video: str,
    clip_start_ms: int,
    clip_end_ms: int,
    audio_tracks: list[dict],
    speech_ranges: list[tuple[int, int]],
    output_audio: str,
) -> list[str]:
    """
    Build an ffmpeg command that produces a mixed audio file.

    speech_ranges: list of (start_ms, end_ms) of speech segments used for ducking.
    Returns the ffmpeg argv list.
    """
    duration_s = (clip_end_ms - clip_start_ms) / 1000.0
    start_s = clip_start_ms / 1000.0

    if not audio_tracks:
        # Re-encode to AAC for sample-accurate seeking (acodec copy snaps to
        # nearest audio keyframe, causing audio to arrive before the video).
        return [
            "ffmpeg", "-y",
            "-ss", str(start_s),
            "-t", str(duration_s),
            "-i", source_video,
            "-vn",
            "-acodec", "aac", "-b:a", "192k",
            output_audio,
        ]

    # We have background tracks — need to mix with volume automation
    inputs = [
        "-ss", str(start_s),
        "-t", str(duration_s),
        "-i", source_video,
    ]

    filter_parts: list[str] = []
    # Speech audio from source: [0:a]
    speech_label = "[speech]"
    filter_parts.append(f"[0:a]volume=1.0{speech_label}")

    music_labels: list[str] = []
    for i, track in enumerate(audio_tracks):
        track_idx = i + 1
        inputs += [
            "-ss", str(track.get("start_ms", 0) / 1000.0),
            "-i", track["storage_path"],
        ]
        base_vol = float(track.get("volume", 0.5))
        label = f"[music{i}]"

        if track.get("duck_under_speech") and speech_ranges:
            # Build volume automation using ffmpeg volume filter with enable ranges
            duck_vol = base_vol * 0.15
            enable_expr = "+".join(
                f"between(t,{s/1000:.3f},{e/1000:.3f})" for s, e in speech_ranges
            )
            vol_expr = (
                f"if({enable_expr},{duck_vol},{base_vol})"
            )
            filter_parts.append(f"[{track_idx}:a]volume='{vol_expr}'{label}")
        else:
            filter_parts.append(f"[{track_idx}:a]volume={base_vol}{label}")

        music_labels.append(label)

    # Mix all
    all_labels = [speech_label] + music_labels
    mix_inputs = "".join(all_labels)
    n = len(all_labels)
    filter_parts.append(f"{mix_inputs}amix=inputs={n}:duration=first:dropout_transition=0[aout]")

    return [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[aout]",
        "-acodec", "aac",
        "-b:a", "192k",
        output_audio,
    ]


def build_segment_audio_args(
    source_video: str,
    clip_start_ms: int,
    clip_end_ms: int,
    segments: list[dict],
    secondary_videos: dict[str, str],
    audio_tracks: list[dict],
    speech_ranges: list[tuple[int, int]],
    output_audio: str,
) -> list[str]:
    """
    Build audio that correctly handles B-roll INSERT segments.

    For each segment in timeline order:
      - B-roll (has source_video_id): audio taken from the secondary video
        starting at source_offset_ms for the segment duration.
      - Main segment: audio taken from source_video at the clip-relative
        video_offset_ms position (or start_ms when video_offset_ms is absent).

    The per-segment pieces are concatenated to produce a single audio track
    whose length equals the full extended timeline (B-roll included).
    Falls back to the simple single-clip extraction when there are no B-roll
    segments with available secondary video files.
    """
    sorted_segs = sorted(segments, key=lambda s: (s.get("sort_order", 0), s.get("start_ms", 0)))

    # Check whether any B-roll segment has an available secondary file
    has_broll_audio = any(
        (s.get("crop_boxes") or [{}])[0].get("source_video_id") in secondary_videos
        for s in sorted_segs
        if (s.get("crop_boxes") or [{}])[0].get("source_video_id")
    )

    if not has_broll_audio:
        return build_ffmpeg_audio_args(
            source_video, clip_start_ms, clip_end_ms,
            audio_tracks, speech_ranges, output_audio,
        )

    # ── Accurate sync approach: use atrim in filter_complex ──────────────────
    # Opening the pre-seeked source once (same as the video pipeline) and using
    # atrim=start=X:end=Y,asetpts=PTS-STARTPTS is sample-accurate — no frame-
    # alignment drift from repeated -ss -t -i pairs.

    clip_start_s = clip_start_ms / 1000.0

    # Collect which B-roll video IDs are actually used and count their uses
    broll_uses: dict[str, int] = {}
    main_uses = 0
    for seg in sorted_segs:
        box = (seg.get("crop_boxes") or [{}])[0]
        vid_id = box.get("source_video_id") if box else None
        if vid_id and vid_id in secondary_videos:
            broll_uses[vid_id] = broll_uses.get(vid_id, 0) + 1
        else:
            main_uses += 1

    # Build inputs: main source pre-seeked, then each needed B-roll file
    inputs: list[str] = [
        "ffmpeg", "-y",
        "-ss", f"{clip_start_s:.3f}",
        "-i", source_video,   # index 0
    ]
    broll_idx: dict[str, int] = {}
    for vid_id in broll_uses:
        broll_idx[vid_id] = len(broll_idx) + 1
        inputs += ["-i", secondary_videos[vid_id]]

    # Build filter_complex
    fp: list[str] = []

    # Pre-split streams that appear more than once
    main_pool: list[str] = []
    if main_uses > 1:
        lbls = [f"[ma{i}]" for i in range(main_uses)]
        fp.append(f"[0:a]asplit={main_uses}{''.join(lbls)}")
        main_pool = lbls
    elif main_uses == 1:
        main_pool = ["[0:a]"]

    broll_pools: dict[str, list[str]] = {}
    for vid_id, count in broll_uses.items():
        idx = broll_idx[vid_id]
        safe = vid_id.replace("-", "_").replace(":", "_")
        if count > 1:
            lbls = [f"[br{safe}{i}]" for i in range(count)]
            fp.append(f"[{idx}:a]asplit={count}{''.join(lbls)}")
            broll_pools[vid_id] = lbls
        else:
            broll_pools[vid_id] = [f"[{idx}:a]"]

    main_it = iter(main_pool)
    broll_its: dict[str, "Iterator[str]"] = {v: iter(p) for v, p in broll_pools.items()}

    norm_labels: list[str] = []
    n_segs = len(sorted_segs)
    for i, seg in enumerate(sorted_segs):
        box = (seg.get("crop_boxes") or [{}])[0]
        vid_id = box.get("source_video_id") if box else None
        dur_ms = int(seg["end_ms"]) - int(seg["start_ms"])
        dur_s  = dur_ms / 1000.0
        raw_lbl  = f"[sa{i}]"
        norm_lbl = f"[san{i}]"

        if vid_id and vid_id in broll_pools:
            off_s  = int(box.get("source_offset_ms") or 0) / 1000.0
            end_s  = off_s + dur_s
            src    = next(broll_its[vid_id])
            fp.append(f"{src}atrim=start={off_s:.3f}:end={end_s:.3f},asetpts=PTS-STARTPTS{raw_lbl}")
        else:
            vid_off_ms   = int(seg["video_offset_ms"]) if seg.get("video_offset_ms") is not None else int(seg["start_ms"])
            trim_start   = vid_off_ms / 1000.0   # relative to pre-seeked clip_start
            trim_end     = trim_start + dur_s
            src          = next(main_it)
            fp.append(f"{src}atrim=start={trim_start:.3f}:end={trim_end:.3f},asetpts=PTS-STARTPTS{raw_lbl}")

        # Normalise each piece to stereo 48 kHz to handle mixed formats (e.g. B-roll at 44.1 kHz)
        fp.append(f"{raw_lbl}aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo{norm_lbl}")
        norm_labels.append(norm_lbl)

    concat_in = "".join(norm_labels)
    fp.append(f"{concat_in}concat=n={n_segs}:v=0:a=1[speech]")

    n_broll_inputs = len(broll_idx)

    if not audio_tracks:
        return [
            *inputs,
            "-filter_complex", ";".join(fp),
            "-map", "[speech]",
            "-acodec", "aac", "-b:a", "192k",
            output_audio,
        ]

    # Mix concatenated speech with background music tracks
    music_labels: list[str] = []
    for i, track in enumerate(audio_tracks):
        track_idx = 1 + n_broll_inputs + i
        inputs += ["-ss", str(track.get("start_ms", 0) / 1000.0), "-i", track["storage_path"]]
        base_vol = float(track.get("volume", 0.5))
        label = f"[music{i}]"
        if track.get("duck_under_speech") and speech_ranges:
            duck_vol = base_vol * 0.15
            enable_expr = "+".join(
                f"between(t,{s/1000:.3f},{e/1000:.3f})" for s, e in speech_ranges
            )
            fp.append(f"[{track_idx}:a]volume='if({enable_expr},{duck_vol},{base_vol})'{label}")
        else:
            fp.append(f"[{track_idx}:a]volume={base_vol}{label}")
        music_labels.append(label)

    all_labels = ["[speech]"] + music_labels
    n = len(all_labels)
    fp.append(f"{''.join(all_labels)}amix=inputs={n}:duration=first:dropout_transition=0[aout]")

    return [
        *inputs,
        "-filter_complex", ";".join(fp),
        "-map", "[aout]",
        "-acodec", "aac", "-b:a", "192k",
        output_audio,
    ]


def extract_speech_ranges(words: list[dict]) -> list[tuple[int, int]]:
    """
    Merge consecutive word timestamps into contiguous speech segments
    (gap < 500ms merges into one range).
    """
    if not words:
        return []

    sorted_words = sorted(words, key=lambda w: w["start_ms"])
    ranges: list[tuple[int, int]] = []
    start = sorted_words[0]["start_ms"]
    end = sorted_words[0]["end_ms"]

    for w in sorted_words[1:]:
        if w["start_ms"] - end < 500:
            end = max(end, w["end_ms"])
        else:
            ranges.append((start, end))
            start = w["start_ms"]
            end = w["end_ms"]

    ranges.append((start, end))
    return ranges
