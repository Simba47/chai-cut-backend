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
    sorted_segs = sorted(segments, key=lambda s: s.get("sort_order", 0))

    # Check whether any B-roll segment has an available secondary file
    has_broll_audio = any(
        (box := (s.get("crop_boxes") or [{}])[0]).get("source_video_id") in secondary_videos
        for s in sorted_segs
        if (s.get("crop_boxes") or [{}])[0].get("source_video_id")
    )

    if not has_broll_audio:
        # No inserts with available audio — use simple single-clip extraction
        return build_ffmpeg_audio_args(
            source_video, clip_start_ms, clip_end_ms,
            audio_tracks, speech_ranges, output_audio,
        )

    inputs: list[str] = ["ffmpeg", "-y"]
    seg_labels: list[str] = []
    n_inputs = 0

    for seg in sorted_segs:
        boxes = seg.get("crop_boxes") or []
        box = boxes[0] if boxes else {}
        vid_id = box.get("source_video_id")
        dur_ms = int(seg["end_ms"]) - int(seg["start_ms"])
        dur_s = dur_ms / 1000.0

        if vid_id and vid_id in secondary_videos:
            # B-roll INSERT: pull audio from the secondary file
            off_s = int(box.get("source_offset_ms") or 0) / 1000.0
            inputs += ["-ss", f"{off_s:.3f}", "-t", f"{dur_s:.3f}", "-i", secondary_videos[vid_id]]
            seg_labels.append(f"[{n_inputs}:a]")
            n_inputs += 1
        else:
            # Main segment: resolve clip-relative video position
            vid_off_ms = int(seg["video_offset_ms"]) if seg.get("video_offset_ms") is not None else int(seg["start_ms"])
            abs_start_s = (clip_start_ms + vid_off_ms) / 1000.0
            inputs += ["-ss", f"{abs_start_s:.3f}", "-t", f"{dur_s:.3f}", "-i", source_video]
            seg_labels.append(f"[{n_inputs}:a]")
            n_inputs += 1

    # Concatenate all segment audio pieces
    concat_in = "".join(seg_labels)
    fp: list[str] = [f"{concat_in}concat=n={n_inputs}:v=0:a=1[speech]"]

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
        track_idx = n_inputs + i
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
