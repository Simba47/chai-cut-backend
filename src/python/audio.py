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
        # No background music — extract original speech audio directly
        return [
            "ffmpeg", "-y",
            "-ss", str(start_s),
            "-t", str(duration_s),
            "-i", source_video,
            "-vn",
            "-acodec", "copy",
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
