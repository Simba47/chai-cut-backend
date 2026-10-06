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

from frames import is_frame, frame_audio_plan, frame_state, main_audio_volume


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


_audio_cache: dict[str, bool] = {}


def _has_audio(path: str) -> bool:
    """Whether a media file has a sound track (assume yes if ffprobe can't tell)."""
    if path not in _audio_cache:
        try:
            r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", path],
                               capture_output=True, text=True, timeout=30)
            _audio_cache[path] = r.returncode != 0 or bool(r.stdout.strip())
        except Exception:
            _audio_cache[path] = True
    return _audio_cache[path]


# Music: how long a fade-in / fade-out lasts, and how loud music stays while someone speaks
FADE_S = 0.5
DUCK_LEVEL = 0.15
# Every export is levelled to the loudness social apps play at (EBU R128 / -14 LUFS, peaks under -1 dB)
LOUDNESS_FILTER = "loudnorm=I=-14:TP=-1:LRA=11,aresample=48000"

_len_cache: dict[str, float | None] = {}


def _media_len(path: str) -> float | None:
    """Length of a media file in seconds (None if ffprobe can't tell)."""
    if path not in _len_cache:
        try:
            r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                               capture_output=True, text=True, timeout=60)
            _len_cache[path] = float(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else None
        except Exception:
            _len_cache[path] = None
    return _len_cache[path]


def _between(ranges_ms) -> str:
    return "+".join(f"between(t,{a / 1000:.3f},{b / 1000:.3f})" for a, b in ranges_ms)


def build_segment_audio_args(
    source_video: str,
    clip_start_ms: int,
    clip_end_ms: int,
    segments: list[dict],
    secondary_videos: dict[str, str],
    audio_tracks: list[dict],
    speech_ranges: list[tuple[int, int]],
    output_audio: str,
    mute_ranges: list[tuple[int, int]] | None = None,
    main_detached: bool = False,
    loudness: bool = True,
    main_volume: float = 1.0,
) -> list[str]:
    """
    The clip's whole soundtrack, in one FFmpeg run:

      1. The bed: what the picture plays, section by section — the main video's sound, a B-roll
         shot's own sound (at its volume), or a frame's mix. With the original sound detached
         (main_detached) the main video's part of the bed is silent: the detached bar plays it.
         Otherwise the main video plays at main_volume (the Music panel's "Original video sound").
      2. Muted sections (mute_ranges, clip-relative ms) silence the bed there — not the music.
      3. Each audio track is placed on the timeline: music from `offset_ms` into the song, starting
         at `start_ms`, stopping at `end_ms` (or where the song ends), at its volume, with optional
         fades and the dip under speech; an original-sound bar (kind "original") is the main
         video's sound from `offset_ms` (source time), placed the same way.
      4. Everything is mixed WITHOUT lowering each input (normalize=0, so the voice stays at 100%)
         and levelled to -14 LUFS for social apps (loudness=False skips that, for tests).
    """
    clip_len_s = (clip_end_ms - clip_start_ms) / 1000.0
    clip_start_s = clip_start_ms / 1000.0
    fmt = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
    mute_ranges = [(a, b) for a, b in (mute_ranges or []) if b > a]

    sorted_segs = sorted(segments, key=lambda s: (s.get("start_ms", 0), s.get("sort_order", 0)))
    # Skip overlapping segments — same rule as the preview (first/lowest-sort_order wins)
    _filtered: list[dict] = []
    _next_start = -1
    for _s in sorted_segs:
        if _s.get("start_ms", 0) >= _next_start:
            _filtered.append(_s)
            _next_start = _s.get("end_ms", 0)
    sorted_segs = _filtered

    has_broll_audio = any(
        (s.get("crop_boxes") or [{}])[0].get("source_video_id") in secondary_videos
        for s in sorted_segs
        if (s.get("crop_boxes") or [{}])[0].get("source_video_id")
    )
    has_frames = any(is_frame(s.get("layout")) for s in sorted_segs)
    main_vol = 0.0 if main_detached else max(0.0, float(main_volume))

    fp: list[str] = []
    if not sorted_segs or (not has_broll_audio and not has_frames):
        # ── Bed, simple case: the main video's sound for the clip ──
        inputs: list[str] = ["ffmpeg", "-y", "-ss", f"{clip_start_s:.3f}", "-t", f"{clip_len_s:.3f}", "-i", source_video]
        fp.append(f"[0:a]asetpts=PTS-STARTPTS,{fmt},volume={main_vol:.3f}[speech]")
    else:
        # ── Bed, per section (B-roll shots and frames): atrim on one pre-seeked input is sample-accurate ──
        plans: list[list[tuple[str | None, float, float, float, float]]] = []
        for seg in sorted_segs:
            seg_dur = (int(seg["end_ms"]) - int(seg["start_ms"])) / 1000.0
            if is_frame(seg.get("layout")):
                plan = [p for p in frame_audio_plan(seg, secondary_videos) if not p[0] or _has_audio(secondary_videos[p[0]])]
                plans.append([(v, o, vol * (main_vol if v is None else 1.0), d, dur) for v, o, vol, d, dur in plan])
                continue
            box = (seg.get("crop_boxes") or [{}])[0]
            vid_id = box.get("source_video_id") if box else None
            if vid_id and box.get("muted"):
                # A muted B-roll shot is a cutaway: the speaker keeps talking under it
                plans.append([(None, int(seg["start_ms"]) / 1000.0, main_vol, 0.0, seg_dur)])
            elif vid_id and vid_id in secondary_videos:
                # Its own sound, at the volume set on it
                vol = box.get("volume")
                plans.append([(vid_id, int(box.get("source_offset_ms") or 0) / 1000.0, 1.0 if vol is None else float(vol), 0.0, seg_dur)])
            else:
                _cb_off = box.get("source_offset_ms") if box else None
                vid_off_ms = int(seg["video_offset_ms"]) if seg.get("video_offset_ms") is not None else (int(_cb_off) if _cb_off is not None else int(seg["start_ms"]))
                plans.append([(None, vid_off_ms / 1000.0, main_vol, 0.0, seg_dur)])

        broll_uses: dict[str, int] = {}
        main_uses = 0
        for plan in plans:
            for vid_id, *_ in plan:
                if vid_id:
                    broll_uses[vid_id] = broll_uses.get(vid_id, 0) + 1
                else:
                    main_uses += 1

        inputs = ["ffmpeg", "-y", "-ss", f"{clip_start_s:.3f}", "-i", source_video]   # index 0
        broll_idx: dict[str, int] = {}
        for vid_id in broll_uses:
            broll_idx[vid_id] = len(broll_idx) + 1
            # looped: frames loop videos shorter than their format
            inputs += ["-stream_loop", "-1", "-i", secondary_videos[vid_id]]

        # After the input-level seek the audio PTS start near clip_start_s, not 0: normalise first
        main_pool: list[str] = []
        if main_uses > 0:
            fp.append("[0:a]asetpts=PTS-STARTPTS[main_a_n]")
            if main_uses > 1:
                lbls = [f"[ma{i}]" for i in range(main_uses)]
                fp.append(f"[main_a_n]asplit={main_uses}{''.join(lbls)}")
                main_pool = lbls
            else:
                main_pool = ["[main_a_n]"]
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
        broll_its = {v: iter(p) for v, p in broll_pools.items()}

        norm_labels: list[str] = []
        for i, (seg, plan) in enumerate(zip(sorted_segs, plans)):
            dur_s = (int(seg["end_ms"]) - int(seg["start_ms"])) / 1000.0
            norm_lbl = f"[san{i}]"
            pieces: list[str] = []
            for j, (vid_id, off_s, vol, delay_s, piece_s) in enumerate(plan):
                src = next(broll_its[vid_id]) if vid_id else next(main_it)
                lbl = f"[sa{i}_{j}]"
                vol_f = f",volume={vol:.3f}" if abs(vol - 1.0) > 1e-3 else ""
                delay_f = f",adelay=delays={int(round(delay_s * 1000))}:all=1" if delay_s > 0.0005 else ""
                # Each piece to stereo 48 kHz (B-roll may be 44.1 kHz)
                fp.append(f"{src}atrim=start={off_s:.3f}:end={off_s + piece_s:.3f},asetpts=PTS-STARTPTS{vol_f},{fmt}{delay_f}{lbl}")
                pieces.append(lbl)
            if is_frame(seg.get("layout")):
                # A silent bed of exactly the frame's length with its sounds on top, each at its own volume
                bed = f"[sbed{i}]"
                fp.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={dur_s:.3f},{fmt}{bed}")
                if pieces:
                    fp.append(f"{bed}{''.join(pieces)}amix=inputs={len(pieces) + 1}:duration=first:dropout_transition=0:normalize=0,{fmt}{norm_lbl}")
                else:
                    fp.append(f"{bed}anull{norm_lbl}")
            else:
                fp.append(f"{pieces[0]}anull{norm_lbl}")
            norm_labels.append(norm_lbl)
        fp.append(f"{''.join(norm_labels)}concat=n={len(sorted_segs)}:v=0:a=1[speech]")

    # ── Muted sections silence the bed (the picture's own sound), never the music ──
    if mute_ranges:
        fp.append(f"[speech]volume=0:enable='{_between(mute_ranges)}'[bed]")
    else:
        fp.append("[speech]anull[bed]")

    # ── Audio tracks: music, and detached original-sound bars ──
    track_labels: list[str] = []
    for tr in audio_tracks:
        original = tr.get("kind") == "original"
        path = source_video if original else (tr.get("local_path") or tr.get("storage_path"))
        if not path:
            continue
        start_s = max(0.0, float(tr.get("start_ms") or 0) / 1000.0)
        end_ms = tr.get("end_ms")
        end_s = min(clip_len_s, float(end_ms) / 1000.0 if end_ms is not None else clip_len_s)
        off_s = max(0.0, float(tr.get("offset_ms") or 0) / 1000.0)
        dur = end_s - start_s
        if not original:
            # A file FFmpeg can't read (any file type can be picked) is left out instead of failing
            # the export; a song shorter than its bar stops where it ends (and fades fit what plays)
            flen = _media_len(path)
            if flen is None or not _has_audio(path):
                print(f"[audio] skipping a music file that can't be read: {tr.get('storage_path')}", flush=True)
                continue
            dur = min(dur, flen - off_s)
        if dur <= 0.05:
            continue
        idx = sum(1 for a in inputs if a == "-i")
        # Seek on the input (fast on long files), then cut exactly the bar's length
        inputs += ["-ss", f"{off_s:.3f}", "-i", path]
        vol = float(tr.get("volume") if tr.get("volume") is not None else (1.0 if original else 0.5))
        chain = f"[{idx}:a]atrim=start=0:end={dur:.3f},asetpts=PTS-STARTPTS,{fmt},volume={vol:.3f}"
        fade = min(FADE_S, dur / 2)
        if tr.get("fade_in"):
            chain += f",afade=t=in:st=0:d={fade:.3f}"
        if tr.get("fade_out"):
            chain += f",afade=t=out:st={dur - fade:.3f}:d={fade:.3f}"
        if start_s > 0.0005:
            chain += f",adelay=delays={int(round(start_s * 1000))}:all=1"
        # From here t is clip time: dip under speech (music), muted sections (original sound)
        if not original and tr.get("duck_under_speech") and speech_ranges:
            chain += f",volume='if({_between(speech_ranges)},{DUCK_LEVEL},1)':eval=frame"
        if original and mute_ranges:
            chain += f",volume=0:enable='{_between(mute_ranges)}'"
        label = f"[trk{len(track_labels)}]"
        fp.append(chain + label)
        track_labels.append(label)

    # ── Mix (each input at its own level) and level for social apps ──
    if track_labels:
        fp.append(f"[bed]{''.join(track_labels)}amix=inputs={len(track_labels) + 1}:duration=first:dropout_transition=0:normalize=0[mix]")
    else:
        fp.append("[bed]anull[mix]")
    fp.append(f"[mix]{LOUDNESS_FILTER if loudness else 'anull'}[aout]")

    return [
        *inputs,
        "-filter_complex", ";".join(fp),
        "-map", "[aout]",
        "-acodec", "aac", "-b:a", "192k",
        output_audio,
    ]


def _subtract(ranges, cuts):
    """`ranges` with every part inside `cuts` removed."""
    out = []
    for a, b in ranges:
        pieces = [(a, b)]
        for c, d in cuts:
            nxt = []
            for x, y in pieces:
                if d <= x or c >= y:
                    nxt.append((x, y))
                    continue
                if c > x:
                    nxt.append((x, c))
                if d < y:
                    nxt.append((d, y))
            pieces = nxt
        out += pieces
    return [(a, b) for a, b in out if b > a]


def heard_speech_ranges(
    speech_ms: list[tuple[int, int]],
    clip_start_ms: int,
    clip_len_ms: int,
    segments: list[dict],
    secondary_videos: dict[str, str],
    mute_ranges: list[tuple[int, int]],
    audio_tracks: list[dict],
    main_detached: bool,
    main_volume: float,
) -> list[tuple[int, int]]:
    """
    Where the voice is actually HEARD, in clip time — music dips only there. `speech_ms` are the
    spoken stretches in video time. Not heard: the original sound muted or at 0%, muted sections,
    sections where an added video's own sound plays instead, frames whose main video is silent.
    With the original sound detached, the voice is heard where its bars play (shifted when a bar
    was moved, nowhere when it's at 0%).
    """
    if main_detached:
        heard = []
        for tr in audio_tracks:
            if tr.get("kind") != "original":
                continue
            if tr.get("volume") is not None and float(tr["volume"]) <= 0:
                continue
            start = max(0, int(tr.get("start_ms") or 0))
            end = int(tr["end_ms"]) if tr.get("end_ms") is not None else clip_len_ms
            off = int(tr["offset_ms"]) if tr.get("offset_ms") is not None else clip_start_ms + start
            for a, b in speech_ms:
                x, y = max(start, start + a - off), min(end, start + b - off)
                if y > x:
                    heard.append((x, y))
    else:
        if main_volume <= 0:
            return []
        heard = [(max(0, a - clip_start_ms), min(clip_len_ms, b - clip_start_ms)) for a, b in speech_ms]
        quiet = []
        for seg in segments:
            span = (int(seg["start_ms"]), int(seg["end_ms"]))
            if is_frame(seg.get("layout")):
                if main_audio_volume(frame_state(seg)) <= 0:
                    quiet.append(span)
                continue
            box = (seg.get("crop_boxes") or [{}])[0] or {}
            if box.get("source_video_id") and box["source_video_id"] in secondary_videos and not box.get("muted"):
                quiet.append(span)   # its own sound plays instead of the voice
        heard = _subtract(heard, quiet)
    heard = sorted(_subtract(heard, mute_ranges))
    merged: list[tuple[int, int]] = []
    for a, b in heard:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


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
