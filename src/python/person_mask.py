"""
Captions drawn behind the person (the Creator styles "Text behind": creator_captions.py, look
"behind"). The clip is rendered without its captions first; this then finds the person in every
frame of it (MediaPipe's selfie segmenter, selfie_segmenter.tflite checked in next to this file),
burns the captions in and lays the person back over them — the words go behind their head.

The editor does the same live with the same model (modules/captions/personMask.ts).
"""
import os
import subprocess

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "selfie_segmenter.tflite")

MASK_W = 256           # the person is found on a copy this wide (the model works at 256 x 256)
EDGE_LO, EDGE_HI = 0.35, 0.65   # confidence → mask: below LO background, above HI person, a soft edge between
SMOOTH = 0.35          # each frame's mask keeps this much of the last one (no flicker at the edges)


class _Segmenter:
    def __init__(self):
        os.environ.setdefault("MPLBACKEND", "Agg")  # mediapipe's drawing helpers import matplotlib
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        self._mp = mp
        self._seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        ))

    def person(self, rgb: np.ndarray, t_ms: int) -> np.ndarray:
        """How sure it is that each pixel is the person, 0..1"""
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        masks = self._seg.segment_for_video(image, t_ms).confidence_masks
        return masks[-1].numpy_view().astype(np.float32).reshape(rgb.shape[:2])

    def close(self):
        self._seg.close()


def _probe(path: str) -> tuple[int, int, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout.strip().split(",")
    n, d = (out[2].split("/") + ["1"])[:2]
    return int(out[0]), int(out[1]), (float(n) / float(d or 1)) or 30.0


def mask_video(video_path: str, mask_path: str) -> None:
    """A small grey video, white where the person is: one frame for each of the video's"""
    w, h, fps = _probe(video_path)
    mw = MASK_W
    mh = max(2, round(h * mw / w / 2) * 2)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", video_path, "-vf", f"scale={mw}:{mh}:flags=area",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{mw}x{mh}",
                            "-r", f"{fps:.6f}", "-i", "-", "-c:v", "ffv1", mask_path], stdin=subprocess.PIPE)
    seg = _Segmenter()
    size = mw * mh * 3
    prev = None
    i = 0
    try:
        while True:
            buf = dec.stdout.read(size)
            if len(buf) < size:
                break
            rgb = np.frombuffer(buf, np.uint8).reshape(mh, mw, 3)
            m = np.clip((seg.person(rgb, round(i * 1000 / fps)) - EDGE_LO) / (EDGE_HI - EDGE_LO), 0, 1)
            if prev is not None:
                m = m * (1 - SMOOTH) + prev * SMOOTH
            prev = m
            enc.stdin.write((m * 255).astype(np.uint8).tobytes())
            i += 1
    finally:
        seg.close()
        enc.stdin.close()
        dec.stdout.close()
        dec.wait()
        enc.wait()
    if enc.returncode != 0 or i == 0:
        raise RuntimeError(f"person mask failed (frames {i}, encoder exit {enc.returncode})")
    print(f"[render] person found in {i} frames", flush=True)


def captions_behind(video_path: str, subtitles: str, out_path: str, tmp: str, venc: list[str]) -> None:
    """
    Burns the captions (`subtitles`: the filter, e.g. subtitles=filename=…) into `video_path`
    under the person and writes `out_path`. `venc`: the video encoder options render.py uses.
    """
    mask_path = os.path.join(tmp, "person-mask.mkv")
    mask_video(video_path, mask_path)
    w, h, _ = _probe(video_path)
    graph = (f"[1:v]scale={w}:{h}:flags=bicubic,format=gray,setpts=PTS-STARTPTS[m];"
             f"[0:v]setpts=PTS-STARTPTS,split[a][b];"
             f"[a]{subtitles}[cap];"
             f"[b][m]alphamerge[p];"
             f"[cap][p]overlay=format=auto:shortest=1,format=yuv420p[v]")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", video_path, "-i", mask_path, "-filter_complex", graph,
           "-map", "[v]", "-map", "0:a?", *venc, "-c:a", "copy", "-map_chapters", "-1", out_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[render] captions behind the person failed:\n{result.stderr[-3000:]}", flush=True)
        raise RuntimeError(f"FFmpeg (captions behind) failed (exit {result.returncode})")
