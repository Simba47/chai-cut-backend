"""The reference transcript of one audio file (architecture plan, 1 Nov 2026).

  1. cut at pauses (Silero VAD, MIT) into pieces of at most 25 s
  2. words: Bodhan AI Indic-Transcribe core (Indic Open Model Licence: credit line in the app)
  3. word times: IISc SraVaani 1.0 (MIT) times, carried onto those words (align.py)
Output: {"model", "language", "audio_seconds", "gpu_seconds", "words": [{"word", "start_ms", "end_ms"}]}

Run on one file (for checks):  python -m transcriber.pipeline audio.wav out.json [--language te]
"""
import json
import subprocess
import sys
import time
from collections import Counter

import numpy as np
import torch

from transcriber.align import transfer_timings

SR = 16_000
MAX_PIECE_S = 25.0
PAD_S = 0.15
TEXT_MODEL = "bodhan-ai/indic-transcribe-core"
TIME_MODEL = "ARTPARK-IISc/SraVaani-1.0"


def load_audio(source: str) -> np.ndarray:
    """Any audio or video file or URL -> 16 kHz mono float32 (ffmpeg)"""
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", source, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        check=True, capture_output=True, timeout=30 * 60,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def pieces(audio: np.ndarray, vad) -> list[tuple[float, float]]:
    """Speech stretches, joined while short, cut at pauses, none longer than MAX_PIECE_S"""
    from silero_vad import get_speech_timestamps
    total = len(audio) / SR
    segs: list[list[float]] = []
    for p in get_speech_timestamps(torch.from_numpy(audio), vad, sampling_rate=SR, return_seconds=True):
        s, e = max(0.0, p["start"] - PAD_S), min(total, p["end"] + PAD_S)
        if segs and e - segs[-1][0] <= MAX_PIECE_S:
            segs[-1][1] = e
        else:
            while e - s > MAX_PIECE_S:
                segs.append([s, s + MAX_PIECE_S]); s += MAX_PIECE_S
            segs.append([s, e])
    return [(a, b) for a, b in segs]


class Transcriber:
    """Loads the models once (a service keeps one of these per GPU)"""

    def __init__(self, device: str | None = None):
        from huggingface_hub import snapshot_download
        from silero_vad import load_silero_vad
        from transformers import AutoModel
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.vad = load_silero_vad()
        text_dir = snapshot_download(TEXT_MODEL)
        sys.path.insert(0, text_dir)          # the model's code ships inside its download
        from indic_transcribe import IndicTranscribe
        self.text = IndicTranscribe.from_pretrained(text_dir, device=self.device)
        # SraVaani's TorchScript runs on CUDA or CPU (not Apple's GPU)
        self.time_device = "cuda" if self.device.startswith("cuda") else "cpu"
        self.timer = AutoModel.from_pretrained(TIME_MODEL, trust_remote_code=True).to(self.time_device).eval()

    def _language(self, audio: np.ndarray, segs: list[tuple[float, float]]) -> str:
        """The video's language, decided once from a few of its longest pieces"""
        longest = sorted(segs, key=lambda s: s[1] - s[0], reverse=True)[:3]
        votes = Counter()
        for a, b in longest:
            _, lid = self.text.transcribe(audio[int(a * SR):int(b * SR)], lang=None, return_lid=True)
            votes[lid["lang"]] += 1
        return votes.most_common(1)[0][0] if votes else "te"

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, language: str | None = None) -> dict:
        t0 = time.time()
        segs = pieces(audio, self.vad)
        lang = language or (self._language(audio, segs) if segs else "te")
        chunks = [audio[int(a * SR):int(b * SR)] for a, b in segs]
        timed = self.timer.transcribe(chunks, batch_size=8, timestamps=True) if chunks else []
        words = []
        for (a, b), chunk, hyp in zip(segs, chunks, timed):
            text = self.text.transcribe(chunk, lang=lang, mode="native").strip()
            if not text:
                continue
            ref = [{"word": w["word"], "start": a + w["start"], "end": a + w["end"]}
                   for w in (getattr(hyp, "timestamp", None) or {}).get("word", [])]
            for w in transfer_timings(text.split(), ref, a, b):
                words.append({"word": w["word"], "start_ms": round(w["start"] * 1000), "end_ms": round(w["end"] * 1000)})
        return {"model": "indic-transcribe-core+sravaani-times", "language": lang,
                "audio_seconds": round(len(audio) / SR, 2), "gpu_seconds": round(time.time() - t0, 2), "words": words}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("audio"); ap.add_argument("out"); ap.add_argument("--language")
    args = ap.parse_args()
    t = Transcriber()
    result = t.transcribe(load_audio(args.audio), args.language)
    json.dump(result, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"{result['audio_seconds'] / 60:.1f} min -> {len(result['words'])} words, {result['language']}, in {result['gpu_seconds'] / 60:.1f} min on {t.device}")
