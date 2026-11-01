"""The reference transcript of one audio file (architecture plan, 1 Nov 2026).

  1. cut at pauses (Silero VAD, MIT) into pieces of at most 25 s
  2. words: Bodhan AI Indic-Transcribe core (Indic Open Model Licence: credit line in the app)
  3. word times: those words fitted, letter by letter, to a wav2vec2 model's (Apache-2.0) frames (align.py)
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

from transcriber.align import word_times

SR = 16_000
MAX_PIECE_S = 25.0
PAD_S = 0.15
TEXT_MODEL = "bodhan-ai/indic-transcribe-core"
# Letter-level timing models (wav2vec2 CTC, one frame per 20 ms) by language, pinned to the checked
# revision. Telugu: XLSR-53 fine-tuned on OpenSLR 66, Apache-2.0. A language without one gets its
# pieces shared out by word length (other languages: Phase 4).
TIME_MODELS = {"te": ("anuragshas/wav2vec2-large-xlsr-53-telugu", "102b87629072c9e8562f43a1864fb3ede8df11fa")}
FRAME_S = 0.02


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


class LetterFrames:
    """One language's timing model: frames x letters log-probabilities for a piece of audio"""

    def __init__(self, lang: str, device: str):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        name, revision = TIME_MODELS[lang]
        self.device = device
        self.proc = Wav2Vec2Processor.from_pretrained(name, revision=revision)
        self.model = Wav2Vec2ForCTC.from_pretrained(name, revision=revision).to(device).eval()
        self.vocab = self.proc.tokenizer.get_vocab()
        self.blank = self.proc.tokenizer.pad_token_id

    @torch.inference_mode()
    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        x = self.proc(chunk, sampling_rate=SR, return_tensors="pt").input_values.to(self.device)
        return self.model(x).logits[0].log_softmax(-1).float().cpu().numpy()


class Transcriber:
    """Loads the models once (a service keeps one of these per GPU)"""

    def __init__(self, device: str | None = None):
        from huggingface_hub import snapshot_download
        from silero_vad import load_silero_vad
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.vad = load_silero_vad()
        text_dir = snapshot_download(TEXT_MODEL)
        sys.path.insert(0, text_dir)          # the model's code ships inside its download
        from indic_transcribe import IndicTranscribe
        self.text = IndicTranscribe.from_pretrained(text_dir, device=self.device)
        self.timers = {lang: LetterFrames(lang, self.device) for lang in TIME_MODELS}

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
        timer = self.timers.get(lang)
        words = []
        for a, b in segs:
            chunk = audio[int(a * SR):int(b * SR)]
            text = self.text.transcribe(chunk, lang=lang, mode="native").strip()
            if not text:
                continue
            lp = timer(chunk) if timer else None
            for w in word_times(text.split(), lp, timer.vocab if timer else {}, timer.blank if timer else -1, FRAME_S, a, b):
                words.append({"word": w["word"], "start_ms": round(w["start"] * 1000), "end_ms": round(w["end"] * 1000)})
        return {"model": "indic-transcribe-core+wav2vec2-letter-times", "language": lang,
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
