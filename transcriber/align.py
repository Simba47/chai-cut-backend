"""Word timings for the reference transcript, with commercially usable parts only.

The text model (Indic-Transcribe) writes the best words but gives no word times. A letter-level
wav2vec2 model (Apache-2.0) gives, for every 20 ms of sound, how likely each letter is (its CTC
output). The text's own words are fitted to that, letter by letter (forced alignment, the method
Meta's MMS aligner uses; MMS itself is non-commercial): each word starts at the frame where its
first letter is heard. Pieces of audio that cannot be fitted are shared out by word length.

Checked on the 63-minute hour_test, 2 Nov 2026 (share of words starting within 0.2 s of Gemini's):
this, 94.6% (97.4% of MMS's, median 15 ms apart); MMS 96.8%; SraVaani's times carried across 72%;
IndicConformer's 80 ms CTC frames 29% (they drift by up to 2 s).
"""
import numpy as np

WORD_GAP = "|"


def letters(word: str, vocab: dict[str, int]) -> list[int]:
    """The word's letters the model knows (others are left out); none known: the word-gap letter"""
    return [vocab[c] for c in word if c in vocab] or [vocab[WORD_GAP]]


def forced_align(logprobs: np.ndarray, targets: list[int], blank: int) -> list[tuple[int, int]] | None:
    """
    The most likely way the frames (logprobs: frames x vocab, log-probabilities) say `targets` in order,
    blanks allowed between them (CTC Viterbi). Returns each target's (first frame, last frame),
    or None when there are too few frames to say them all.
    """
    if not targets:
        return []
    T = len(logprobs)
    repeats = sum(a == b for a, b in zip(targets, targets[1:]))
    if T < len(targets) + repeats:
        return None
    ext = [blank]
    for k in targets:
        ext += [k, blank]
    S = len(ext)
    emit = np.asarray(logprobs, dtype=np.float64)[:, ext]
    can_skip = np.array([s >= 2 and ext[s] != blank and ext[s] != ext[s - 2] for s in range(S)])
    NEG = -1e30
    score = np.full(S, NEG)
    score[:2] = emit[0, :2]
    back = np.zeros((T, S), dtype=np.int8)
    for t in range(1, T):
        prev1 = np.concatenate(([NEG], score[:-1]))
        prev2 = np.where(can_skip, np.concatenate(([NEG, NEG], score[:-2])), NEG)
        cand = np.stack([score, prev1, prev2])
        best = cand.argmax(axis=0)
        score = cand[best, np.arange(S)] + emit[t]
        back[t] = best
    s = S - 1 if score[S - 1] >= score[S - 2] else S - 2
    if score[s] < NEG / 2:
        return None
    spans: list[list[int]] = [[-1, -1] for _ in targets]
    for t in range(T - 1, -1, -1):
        if s % 2:                       # odd places in ext are the targets
            k = s // 2
            spans[k][0] = t
            if spans[k][1] < 0:
                spans[k][1] = t
        s -= int(back[t, s])
    return [(a, b) for a, b in spans]


def _share_by_length(words: list[str], start: float, end: float) -> list[dict]:
    total = sum(len(w) for w in words) or 1
    out, cur = [], start
    for w in words:
        d = (end - start) * len(w) / total
        out.append({"word": w, "start": cur, "end": cur + d})
        cur += d
    return out


def word_times(words: list[str], logprobs: np.ndarray | None, vocab: dict[str, int], blank: int,
               frame_s: float, offset: float, piece_end: float) -> list[dict]:
    """
    Times for the words of one piece of audio (offset..piece_end seconds), from the timing model's
    frames for that piece (None: no frames, e.g. a language without a timing model).
    Returns [{"word", "start", "end"}] in order, each inside the piece, with some length, never
    overlapping the next.
    """
    if not words:
        return []
    toks = [letters(w, vocab) for w in words] if logprobs is not None else []
    spans = forced_align(logprobs, [k for t in toks for k in t], blank) if logprobs is not None else None
    if spans is None:
        return _share_by_length(words, offset, piece_end)
    out, i = [], 0
    for w, t in zip(words, toks):
        first, last = spans[i][0], spans[i + len(t) - 1][1]
        i += len(t)
        out.append({"word": w, "start": min(offset + first * frame_s, piece_end),
                    "end": min(offset + (last + 1) * frame_s, piece_end)})
    for k, w in enumerate(out):
        nxt = out[k + 1]["start"] if k + 1 < len(out) else piece_end
        w["end"] = min(max(w["end"], w["start"] + frame_s / 2), max(nxt, w["start"] + 1e-3), piece_end)
        if w["end"] <= w["start"]:
            w["start"] = max(offset, w["end"] - 1e-3)
    return out
