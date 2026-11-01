"""Word timings for the reference transcript, with MIT-licensed parts only.

The text model (Indic-Transcribe) writes the best words but gives no word times. The timing model
(IISc SraVaani, MIT) gives its own words with times. Both texts are turned into the same plain
sound-letters (romanize) and lined up letter by letter; each text word takes the times of the
letters it lined up with. Words the timing model missed sit between their neighbours, sized by
length. (Meta's MMS aligner did this in the 1 Nov tests, but its licence is non-commercial.)
"""
import difflib
import re

# Telugu letters -> plain sound letters (the same for different spellings of one word:
# టైమ్ and టైం both give "taim"). Not for display: the captions' Tinglish comes from elsewhere.
_VOWELS = {"అ": "a", "ఆ": "a", "ఇ": "i", "ఈ": "i", "ఉ": "u", "ఊ": "u", "ఋ": "ru", "ౠ": "ru",
           "ఎ": "e", "ఏ": "e", "ఐ": "ai", "ఒ": "o", "ఓ": "o", "ఔ": "au"}
_SIGNS = {"ా": "a", "ి": "i", "ీ": "i", "ు": "u", "ూ": "u", "ృ": "ru", "ౄ": "ru",
          "ె": "e", "ే": "e", "ై": "ai", "ొ": "o", "ో": "o", "ౌ": "au"}
_CONS = {"క": "k", "ఖ": "k", "గ": "g", "ఘ": "g", "ఙ": "n", "చ": "c", "ఛ": "c", "జ": "j", "ఝ": "j",
         "ఞ": "n", "ట": "t", "ఠ": "t", "డ": "d", "ఢ": "d", "ణ": "n", "త": "t", "థ": "t", "ద": "d",
         "ధ": "d", "న": "n", "ప": "p", "ఫ": "p", "బ": "b", "భ": "b", "మ": "m", "య": "y", "ర": "r",
         "ఱ": "r", "ల": "l", "ళ": "l", "ఴ": "l", "వ": "v", "శ": "s", "ష": "s", "స": "s", "హ": "h",
         "ౘ": "c", "ౙ": "j"}
_VIRAMA, _ANUSVARA = "్", "ం"


def romanize(word: str) -> str:
    """One word as plain lower-case sound letters (Telugu script or English)"""
    out, i = [], 0
    word = word.replace("‌", "").replace("‍", "")
    while i < len(word):
        ch, nxt = word[i], word[i + 1] if i + 1 < len(word) else ""
        if ch in _CONS:
            out.append(_CONS[ch])
            if nxt == _VIRAMA:
                i += 1
            elif nxt in _SIGNS:
                out.append(_SIGNS[nxt]); i += 1
            else:
                out.append("a")
        elif ch in _VOWELS:
            out.append(_VOWELS[ch])
        elif ch == _ANUSVARA:
            out.append("m")
        elif ch in _SIGNS:
            out.append(_SIGNS[ch])
        else:
            out.append(ch.lower())
        i += 1
    return re.sub(r"[^a-z0-9]", "", "".join(out))


def transfer_timings(words: list[str], timed: list[dict], seg_start: float, seg_end: float) -> list[dict]:
    """
    Times for `words` (one piece of audio, seg_start..seg_end seconds) from `timed`
    ([{"word", "start", "end"}], another model's words for the same piece). Returns
    [{"word", "start", "end"}] in order, inside the piece, never going backwards.
    """
    if not words:
        return []
    timed = sorted(({"word": t["word"], "start": min(max(t["start"], seg_start), seg_end),
                     "end": min(max(t["end"], seg_start), seg_end)} for t in timed if t.get("word")), key=lambda t: t["start"])

    # Letters of the text words, and which word each letter belongs to
    w_rom = [romanize(w) or "x" for w in words]
    w_owner = [k for k, r in enumerate(w_rom) for _ in r]
    # Letters of the timed words, each with its own slice of its word's time
    t_letters: list[tuple[float, float]] = []
    t_str = []
    for t in timed:
        r = romanize(t["word"])
        if not r:
            continue
        step = (t["end"] - t["start"]) / len(r)
        t_letters += [(t["start"] + step * m, t["start"] + step * (m + 1)) for m in range(len(r))]
        t_str.append(r)

    starts: list[float | None] = [None] * len(words)
    ends: list[float | None] = [None] * len(words)
    if t_letters:
        matcher = difflib.SequenceMatcher(None, "".join(w_rom), "".join(t_str), autojunk=False)
        for a, b, n in matcher.get_matching_blocks():
            for m in range(n):
                k = w_owner[a + m]
                s, e = t_letters[b + m]
                starts[k] = s if starts[k] is None else min(starts[k], s)
                ends[k] = e if ends[k] is None else max(ends[k], e)

    # Words without times: share the gap between the timed words around them, by length
    out = []
    k = 0
    while k < len(words):
        if starts[k] is not None:
            out.append({"word": words[k], "start": starts[k], "end": ends[k]})
            k += 1
            continue
        j = k
        while j < len(words) and starts[j] is None:
            j += 1
        left = out[-1]["end"] if out else seg_start
        right = starts[j] if j < len(words) else seg_end
        if right < left:
            right = left
        total = sum(len(w_rom[x]) for x in range(k, j))
        cur = left
        for x in range(k, j):
            d = (right - left) * len(w_rom[x]) / total if total else 0.0
            out.append({"word": words[x], "start": cur, "end": cur + d})
            cur += d
        k = j

    # Never backwards, never outside the piece
    prev_start = seg_start
    for i, w in enumerate(out):
        w["start"] = min(max(w["start"], prev_start, seg_start), seg_end)
        w["end"] = min(max(w["end"], w["start"]), seg_end)
        if i and out[i - 1]["end"] > w["start"]:
            out[i - 1]["end"] = w["start"]
        prev_start = w["start"]
    return out
