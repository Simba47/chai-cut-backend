"""
The Creator caption styles: caption_templates.json templates with a `look` (group "Creator").
Mirrored by the editor's modules/captions/creator.ts (drawCreatorCaptions): same lines, same
places, same timing.

Every size is px at 1080 wide (k = out_w / 1080). A font of size N is drawn so its full height
(OS/2 winAscent + winDescent) is N, as libass sizes it; its baseline is `asc` of that from the top
(the fonts' em / asc in caption_templates.json). Words appear when they are spoken (a short fade).

  behind      one big word at a time at the top, drawn behind the person (render.py cuts the
              person out of the finished video and lays them back over it: person_mask.py)
  difference  the key word big in difference blend (it inverts the picture under it: render.py
              draws it on black and blends that with the video), the words before it small above
              and those after it small below
  build       one word a line, centred, building to three; the key word in the highlight colour
  multi       the line's words small and the key word big (its own font), placed in a new corner
              each line
  scroll      a teleprompter: the spoken word big in the highlight colour, the one before above and
              the next below faded; the column moves up a step at each word
  split       one line: plain words and the key word big in its own font, in the highlight colour
  subtitle    a small plain sentence that builds word by word; the spoken word bright, the rest dimmed
  scribble    a handwritten line; key words get a hand-drawn wavy underline in the highlight colour

The key word of a line is the first emphasised one (caption_styles.emphasis, picked by AI and
changed in the editor), else its longest word ("build": its middle one).
"""
import json
import math
import sys
from pathlib import Path

_DIR = Path(__file__).parent
_FONTS_DIR = _DIR / "fonts"

FADE_MS = 120          # a word fades in over this when spoken
BUILD_MS = 150         # build: a word rises into place
SCROLL_MS = 180        # scroll: the column's step up
SCRIBBLE_MS = 260      # scribble: the underline is drawn over this
SUBTITLE_DIM = 0x59    # subtitle: words already spoken (~65% opaque)
GHOST = 0xB8           # scroll: the words before / after (~28% opaque)
FIT = 0.9              # split / subtitle / scribble: a line narrowed to fit this share of the width

LOOKS = {"behind", "difference", "build", "multi", "scroll", "split", "subtitle", "scribble"}
# Where a look sits when the captions haven't been moved (position_y)
DEFAULT_Y = {"behind": 0.2, "difference": 0.22, "build": 0.58, "multi": 0.1, "scroll": 0.62,
             "split": 0.76, "subtitle": 0.62, "scribble": 0.72}
# multi: a new corner each line (left/right, top of the block as a share of the height)
MULTI_SPOTS = [("left", 0.10), ("right", 0.10), ("right", 0.40), ("left", 0.40)]


def _load_fonts() -> dict[str, dict]:
    try:
        with open(_DIR / "caption_templates.json", encoding="utf-8") as f:
            return json.load(f).get("fonts", {})
    except (OSError, ValueError) as e:
        print(f"[render] creator caption fonts not loaded: {e}", file=sys.stderr)
        return {}


_FONTS = _load_fonts()


def is_creator(tpl: dict) -> bool:
    return tpl.get("look") in LOOKS


def style_line() -> str:
    """The ASS style the Creator events are drawn with (each event sets its own font and size)"""
    return ("Style: Creator,Roboto,52,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
            "0,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1")


# ── small helpers ─────────────────────────────────────────────────────────────

def _ts(ms: int) -> str:
    ms = max(0, int(ms))
    return f"{ms // 3_600_000}:{(ms // 60_000) % 60:02d}:{(ms // 1000) % 60:02d}.{(ms // 10) % 100:02d}"


def _color(hex_color: str) -> str:
    h = (hex_color or "#FFFFFF").lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return f"&H{h[4:6]}{h[2:4]}{h[0:2]}&".upper()


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def _font(key: str) -> dict:
    return _FONTS.get(key) or _FONTS.get("poppins") or {"family": "Roboto", "bold": 0, "italic": 0, "file": "Roboto-Regular.ttf", "em": 0.758, "asc": 0.75}


# Lines in another script are drawn in its Noto font, with the style's own weight and slant put on
# it (libass thickens / slants it; the editor's canvas does the same). Latin letters in such a line
# are found by font fallback. The editor's creator.ts has the same list.
SCRIPTS = [(0x0C00, 0x0C7F, "noto-telugu"), (0x0900, 0x097F, "noto-devanagari"), (0x0B80, 0x0BFF, "noto-tamil"),
           (0x0C80, 0x0CFF, "noto-kannada"), (0x0D00, 0x0D7F, "noto-malayalam"), (0x0980, 0x09FF, "noto-bengali")]


def script_font(text: str) -> str | None:
    """The Noto font for the first non-Latin script in the text, None for Latin text"""
    for ch in text:
        o = ord(ch)
        for lo, hi, key in SCRIPTS:
            if lo <= o <= hi:
                return key
    return None


def _scripted(key: str, script: str | None) -> str:
    """The font a style's font `key` is drawn in for a line in `script` (a key of _FONTS)"""
    if not script or script not in _FONTS:
        return key
    k = f"{script}|{key}"
    if k not in _FONTS:
        s, f = _FONTS[script], _font(key)
        bold = int(f.get("weight", 400)) >= 600 or key in HEAVY
        _FONTS[k] = {**s, "bold": 1 if bold else 0, "italic": int(f.get("italic", 0))}
    return k


# Display fonts that are one heavy weight though their files say regular (400)
HEAVY = {"anton", "bangers", "bebas", "lilita", "luckiest"}


def _fn(key: str, n: float) -> str:
    f = _font(key)
    return f"\\fn{f['family']}\\b{f['bold']}\\i{f['italic']}\\fs{n:.1f}"


_pil_fonts: dict[tuple[str, float], object] = {}


def _width(key: str, n: float, text: str) -> float:
    """How wide `text` is in font `key` at size n (as libass lays it out: the em is n × em)"""
    from PIL import ImageFont
    f = _font(key)
    size = round(max(1.0, n * float(f["em"])), 1)
    pf = _pil_fonts.get((key, size))
    if pf is None:
        pf = ImageFont.truetype(str(_FONTS_DIR / f["file"]), size=size)
        _pil_fonts[(key, size)] = pf
    return float(pf.getlength(text))


def _reveal(on: int, to: int = 0) -> str:
    """Hidden until spoken (`on`, ms into the event), then fades in to alpha `to`"""
    if on <= 0:
        return f"\\alpha&H{to:02X}&"
    return f"\\alpha&HFF&\\t({on},{on + FADE_MS},\\alpha&H{to:02X}&)"


def _letters(t: str) -> int:
    return sum(ch.isalnum() for ch in t)


def key_index(texts: list[str], emph: list[bool], look: str) -> int:
    """The line's key word: the first emphasised, else the longest (build: the middle one)"""
    for j, e in enumerate(emph):
        if e:
            return j
    if look == "build":
        return min(1, len(texts) - 1)
    best = 0
    for j, t in enumerate(texts):
        if _letters(t) > _letters(texts[best]):
            best = j
    return best


def _keys(texts: list[str], emph: list[bool], any_emphasis: bool) -> set[int]:
    """scribble: every emphasised word; with no AI picks at all, each line's longest"""
    if any(emph):
        return {j for j, e in enumerate(emph) if e}
    return set() if any_emphasis else {key_index(texts, emph, "scribble")}


def _wave_path(x0: float, x1: float, yc: float, amp: float, lam: float, th: float) -> str:
    """A hand-drawn underline: a wavy band from x0 to x1 (ASS drawing; the editor draws the same)"""
    n = max(2, int((x1 - x0) / (lam / 8)))
    xs = [x0 + (x1 - x0) * i / n for i in range(n + 1)]
    mid = [yc + amp * math.sin(2 * math.pi * (x - x0) / lam) for x in xs]
    pts = [(x, y - th / 2) for x, y in zip(xs, mid)] + [(x, y + th / 2) for x, y in reversed(list(zip(xs, mid)))]
    first, *rest = pts
    return f"m {first[0]:.0f} {first[1]:.0f} l " + " ".join(f"{x:.0f} {y:.0f}" for x, y in rest)


# ── the events ────────────────────────────────────────────────────────────────

def events(lines: list[tuple[list[tuple[dict, str]], int, int]], style: dict, tpl: dict,
           emphasis: dict, out_w: int, out_h: int, pos_x: int) -> tuple[list[str], list[str], bool]:
    """
    ASS events for a Creator style. `lines`: each caption line's words (word, the text shown) and
    when it is on screen (clip ms). Returns (events, difference events, behind): the difference
    events go in a file of their own (render.py blends it), `behind` = draw under the person.
    """
    look = tpl["look"]
    k = out_w / 1080
    W, H = out_w, out_h
    color = style.get("color") or tpl.get("color") or "#FFFFFF"
    hl = style.get("highlight_color") or tpl.get("hl") or "#FFE700"
    S = float(style.get("size") or 52) * float(tpl.get("size") or 100) / 100 * k
    key_n = S * float(tpl.get("keyScale") or 100) / 100
    base_font, base_key_font = tpl["font"], tpl.get("keyFont") or tpl["font"]
    font, key_font = base_font, base_key_font
    shad = f"\\shad{float(tpl.get('shadow') or 0) * k:.1f}\\bord0"
    py = style.get("position_y")
    cy = float(py if py is not None else DEFAULT_Y[look]) * H
    any_emphasis = any(bool(v) for v in emphasis.values())
    out: list[str] = []
    diff: list[str] = []

    def dialogue(a: int, b: int, text: str, layer: int = 1) -> str:
        return f"Dialogue: {layer},{_ts(a)},{_ts(b)},Creator,,0,0,0,,{text}"

    def fitted(n: float, texts: list[str]) -> float:
        """A one-font line's size, narrowed so the line fits FIT of the width"""
        w = sum(_width(font, n, t) for t in texts) + _width(font, n, " ") * (len(texts) - 1)
        return n * min(1.0, FIT * W / w) if w > 0 else n

    def baseline_center(parts: list[tuple[str, float]], center: float) -> float:
        """Where a line of (font, size) runs centred on `center` has its baseline (libass: the
        tallest ascent and deepest descent in the line make its height)"""
        a = max(n * float(_font(f).get("asc", 0.8)) for f, n in parts)
        d = max(n * (1 - float(_font(f).get("asc", 0.8))) for f, n in parts)
        return center - (a + d) / 2 + a

    for li, (shown, a, b) in enumerate(lines):
        texts = [t for _, t in shown]
        # (a line in another script: its Noto font, in the style's weight and slant)
        script = script_font(" ".join(texts))
        font, key_font = _scripted(base_font, script), _scripted(base_key_font, script)
        emph = [bool(emphasis.get(str(w.get("_src_start", w["start_ms"])))) for w, _ in shown]
        ons = [w["start_ms"] - a for w, _ in shown]
        offs = [(shown[j + 1][0]["start_ms"] if j + 1 < len(shown) else b) - a for j in range(len(shown))]
        kk = key_index(texts, emph, look)

        if look == "behind":
            # One word: as big as the size asks, narrowed to fit 88% of the width
            t = texts[0]
            w0 = _width(font, S, t)
            n = S * min(1.0, 0.88 * W / w0) if w0 > 0 else S
            out.append(dialogue(a, b, f"{{\\an5\\pos({pos_x},{cy:.0f}){_fn(font, n)}\\1c{_color(color)}\\bord0\\shad0}}{_esc(t)}"))

        elif look == "difference":
            # The words before the key word small above it, those after it small below (read in order)
            rows = [(list(range(0, kk)), S * 0.95), ([kk], key_n * 0.9), (list(range(kk + 1, len(texts))), S * 0.95)]
            rows = [(idx, h) for idx, h in rows if idx]
            y = cy - sum(h for _, h in rows) / 2
            for idx, h in rows:
                yc = y + h / 2
                if idx == [kk]:
                    diff.append(dialogue(a, b, f"{{\\an5\\pos({pos_x},{yc:.0f}){_fn(key_font, key_n)}\\1c{_color(color)}\\bord0\\shad0"
                                               f"{_reveal(ons[kk])}}}{_esc(texts[kk])}"))
                else:
                    runs = " ".join(f"{{{_reveal(ons[j])}}}{_esc(texts[j])}" for j in idx)
                    out.append(dialogue(a, b, f"{{\\an5\\pos({pos_x},{yc:.0f}){_fn(font, S)}\\1c{_color(color)}{shad}}}{runs}"))
                y += h

        elif look == "build":
            n_words = len(texts)
            pitch = S * 0.92
            for j, t in enumerate(texts):
                y = cy + (j - (n_words - 1) / 2) * pitch
                c = hl if j == kk else color
                on = ons[j]
                place = (f"\\move({pos_x},{y + S * 0.18:.0f},{pos_x},{y:.0f},{on},{on + BUILD_MS})" if on > 0
                         else f"\\pos({pos_x},{y:.0f})")
                out.append(dialogue(a, b, f"{{\\an5{place}{_fn(font, S)}\\1c{_color(c)}{shad}{_reveal(on)}}}{_esc(t)}"))

        elif look == "multi":
            side, top_share = MULTI_SPOTS[li % len(MULTI_SPOTS)]
            # (moved left or right with the captions, as the editor moves them)
            x = (round(W * 0.08) if side == "left" else round(W * 0.92)) + pos_x - W // 2
            an = 7 if side == "left" else 9
            rows = [(range(0, kk), font, S), (range(kk, kk + 1), key_font, key_n), (range(kk + 1, len(texts)), font, S)]
            y = top_share * H
            for idx, f, n in rows:
                idx = list(idx)
                if not idx:
                    continue
                runs = " ".join(f"{{{_reveal(ons[j])}}}{_esc(texts[j])}" for j in idx)
                out.append(dialogue(a, b, f"{{\\an{an}\\pos({x},{y:.0f}){_fn(f, n)}\\1c{_color(color)}{shad}}}{runs}"))
                y += n * 0.95

        elif look == "scroll":
            pitch = S * 1.05
            for j in range(len(texts)):
                s0 = a + (ons[j] if j else 0)
                s1 = a + offs[j]
                if s1 <= s0:
                    continue
                for d in (-1, 0, 1):
                    i = j + d
                    if not 0 <= i < len(texts):
                        continue
                    y = cy + d * pitch
                    n = S if d == 0 else S * 0.8
                    alpha = "&H00&" if d == 0 else f"&H{GHOST:02X}&"
                    place = (f"\\move({pos_x},{y + pitch:.0f},{pos_x},{y:.0f},0,{SCROLL_MS})" if j
                             else f"\\pos({pos_x},{y:.0f})")
                    out.append(dialogue(s0, s1, f"{{\\an5{place}{_fn(font, n)}\\1c{_color(hl)}\\alpha{alpha}{shad}}}{_esc(texts[i])}"))

        elif look == "split":
            # One line, the key word in its own font and size: libass centres the line's height
            # (tallest ascent + deepest descent) on cy, the words on one baseline. Narrowed to fit.
            parts = [(key_font, key_n) if j == kk else (font, S) for j in range(len(texts))]
            total = sum(_width(f, n, t + (" " if j + 1 < len(texts) else "")) for j, ((f, n), t) in enumerate(zip(parts, texts)))
            fit = min(1.0, FIT * W / total) if total > 0 else 1.0
            runs = []
            for j, t in enumerate(texts):
                f, n = parts[j][0], parts[j][1] * fit
                c = hl if j == kk else color
                runs.append(f"{{{_fn(f, n)}\\1c{_color(c)}{_reveal(ons[j])}}}{_esc(t)}{' ' if j + 1 < len(texts) else ''}")
            out.append(dialogue(a, b, f"{{\\an5\\pos({pos_x},{cy:.0f}){shad}}}" + "".join(runs)))

        elif look == "subtitle":
            n = fitted(S, texts)
            runs = []
            for j, t in enumerate(texts):
                tag = _reveal(ons[j])
                if 0 < offs[j] < b - a:
                    tag += f"\\t({offs[j]},{offs[j] + 1},\\alpha&H{SUBTITLE_DIM:02X}&)"
                elif offs[j] <= 0:
                    tag = f"\\alpha&H{SUBTITLE_DIM:02X}&"
                runs.append(f"{{{tag}}}{_esc(t)}")
            out.append(dialogue(a, b, f"{{\\an5\\pos({pos_x},{cy:.0f}){_fn(font, n)}\\1c{_color(color)}{shad}}}" + " ".join(runs)))

        elif look == "scribble":
            n = fitted(S, texts)
            keys = _keys(texts, emph, any_emphasis)
            runs = " ".join(f"{{{_reveal(ons[j])}}}{_esc(t)}" for j, t in enumerate(texts))
            out.append(dialogue(a, b, f"{{\\an5\\pos({pos_x},{cy:.0f}){_fn(font, n)}\\1c{_color(color)}{shad}}}{runs}"))
            # The underlines, under each key word (laid out as libass lays the line: measured)
            space = _width(font, n, " ")
            widths = [_width(font, n, t) for t in texts]
            x = pos_x - (sum(widths) + space * (len(texts) - 1)) / 2
            base = baseline_center([(font, n)], cy)
            amp, lam, th = n * 0.06, n * 0.42, max(2.0, n * 0.07)
            for j, t in enumerate(texts):
                if j in keys:
                    x0, x1 = x, x + widths[j]
                    path = _wave_path(x0, x1, base + n * 0.14, amp, lam, th)
                    on = ons[j]
                    clip = (f"\\clip({x0 - th:.0f},0,{x0 - th:.0f},{H})\\t({on},{on + SCRIBBLE_MS},\\clip({x0 - th:.0f},0,{x1 + th:.0f},{H}))"
                            if on > 0 else "")
                    out.append(dialogue(a, b, f"{{\\an7\\pos(0,0)\\1c{_color(hl)}\\bord0\\shad0{clip}\\p1}}{path}{{\\p0}}", layer=0))
                x += widths[j] + space

    return out, diff, look == "behind"
