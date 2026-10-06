"""
Colour emoji for the export (captions, text overlays, frame text), from fonts/ChaiEmoji.ttf
(built by build_emoji_font.py from Twemoji 17). The editor's preview draws the same font, so an
emoji looks the same before and after downloading.

An emoji is a stack of single-colour layers (emoji_layers.json: per emoji, each layer's
private-use codepoint and colour). The caption renderer stacks them as caption layers
(colour_emoji_ass); text overlays get a picture drawn from them (emoji_png).
"""
from __future__ import annotations

import json
import os
import re
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(HERE, "fonts", "ChaiEmoji.ttf")
FONT_NAME = "Chai Emoji"
SPACER = "\uE000"          # an empty glyph as wide as an emoji: holds its place in a caption line
_MAX_SEQ = 10              # the longest emoji sequence, in codepoints (families with skin tones)
# The font's emoji metrics, in em (build_emoji_font.py): 1.1 em of room, the square 0.05 em in from
# each side, 0.82 em above the baseline and 0.18 em below — text overlays place pictures the same way
ADVANCE_EM, SIDE_EM, ABOVE_EM = 1.1, 0.05, 0.82

_layers: dict[str, list] | None = None
_spaces: dict[str, str] | None = None


def _emoji_wide_spaces(font: str) -> str | None:
    """The caption font's own spaces, as wide as an emoji (build_emoji_font.py --spaces)"""
    global _spaces
    if _spaces is None:
        try:
            with open(os.path.join(HERE, "emoji_spaces.json"), encoding="utf-8") as fh:
                _spaces = json.load(fh)
        except OSError:
            _spaces = {}
    return _spaces.get(font)


def _map() -> dict[str, list]:
    global _layers
    if _layers is None:
        with open(os.path.join(HERE, "emoji_layers.json"), encoding="utf-8") as fh:
            _layers = json.load(fh)
    return _layers


def _key(seq: str) -> str:
    return "-".join(f"{ord(c):x}" for c in seq if c != "\uFE0F")


def _can_start(c: str) -> bool:
    """A character that may begin an emoji. Below U+2190 (©, ®, digits, #, *) only with FE0F, as phones do."""
    o = ord(c)
    return o >= 0x2190 and o != 0xFE0F and o != 0x200D


def split_emoji(text: str) -> list[tuple[bool, str]]:
    """Alternating (is_emoji, text) runs; each emoji run is ONE emoji (a whole ZWJ sequence, flag, …)"""
    m = _map()
    runs: list[tuple[bool, str]] = []
    plain = ""
    i, n = 0, len(text)
    while i < n:
        found = 0
        c = text[i]
        maybe = _can_start(c) or (i + 1 < n and text[i + 1] == "\uFE0F")
        if maybe:
            # Longest match first; an FE0F after the emoji belongs to it
            for j in range(min(n, i + _MAX_SEQ * 2), i, -1):
                if _key(text[i:j]) in m:
                    found = j
                    break
        if found:
            while found < n and text[found] == "\uFE0F":
                found += 1
            if plain:
                runs.append((False, plain))
                plain = ""
            runs.append((True, text[i:found]))
            i = found
        else:
            plain += c
            i += 1
    if plain:
        runs.append((False, plain))
    return runs


def has_emoji(text: str) -> bool:
    return any(e for e, _ in split_emoji(text))


def layers(emoji: str) -> list[tuple[str, str, float]]:
    """[(layer character in the Chai Emoji font, '#RRGGBB', opacity)] bottom to top"""
    return [(chr(e[0]), e[1], e[2] if len(e) > 2 else 1.0) for e in _map().get(_key(emoji), [])]


# ── Pictures (text overlays, frame text) ─────────────────────────────────────────

_PNG_DIR = os.path.join(tempfile.gettempdir(), "chai-cut-emoji")


def emoji_png(emoji: str, size: int) -> str | None:
    """A size×size PNG of the emoji (transparent background), drawn locally from its layers; cached"""
    ls = layers(emoji)
    if not ls or size < 2:
        return None
    path = os.path.join(_PNG_DIR, f"{_key(emoji)}_{size}.png")
    if os.path.exists(path):
        return path
    from PIL import Image, ImageFont   # Pillow (installed with matplotlib)
    os.makedirs(_PNG_DIR, exist_ok=True)
    # The font's emoji square is 1000 units at x 50…1050, y -180…820: at font size `size` it's size px
    font = ImageFont.truetype(FONT_PATH, size)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    x, baseline = -round(size * 0.05), round(size * 0.82)
    for ch, col, op in ls:
        mask = Image.new("L", (size, size), 0)
        from PIL import ImageDraw
        ImageDraw.Draw(mask).text((x, baseline), ch, font=font, fill=255, anchor="ls")
        if op < 1:
            mask = mask.point(lambda v: round(v * op))
        layer = Image.new("RGBA", (size, size), tuple(int(col[k:k + 2], 16) for k in (1, 3, 5)) + (255,))
        img.paste(layer, (0, 0), mask)
    tmp = f"{path}.{os.getpid()}.tmp"
    img.save(tmp, format="PNG")
    os.replace(tmp, path)       # two renders at once never read a half-written file
    return path


# ── Captions (libass) ────────────────────────────────────────────────────────────

_COLOUR_TAGS = re.compile(r"\\(?:[1-4]?c|[1-4]a|alpha|bord|xbord|ybord|shad|xshad|yshad|blur|be)(?:&H[0-9A-Fa-f]+&?|-?[\d.]+)?")


def _ass_colour(hex_colour: str) -> str:
    h = hex_colour.lstrip("#")
    return f"&H{h[4:6]}{h[2:4]}{h[0:2]}&"


def _ass_alpha(op: float) -> str:
    return f"&H{255 - round(op * 255):02X}&"


def _strip_colour(block: str) -> str:
    """An override block {…} without colour, alpha, outline, shadow or blur (also inside \\t(…))"""
    body = _COLOUR_TAGS.sub("", block[1:-1])
    body = re.sub(r"\\t\((?:[\d.,\s-]*)\)", "", body)     # \t(…) left empty
    return "{" + body + "}"


def colour_emoji_ass(lines: list[str]) -> list[str]:
    """
    ASS lines with emoji drawn in full colour. Each caption line with emoji is kept, its emoji
    replaced by an empty spacer of the same width; on top go copies of the line, one per colour
    layer, where only that layer of each emoji shows (everything else invisible). The copies keep
    every position, size and animation tag, so the colours move with the words; colour, outline
    and blur tags are dropped from them so the layers keep their own colours.
    """
    style_font: dict[str, str] = {}
    # Styles whose text is invisible (alpha FF): the copies presets draw boxes from
    box_only: set[str] = set()
    for ln in lines:
        if ln.startswith("Style:"):
            parts = [x.strip() for x in ln[6:].split(",")]
            style_font[parts[0]] = parts[1]
            if len(parts) > 3 and parts[3].upper().startswith("&HFF"):
                box_only.add(parts[0])
    out: list[str] = []
    extra: list[str] = []
    for ln in lines:
        if not ln.startswith("Dialogue:"):
            out.append(ln)
            continue
        fields = ln[len("Dialogue:"):].split(",", 9)
        if len(fields) < 10 or not has_emoji(re.sub(r"\{[^}]*\}", "", fields[9])):
            out.append(ln)
            continue
        base_font = style_font.get(fields[3].strip(), "Roboto")
        tokens = re.split(r"(\{[^}]*\})", fields[9])
        if fields[3].strip() in box_only:
            # Only its box shows: the emoji's width in the font's own spaces keeps the box in one
            # piece (a font switch splits it into overlapping, darker strips); no colour layers
            font, parts = base_font, []
            for tok in tokens:
                if tok.startswith("{"):
                    m = re.findall(r"\\fn([^\\}]+)", tok)
                    font = m[-1] if m else font
                    parts.append(tok)
                    continue
                for is_e, s in split_emoji(tok):
                    sp = _emoji_wide_spaces(font) if is_e else None
                    parts.append(s if not is_e else sp if sp else f"{{\\fn{FONT_NAME}}}{SPACER}{{\\fn{font}}}")
            out.append("Dialogue:" + ",".join(fields[:9] + ["".join(parts)]))
            continue

        def rebuild(k: int | None) -> str:
            """The line's text; k=None: the line itself, else only colour layer k of its emoji"""
            font = base_font
            parts: list[str] = []
            hide = "\\1a&HFF&\\3a&HFF&\\4a&HFF&\\bord0\\shad0\\blur0\\be0"
            if k is not None:
                parts.append("{" + hide + "}")
            for tok in tokens:
                if tok.startswith("{"):
                    m = re.findall(r"\\fn([^\\}]+)", tok)
                    if m:
                        font = m[-1]
                    if k is None:
                        parts.append(tok)
                    else:
                        t = _strip_colour(tok)
                        # \r resets to the style (visible again): hide right after it
                        parts.append(t[:-1] + hide + "}" if "\\r" in t else t)
                    continue
                for is_e, s in split_emoji(tok):
                    if not is_e:
                        parts.append(s)
                        continue
                    ls = layers(s)
                    if k is None or k >= len(ls):
                        parts.append(f"{{\\fn{FONT_NAME}}}{SPACER}{{\\fn{font}}}")
                    else:
                        ch, col, op = ls[k]
                        parts.append(f"{{\\fn{FONT_NAME}\\1c{_ass_colour(col)}\\1a{_ass_alpha(op)}}}{ch}"
                                     f"{{\\fn{font}\\1a&HFF&}}")
            return "".join(parts)

        depth = max((len(layers(s)) for t in tokens if not t.startswith("{") for e, s in split_emoji(t) if e), default=0)
        out.append("Dialogue:" + ",".join(fields[:9] + [rebuild(None)]))
        for k in range(depth):
            layer = str(int(fields[0].strip() or 0) + 50 + k)    # above the line and its boxes
            extra.append("Dialogue:" + ",".join([layer] + fields[1:9] + [rebuild(k)]))
    return out + extra
