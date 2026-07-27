"""
Caption burn-in for Telugu/Devanagari/Latin text.

Font search order:
  1. FONTS_DIR env var (Docker volume or Railway mount)
  2. Same directory as this script
  3. /usr/share/fonts (system)

Karaoke: active word is highlighted; the rest are at normal opacity.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Telugu → Roman fallback transliterator ───────────────────────────────────
_TE_CONSONANTS = {
    'క':'k','ఖ':'kh','గ':'g','ఘ':'gh','ఙ':'ng',
    'చ':'ch','ఛ':'chh','జ':'j','ఝ':'jh','ఞ':'ny',
    'ట':'t','ఠ':'th','డ':'d','ఢ':'dh','ణ':'n',
    'త':'t','థ':'th','ద':'d','ధ':'dh','న':'n',
    'ప':'p','ఫ':'ph','బ':'b','భ':'bh','మ':'m',
    'య':'y','ర':'r','ల':'l','వ':'v',
    'శ':'sh','ష':'sh','స':'s','హ':'h',
    'ళ':'l','ఱ':'r',
}
_TE_VOWELS = {
    'అ':'a','ఆ':'aa','ఇ':'i','ఈ':'ee',
    'ఉ':'u','ఊ':'oo','ఋ':'ru',
    'ఎ':'e','ఏ':'e','ఐ':'ai',
    'ఒ':'o','ఓ':'o','ఔ':'au',
}
_TE_VOWEL_SIGNS = {
    'ా':'aa','ి':'i','ీ':'ee','ు':'u','ూ':'oo','ృ':'ru',
    'ె':'e','ే':'e','ై':'ai','ొ':'o','ో':'o','ౌ':'au',
    '్':'','ఁ':'',
}

def _transliterate_telugu(word: str) -> str:
    chars = list(word)
    out: list[str] = []
    i = 0
    while i < len(chars):
        c = chars[i]
        nxt = chars[i + 1] if i + 1 < len(chars) else ''
        if c in _TE_CONSONANTS:
            base = _TE_CONSONANTS[c]
            if nxt == '్':
                out.append(base); i += 2
            elif nxt in _TE_VOWEL_SIGNS:
                out.append(base + _TE_VOWEL_SIGNS[nxt]); i += 2
                c2 = chars[i] if i < len(chars) else ''
                if c2 == 'ం': out.append('m'); i += 1
                elif c2 == 'ః': out.append('h'); i += 1
            else:
                out.append(base + 'a'); i += 1
                c2 = chars[i] if i < len(chars) else ''
                if c2 == 'ం': out.append('m'); i += 1
                elif c2 == 'ః': out.append('h'); i += 1
        elif c in _TE_VOWELS:
            out.append(_TE_VOWELS[c]); i += 1
            c2 = chars[i] if i < len(chars) else ''
            if c2 == 'ం': out.append('m'); i += 1
            elif c2 == 'ః': out.append('h'); i += 1
        elif c == 'ం':
            out.append('m'); i += 1
        elif c == 'ః':
            out.append('h'); i += 1
        elif c in _TE_VOWEL_SIGNS:
            out.append(_TE_VOWEL_SIGNS[c]); i += 1
        elif c.isascii():
            out.append(c); i += 1
        else:
            i += 1
    return ''.join(out).lower()

def _word_display(w: dict) -> str:
    roman = w.get("word_roman")
    if roman:
        return roman
    raw = w.get("word", "")
    if raw and any('ఀ' <= ch <= '౿' for ch in raw):
        return _transliterate_telugu(raw)
    return raw

FONT_DIR_ENV = os.environ.get("FONTS_DIR", "")

FONT_FILES = {
    "noto-sans-telugu":     "NotoSansTelugu-Regular.ttf",
    "noto-sans-devanagari": "NotoSansDevanagari-Regular.ttf",
    "roboto":               "Roboto-Regular.ttf",
    "montserrat-bold":      "Montserrat-Bold.ttf",
}

SEARCH_DIRS = [
    Path(FONT_DIR_ENV) if FONT_DIR_ENV else None,
    Path(__file__).parent / "fonts",
    Path("/usr/share/fonts"),
    Path("/System/Library/Fonts"),  # macOS
]


_font_cache: dict[tuple, ImageFont.FreeTypeFont] = {}

def _find_font(font_id: str, size: int) -> ImageFont.FreeTypeFont:
    key = (font_id, size)
    if key in _font_cache:
        return _font_cache[key]
    filename = FONT_FILES.get(font_id, FONT_FILES["roboto"])
    for directory in SEARCH_DIRS:
        if directory is None:
            continue
        for path in directory.rglob(filename):
            try:
                font = ImageFont.truetype(str(path), size)
                _font_cache[key] = font
                return font
            except Exception:
                pass
    # Fallback: PIL default (Latin only, safe across Pillow versions)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:
        font = ImageFont.load_default()
    _font_cache[key] = font
    return font


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def burn_captions(
    frame_bgr: np.ndarray,
    t_ms: int,
    words: list[dict],
    style: dict,
) -> np.ndarray:
    """
    Burn captions onto a BGR numpy frame.

    words    : [{word, start_ms, end_ms}, ...]
    style    : {font, size, color, position, animation}
    """
    animation  = style.get("animation") or "karaoke"
    font_id    = style.get("font") or "noto-sans-telugu"
    size       = int(style.get("size") or 52)
    color_hex  = style.get("color") or "#ffffff"
    position_y = style.get("position_y")  # float 0-1, None = use legacy position string
    position   = style.get("position") or "bottom-center"

    # Filter to words currently visible
    # For karaoke: show current sentence (words around the active word)
    # For fade/none: show all words in a ±2s window as plain text
    active_words = _get_visible_words(t_ms, words, animation)
    if not active_words:
        return frame_bgr

    font = _find_font(font_id, size)
    text_color = _hex_to_rgb(color_hex)

    # Convert BGR → PIL RGB
    pil_img = Image.fromarray(cv2_to_pil(frame_bgr))
    draw = ImageDraw.Draw(pil_img)

    out_w, out_h = pil_img.size

    full_text = " ".join(_word_display(w) for w in active_words)
    if not full_text:
        return frame_bgr
    # NotoSansTelugu has poor Latin glyph coverage — switch to Roboto for romanized text
    if full_text.isascii():
        font = _find_font("roboto", size)
    bbox = draw.textbbox((0, 0), full_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x, y = _get_text_origin(position, out_w, out_h, text_w, text_h, position_y)

    # Shadow
    draw.text((x + 2, y + 2), full_text, font=font, fill=(0, 0, 0, 180))
    draw.text((x, y), full_text, font=font, fill=text_color)

    return pil_to_cv2(np.array(pil_img))


def _get_visible_words(t_ms: int, words: list[dict], animation: str) -> list[dict]:
    if animation == "karaoke":
        # Find the sentence containing the active word
        active_idx = next(
            (i for i, w in enumerate(words) if w["start_ms"] <= t_ms < w["end_ms"]),
            None,
        )
        if active_idx is None:
            # Show the upcoming sentence if within 500ms
            next_idx = next((i for i, w in enumerate(words) if w["start_ms"] > t_ms), None)
            if next_idx is not None and words[next_idx]["start_ms"] - t_ms < 500:
                active_idx = next_idx
            else:
                return []

        # Walk backward and forward to find sentence boundaries
        start = active_idx
        while start > 0 and not _is_sentence_end(words[start - 1]["word"]):
            start -= 1
        end = active_idx
        while end < len(words) - 1 and not _is_sentence_end(words[end]["word"]):
            end += 1

        return words[start : end + 1]
    else:
        return [w for w in words if w["start_ms"] <= t_ms < w["end_ms"] + 500]


def _is_sentence_end(word: str) -> bool:
    return bool(word) and word[-1] in ".!?।"


def _get_text_origin(
    position: str, frame_w: int, frame_h: int, text_w: int, text_h: int,
    position_y: float | None = None,
) -> tuple[int, int]:
    padding = 60
    if position_y is not None:
        # position_y is fraction 0-1 representing the CENTER of the text block (matches editor drag handle)
        y = int(position_y * frame_h) - text_h // 2
        y = max(padding, min(frame_h - text_h - padding, y))
    elif "bottom" in position:
        y = frame_h - text_h - padding
    elif "top" in position:
        y = padding
    else:
        y = (frame_h - text_h) // 2

    x = max(padding, (frame_w - text_w) // 2)
    return x, y


def cv2_to_pil(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR numpy to RGB numpy for PIL."""
    return bgr[:, :, ::-1]


def pil_to_cv2(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB numpy back to BGR for OpenCV."""
    return rgb[:, :, ::-1]
