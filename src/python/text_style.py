"""
A positioned text with its Text-tool look, drawn as the editor's preview draws it (VideoPreview
drawStyledText): weight, italic, capitals, letter spacing, opacity, rotation, outline, background
box, shadow / glow, and its entrance animation — each frame of the entrance, then the text at rest.

Before 2026-11-04 the export drew texts with drawtext (font, size, colour, a fixed shadow) and the
look was never even saved: a "Creator" or "Label" text exported plain (user: "export everything is
missing"). The same numbers as the preview, at the export's size (px at 1080 wide, × out_w / 1080).
"""
from __future__ import annotations

import math
import os
import tempfile

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

import emoji as emoji_art

# The settings that make a text "styled" (anything else is drawn the plain drawtext way, as before)
STYLE_KEYS = ("weight", "italic", "uppercase", "letter_spacing", "opacity", "stroke_color", "stroke_width",
              "bg_color", "bg_opacity", "bg_radius", "shadow", "shadow_color", "animation")
# The preview's own font files (VideoPreview EXPORT_FONTS): a text in one of these is drawn with it
# at the weight asked (regular file: thickened); any other font is the family's Bold file
BUNDLED = {"roboto", "montserrat-bold", "noto-sans-telugu", "noto-sans-devanagari", "noto-sans-tamil",
           "noto-sans-kannada", "noto-sans-malayalam", "noto-sans-bengali"}
# The Indian scripts' letters: the bundled font for each, and a letter to try a font with
SCRIPTS = (("ఀ", "౿", "noto-sans-telugu", "క"), ("ऀ", "ॿ", "noto-sans-devanagari", "क"),
           ("஀", "௿", "noto-sans-tamil", "க"), ("ಀ", "೿", "noto-sans-kannada", "ಕ"),
           ("ഀ", "ൿ", "noto-sans-malayalam", "ക"), ("ঀ", "৿", "noto-sans-bengali", "ক"))
LINE_EM = 1.15            # textStyle.ts TEXT_LINE_EM
BG_PAD = (0.35, 0.18)     # textStyle.ts BG_PAD, em
LONG_ANIMS = {"drop", "bounce", "spin", "glitch", "flicker"}
FAKE_ITALIC = 0.25        # the browser's slant for a font with no italic (Skia: skewX −1/4)


_HAS_LETTER: dict[tuple[str, str], bool] = {}


def _has_letter(path: str, ch: str) -> bool:
    """Whether a font file can draw `ch` (it draws something else than for a letter no font has)"""
    key = (path, ch)
    if key not in _HAS_LETTER:
        font = ImageFont.truetype(path, 48)

        def ink(c: str) -> bytes:
            img = Image.new("L", (160, 120), 0)
            ImageDraw.Draw(img).text((20, 90), c, font=font, fill=255, anchor="ls")
            return img.tobytes()
        _HAS_LETTER[key] = ink(ch) != ink("\U0010ffff")
    return _HAS_LETTER[key]


def font_runs(text: str, base: str, find_font) -> list[tuple[str, str]]:
    """
    `text` in pieces, each with a font that has its letters: (font id, piece). `base` is the text's own
    font; Telugu, Hindi… it has no letters for go in their Noto font, and the English letters a Noto
    font has none of in Roboto (the export draws a piece in one font: the letters it lacks came out as
    boxes). What the preview's font list does letter by letter (VideoPreview letterFallbacks).
    Spaces, digits and punctuation stay with the letters before them.
    """
    script_fonts = {s[2] for s in SCRIPTS}
    in_script = lambda c: next((s for s in SCRIPTS if s[0] <= c <= s[1]), None)
    base_path = find_font(base)
    if not text or not base_path or (base not in script_fonts and not any(in_script(c) for c in text)):
        return [(base, text)] if text else []

    def pick(c: str, cur: str | None) -> str | None:
        script = in_script(c)
        if script:
            return base if _has_letter(base_path, script[3]) else script[2]
        if c.isspace() or c in "‌‍":
            return cur
        if not c.isalpha() and cur and _has_letter(find_font(cur) or base_path, c):
            return cur
        if not c.isalpha() and not cur and _has_letter(base_path, c):
            return None
        return base if _has_letter(base_path, c) else "roboto"

    out: list[tuple[str, str]] = []
    cur, piece = None, ""
    for c in text:
        f = pick(c, cur)
        if cur is None or f is None or f == cur:
            cur, piece = cur or f, piece + c
        else:
            out.append((cur, piece))
            cur, piece = f, c
    out.append((cur or base, piece))
    return out


def text_width(text: str, base: str, size: int, find_font) -> float:
    """How wide `text` is at `size`, each piece measured in its own font"""
    return sum(ImageFont.truetype(find_font(f) or find_font(base), size).getlength(p) for f, p in font_runs(text, base, find_font))


def is_styled(ov: dict) -> bool:
    """Whether a positioned text has any Text-tool look beyond the plain one"""
    if ov.get("x") is None:
        return False
    for k in STYLE_KEYS:
        v = ov.get(k)
        if v in (None, "", False, 0) or (k == "animation" and v == "none") or (k == "opacity" and v >= 0.999) \
                or (k == "shadow" and v == "soft") or (k == "bg_opacity" and not ov.get("bg_color")) \
                or (k == "bg_radius" and not ov.get("bg_color")) or (k == "shadow_color" and ov.get("shadow") in (None, "soft", "none")) \
                or (k == "weight" and v == (400 if (ov.get("font") or "") in BUNDLED else 700)):
            continue
        return True
    return False


def entrance_ms(ov: dict) -> int:
    """How long the entrance animation runs (0: none)"""
    anim = ov.get("animation") or "none"
    if anim == "none":
        return 0
    if anim == "typewriter":
        text = _shown(ov)
        return max(400, len(text) * 45)
    return 620 if anim in LONG_ANIMS else 400


def _shown(ov: dict) -> str:
    t = ov.get("text") or ""
    return t.upper() if ov.get("uppercase") else t


def _rgba(color: str | None, alpha: float = 1.0) -> tuple[int, int, int, int]:
    c = (color or "#ffffff").strip()
    if c.startswith("rgba(") or c.startswith("rgb("):
        parts = [p.strip() for p in c[c.index("(") + 1:c.rindex(")")].split(",")]
        r, g, b = (int(float(x)) for x in parts[:3])
        a = float(parts[3]) if len(parts) > 3 else 1.0
        return r, g, b, int(round(255 * max(0.0, min(1.0, a * alpha))))
    h = c.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    try:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(round(255 * max(0.0, min(1.0, alpha))))
    except ValueError:
        return 255, 255, 255, int(round(255 * alpha))


def _anim(ov: dict, dt: float, W: int, H: int, k: float) -> dict:
    """The entrance at `dt` ms in (VideoPreview drawStyledText, the same curves)"""
    anim = ov.get("animation") or "none"
    p = max(0.0, min(1.0, dt / (620 if anim in LONG_ANIMS else 400)))
    ease = 1 - (1 - p) ** 3

    def back(q: float) -> float:
        c = 2.2
        return 1 + (c + 1) * (q - 1) ** 3 + c * (q - 1) ** 2

    def bounce_out(q: float) -> float:
        n, d = 7.5625, 2.75
        if q < 1 / d:
            return n * q * q
        if q < 2 / d:
            q -= 1.5 / d
            return n * q * q + 0.75
        if q < 2.5 / d:
            q -= 2.25 / d
            return n * q * q + 0.9375
        q -= 2.625 / d
        return n * q * q + 0.984375

    def elastic(q: float) -> float:
        if q <= 0:
            return 0.0
        if q >= 1:
            return 1.0
        return 2 ** (-10 * q) * math.sin((q * 10 - 0.75) * (2 * math.pi) / 3) + 1

    a = dict(alpha=max(0.0, min(1.0, float(ov.get("opacity") if ov.get("opacity") is not None else 1))),
             scale=1.0, dx=0.0, dy=0.0, spin=0.0, blur=0.0, wipe=1.0, glitch=0.0, chars=None)
    full = _shown(ov)
    if anim == "fade":
        a["alpha"] *= ease
    elif anim == "pop":
        a["scale"] = 1 if p >= 1 else 0.6 + 0.4 * back(p)
        a["alpha"] *= min(1.0, p * 2.5)
    elif anim == "zoom-in":
        a["scale"] = 0.3 + 0.7 * ease
        a["alpha"] *= ease
    elif anim == "zoom-out":
        a["scale"] = 1.8 - 0.8 * ease
        a["alpha"] *= ease
    elif anim == "blur":
        a["blur"] = (1 - ease) * 14
        a["alpha"] *= ease
    elif anim == "slide":
        a["dy"] = (1 - ease) * 0.05 * H
        a["alpha"] *= ease
    elif anim == "slide-down":
        a["dy"] = -(1 - ease) * 0.05 * H
        a["alpha"] *= ease
    elif anim == "slide-left":
        a["dx"] = (1 - ease) * 0.08 * W
        a["alpha"] *= ease
    elif anim == "slide-right":
        a["dx"] = -(1 - ease) * 0.08 * W
        a["alpha"] *= ease
    elif anim == "drop":
        a["dy"] = -(1 - bounce_out(p)) * 0.14 * H
        a["alpha"] *= min(1.0, p * 4)
    elif anim == "bounce":
        a["scale"] = 1 if p >= 1 else max(0.05, elastic(p))
    elif anim == "spin":
        a["spin"] = (1 - ease) * -200
        a["scale"] = 0.4 + 0.6 * ease
        a["alpha"] *= ease
    elif anim == "wipe":
        a["wipe"] = ease
    elif anim == "flicker":
        if p < 1:
            seq = [1, 0.15, 0.9, 0.1, 1, 0.4, 1]
            a["alpha"] *= seq[min(len(seq) - 1, int(p * 7))]
    elif anim == "glitch":
        if p < 1:
            a["glitch"] = 1 - p
            a["dx"] = math.sin(dt * 0.9) * 10 * (W / 1080) * a["glitch"]
    elif anim == "typewriter":
        span = max(400, len(full) * 45)
        a["chars"] = math.ceil(len(full) * max(0.0, min(1.0, dt / span)))
    return a


def _bold_file(path: str) -> bool:
    name = os.path.basename(path).lower()
    return any(w in name for w in ("bold", "black", "heavy", "extrabold", "semibold"))


def frame(ov: dict, lines: list[str], font_path: str, W: int, H: int, dt: float, find_font=None) -> Image.Image | None:
    """
    The whole W × H frame (see-through but for the text) with the text as the preview draws it `dt`
    ms after it came in (math.inf: at rest). `lines`: its lines as the preview wraps them.
    """
    k = W / 1080
    size = max(8, int(round(float(ov.get("size") or 72) * k)))
    font = ImageFont.truetype(font_path, size)
    font_id = ov.get("font") or ""
    weight = ov.get("weight") or (400 if font_id in BUNDLED else 700)
    faces = {font_path: font}

    def face(path: str) -> ImageFont.FreeTypeFont:
        return faces.setdefault(path, ImageFont.truetype(path, size))

    # The browser thickens a regular file drawn bold (Skia's fake bold: ~1/32 of the size in all)
    def bold_px(path: str) -> int:
        return max(1, round(size / 64)) if weight >= 600 and not _bold_file(path) else 0
    italic = bool(ov.get("italic"))
    spacing = float(ov.get("letter_spacing") or 0) * k
    a = _anim(ov, dt, W, H, k)
    if a["alpha"] <= 0:
        return None

    full = _shown(ov)

    def pieces(ln: str) -> list[tuple[bool, str, str]]:
        out: list[tuple[bool, str, str]] = []
        for is_e, s in emoji_art.split_emoji(ln):
            if is_e or not find_font:
                out.append((is_e, s, font_path))
            else:
                out += [(False, p, find_font(f) or font_path) for f, p in font_runs(s, font_id or "roboto", find_font)]
        return out

    line_runs = [pieces(ln) for ln in lines]

    def run_w(is_e: bool, s: str, path: str) -> float:
        if is_e:
            return size * emoji_art.ADVANCE_EM
        return face(path).getlength(s) + spacing * len(s)

    widths = [[run_w(e, s, p) for e, s, p in runs] for runs in line_runs]
    text_w = max((sum(ws) for ws in widths), default=0.0)
    line_h = size * LINE_EM
    has_bg = bool(ov.get("bg_color"))
    pad_x = size * BG_PAD[0] if has_bg else 0.0
    pad_y = size * BG_PAD[1] if has_bg else 0.0
    text_box_h = line_h * len(lines) + pad_y * 2
    box_w = max(text_w + pad_x * 2, float(ov.get("w") or 0) * W)
    box_h = max(text_box_h, float(ov.get("h") or 0) * H)
    shift_y = (box_h - text_box_h) / 2

    # The text's own layer, its box at (m, m): room round it for the outline, shadow and slant
    stroke = float(ov.get("stroke_width") or 0) * k if ov.get("stroke_color") else 0.0
    m = int(size * 0.6 + stroke * 2 + 40 * k) + 8
    lw, lh = int(math.ceil(box_w + 2 * m)), int(math.ceil(box_h + 2 * m))
    layer = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))

    # The background box (no shadow on a text in a box, as the preview)
    if has_bg:
        bg = _rgba(ov.get("bg_color"), float(ov.get("bg_opacity") if ov.get("bg_opacity") is not None else 1))
        r = min(float(ov.get("bg_radius") if ov.get("bg_radius") is not None else 12) * k, box_h / 2)
        ImageDraw.Draw(layer).rounded_rectangle([m, m, m + box_w, m + box_h], radius=r, fill=bg)

    # Typewriter: the first characters of the text, line by line
    left = a["chars"] if a["chars"] is not None else len(full)
    shown_lines = []
    for ln in lines:
        shown_lines.append(ln[:max(0, left)])
        left -= len(ln) + 1

    # Each line's glyphs as a mask (and where its emoji go), its top where the canvas's 'top' puts it
    asc, desc = font.getmetrics()
    em_top = size * asc / (asc + desc)          # canvas textBaseline 'top' ≈ the em box's top
    tx = m + pad_x
    ty0 = m + shift_y + pad_y + (line_h - size) / 2

    def glyph_mask(extra: int = 0) -> Image.Image:
        mask = Image.new("L", (lw, lh), 0)
        d = ImageDraw.Draw(mask)
        for li, ln in enumerate(shown_lines):
            if not ln:
                continue
            base = ty0 + li * line_h + em_top
            cursor = tx
            used = 0
            for is_e, s, path in line_runs[li]:
                if used >= len(ln):
                    break
                s = s[:len(ln) - used]
                used += len(s)
                if is_e:
                    cursor += size * emoji_art.ADVANCE_EM
                    continue
                f, thick = face(path), extra + bold_px(path)
                if spacing:
                    for ch in s:
                        d.text((cursor, base), ch, font=f, fill=255, anchor="ls", stroke_width=thick)
                        cursor += f.getlength(ch) + spacing
                else:
                    d.text((cursor, base), s, font=f, fill=255, anchor="ls", stroke_width=thick)
                    cursor += f.getlength(s)
        if italic:
            # Slanted round each line's baseline (top leans right)
            out = Image.new("L", (lw, lh), 0)
            for li in range(len(shown_lines)):
                base = ty0 + li * line_h + em_top
                band_top = int(max(0, base - size * 1.2))
                band_bot = int(min(lh, base + size * 0.6))
                band = mask.crop((0, band_top, lw, band_bot))
                by = base - band_top
                sheared = band.transform(band.size, Image.AFFINE, (1, FAKE_ITALIC, -FAKE_ITALIC * by, 0, 1, 0), resample=Image.BICUBIC)
                out.paste(ImageChops.lighter(out.crop((0, band_top, lw, band_bot)), sheared), (0, band_top))
            mask = out
        return mask

    fill_mask = glyph_mask()
    color = _rgba(ov.get("color") or "#ffffff")

    def colored(mask: Image.Image, rgba: tuple[int, int, int, int]) -> Image.Image:
        solid = Image.new("RGBA", (lw, lh), rgba[:3] + (0,))
        alpha = mask if rgba[3] == 255 else mask.point(lambda v: v * rgba[3] // 255)
        solid.putalpha(alpha)
        return solid

    # The outline: a ring round the glyphs, 2 × width wide, centred on their edge (strokeText)
    stroke_img = None
    if stroke > 0:
        s = max(1, int(round(stroke)))
        outer = glyph_mask(s)
        inner = fill_mask.filter(ImageFilter.MinFilter(2 * s + 1)) if s < 40 else Image.new("L", (lw, lh), 0)
        stroke_img = colored(ImageChops.subtract(outer, inner), _rgba(ov.get("stroke_color")))

    # The shadow (none · soft · hard · glow), cast by the outline when there is one, else by the fill
    shadow = ov.get("shadow") or "soft"
    shadow_layers: list[Image.Image] = []
    if not has_bg and shadow != "none":
        caster = (Image.new("L", (lw, lh), 0) if stroke_img is None else stroke_img.getchannel("A"))
        if stroke_img is None or shadow == "glow":
            caster = ImageChops.lighter(caster, fill_mask)
        if shadow == "soft":
            sc, blur, ox, oy = (0, 0, 0, 178), 8 * k, 0, 2 * k
        elif shadow == "hard":
            sc, blur, ox, oy = _rgba(ov.get("shadow_color") or "#000000"), 0.0, 6 * k, 6 * k
        else:
            sc, blur, ox, oy = _rgba(ov.get("shadow_color") or ov.get("color") or "#ffffff"), 24 * k, 0.0, 0.0
        sh = caster
        if blur > 0:
            sh = sh.filter(ImageFilter.GaussianBlur(blur / 2))
        sh = ImageChops.offset(sh, int(round(ox)), int(round(oy)))
        shadow_layers.append(colored(sh, sc))
        # (a glow reads stronger with a second pass: the preview fills twice)
        if shadow == "glow":
            shadow_layers.append(colored(sh, sc))

    for sl in shadow_layers:
        layer.alpha_composite(sl)
    if stroke_img is not None:
        layer.alpha_composite(stroke_img)
    if color[3] > 0:
        layer.alpha_composite(colored(fill_mask, color))

    # Glitch: red and cyan copies split off either side for a moment
    if a["glitch"] > 0:
        off = int(round(8 * k * a["glitch"]))
        ga = 0.7 * a["glitch"]
        for dxg, col in ((-off, "#ff2a55"), (off, "#22e5ff")):
            layer.alpha_composite(colored(ImageChops.offset(fill_mask, dxg, 0), _rgba(col, ga)))

    # Emoji, as pictures, where their runs are
    for li, ln in enumerate(shown_lines):
        base = ty0 + li * line_h + em_top
        cursor = tx
        used = 0
        for (is_e, s, _), w in zip(line_runs[li], widths[li]):
            if used >= len(ln):
                break
            used += len(s)
            if is_e:
                png = emoji_art.emoji_png(s, int(size))
                if png:
                    em = Image.open(png).convert("RGBA")
                    layer.alpha_composite(em, (int(cursor + size * emoji_art.SIDE_EM), int(base - size * emoji_art.ABOVE_EM)))
            cursor += w

    # Wipe: revealed from the left
    if a["wipe"] < 1:
        keep = int(m - 4 * k + (box_w + 8 * k) * a["wipe"])
        cut = Image.new("L", (lw, lh), 0)
        ImageDraw.Draw(cut).rectangle([0, 0, max(0, keep), lh], fill=255)
        layer.putalpha(ImageChops.multiply(layer.getchannel("A"), cut))

    # Blur in, and the opacity of it all
    if a["blur"] > 0.2:
        layer = layer.filter(ImageFilter.GaussianBlur(a["blur"] * k))
    if a["alpha"] < 1:
        layer.putalpha(layer.getchannel("A").point(lambda v: int(v * a["alpha"])))

    # Scaled and turned round its box's middle; that middle where the preview puts it
    if a["scale"] != 1:
        nw, nh = max(1, int(lw * a["scale"])), max(1, int(lh * a["scale"]))
        layer = layer.resize((nw, nh), Image.BICUBIC)
    turn = float(ov.get("rotation") or 0) + a["spin"]
    if turn:
        layer = layer.rotate(-turn, resample=Image.BICUBIC, expand=True)
    x = float(ov.get("x") if ov.get("x") is not None else 0.1) * W
    y = float(ov.get("y") if ov.get("y") is not None else 0.4) * H + a["dy"]
    cx, cy = x + box_w / 2 + a["dx"], y + box_h / 2
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    # Laid on the see-through frame as it is (alpha_composite: a paste through its own alpha would
    # square every part-see-through pixel — opacity, shadow, glow, edges); parts off the frame are cut
    px, py = int(round(cx - layer.width / 2)), int(round(cy - layer.height / 2))
    sx, sy = max(0, -px), max(0, -py)
    ex, ey = min(layer.width, W - px), min(layer.height, H - py)
    if ex > sx and ey > sy:
        out.alpha_composite(layer, (px + sx, py + sy), (sx, sy, ex, ey))
    return out


def frames(ov: dict, lines: list[str], font_path: str, W: int, H: int, fps: float, tmp: str, tag: str, find_font=None) -> tuple[str | None, str | None, float]:
    """
    The text's entrance as numbered PNGs (an image sequence at `fps`; None when it has none) and
    the text at rest (one PNG), and how long the entrance lasts in s.
    """
    dur_ms = entrance_ms(ov)
    seq = None
    if dur_ms > 0:
        n = max(1, int(math.ceil(dur_ms / 1000 * fps)))
        d = os.path.join(tmp, f"tx_{tag}")
        os.makedirs(d, exist_ok=True)
        blank = None
        for i in range(n):
            img = frame(ov, lines, font_path, W, H, i * 1000 / fps, find_font)
            if img is None:
                blank = blank or Image.new("RGBA", (W, H), (0, 0, 0, 0))
                img = blank
            img.save(os.path.join(d, f"f{i:04d}.png"), compress_level=1)
        seq = os.path.join(d, "f%04d.png")
    rest = frame(ov, lines, font_path, W, H, math.inf, find_font)
    still = None
    if rest is not None:
        fd, still = tempfile.mkstemp(suffix=".png", prefix=f"tx_{tag}_", dir=tmp)
        os.close(fd)
        rest.save(still, compress_level=1)
    return seq, still, dur_ms / 1000
