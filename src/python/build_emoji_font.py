#!/usr/bin/env python3
"""
Builds the app's emoji font from Twemoji 17 — run once, when updating Twemoji (outputs are committed).

    python build_emoji_font.py <twemoji assets/svg dir> <emoji-test.txt> [<chai-cut-frontend dir>]

Every Twemoji drawing is made of flat-colour shapes. Each run of same-colour shapes becomes one
font glyph ("layer"); an emoji is its layers stacked in order. One font, two uses:

  - Export (this worker): caption renderer (libass) and Pillow can't draw colour fonts, but they
    can draw a plain glyph in any colour. Layers sit at private-use codepoints (U+F0000…), listed
    per emoji with their colours in emoji_layers.json; U+E000 is an empty "spacer" with the same
    width, which holds an emoji's place in a caption line.
  - Preview (the editor, clip board): the same layers as a COLR colour font (with the emoji's own
    codepoints and ligatures for ZWJ sequences, flags, skin tones and keycaps), so the browser
    draws exactly what the export draws.

Twemoji graphics: Copyright 2019 Twitter, Inc and other contributors, CC-BY 4.0
(https://creativecommons.org/licenses/by/4.0/) — the app credits it where emoji are picked.

Outputs: fonts/ChaiEmoji.ttf + emoji_layers.json here; with a frontend dir also
public/fonts/ChaiEmoji.woff2 and public/emoji/emoji-list.json (the picker's list).
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

from fontTools.colorLib.builder import buildCOLR, buildCPAL
from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
from fontTools.fontBuilder import FontBuilder
from fontTools.misc.transform import Transform
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.svgLib.path import parse_path
from fontTools.svgLib.path.shapes import PathBuilder

UPEM = 1000
ADVANCE = 1100            # each emoji: a 1000-unit square with 50 units either side
BOX_LEFT, BOX_TOP = 50, 820   # the square spans y -180…820 (sits on the text like phone emoji)
SPACER = 0xE000
FIRST_LAYER = 0xF0000
NS = "{http://www.w3.org/2000/svg}"
HERE = os.path.dirname(os.path.abspath(__file__))


def _transform(t: str | None) -> Transform:
    m = Transform()
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", t or ""):
        v = [float(x) for x in re.split(r"[\s,]+", args.strip()) if x]
        if name == "matrix":
            m = m.transform(v)
        elif name == "translate":
            m = m.translate(v[0], v[1] if len(v) > 1 else 0)
        elif name == "scale":
            m = m.scale(v[0], v[1] if len(v) > 1 else v[0])
        elif name == "rotate":
            if len(v) == 3:
                m = m.translate(v[1], v[2]).rotate(math.radians(v[0])).translate(-v[1], -v[2])
            else:
                m = m.rotate(math.radians(v[0]))
        elif name == "skewX":
            m = m.skew(math.radians(v[0]), 0)
        elif name == "skewY":
            m = m.skew(0, math.radians(v[0]))
    return m


def _colour(fill: str, opacity: float) -> tuple[str, float] | None:
    fill = fill.strip()
    if not fill or fill == "none":
        return None
    h = fill.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", h):
        h = "000000"   # named colours aren't used by Twemoji
    return "#" + h.upper(), max(0.0, min(1.0, opacity))


def svg_layers(path: str) -> list[tuple[str, float, RecordingPen]]:
    """[(colour, opacity, outline in font units)] in paint order; touching same-colour shapes merged."""
    root = ET.parse(path).getroot()
    vb = [float(x) for x in (root.get("viewBox") or "0 0 36 36").split()]
    to_font = Transform().translate(BOX_LEFT, BOX_TOP).scale(1000 / vb[2], -1000 / vb[3]).translate(-vb[0], -vb[1])
    out: list[tuple[str, float, RecordingPen]] = []

    def walk(el, tf: Transform, fill: str, opacity: float):
        tf = tf.transform(_transform(el.get("transform")))
        style = dict(kv.split(":", 1) for kv in (el.get("style") or "").split(";") if ":" in kv)
        fill = style.get("fill", el.get("fill", fill))
        opacity *= float(style.get("opacity", el.get("opacity", 1)))
        fill_op = float(style.get("fill-opacity", el.get("fill-opacity", 1)))
        tag = el.tag.replace(NS, "")
        if tag in ("path", "circle", "ellipse", "rect", "polygon", "polyline"):
            col = _colour(fill, opacity * fill_op)
            if col:
                pb = PathBuilder()
                # Transforms are applied above (fontTools' own parser knows only matrix())
                shape = ET.Element(el.tag, {k: v for k, v in el.attrib.items() if k != "transform"})
                pb.add_path_from_element(shape)
                rec = RecordingPen()
                for d in pb.paths:
                    parse_path(d, TransformPen(rec, to_font.transform(tf)))
                if rec.value:
                    if out and out[-1][0] == col[0] and out[-1][1] == col[1]:
                        out[-1][2].value.extend(rec.value)
                    else:
                        out.append((col[0], col[1], rec))
        for ch in el:
            if ch.tag.replace(NS, "") not in ("defs", "clipPath", "mask", "title", "desc"):
                walk(ch, tf, fill, opacity)

    walk(root, Transform(), "#000000", 1.0)
    return out


def key_of(seq: str) -> str:
    """An emoji's lookup key: its codepoints in hex, joined by '-', without FE0F"""
    return "-".join(f"{ord(c):x}" for c in seq if c != "️")


def glyph_name(cps: list[int]) -> str:
    return "_".join(f"u{c:04X}" for c in cps)


def build_spaces() -> dict[str, str]:
    """
    Per caption font: the mix of its own space characters closest to an emoji's width (1.1 em).
    Box-style captions draw their dark box from an invisible copy of the line; an emoji there is
    replaced by these spaces, not the emoji font's spacer, because switching fonts splits the box
    into overlapping pieces (darker strips). Written to emoji_spaces.json.
    """
    from itertools import combinations_with_replacement
    from fontTools.ttLib import TTFont
    target = ADVANCE / UPEM
    spaces = [0x20, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2009, 0x200A]
    out: dict[str, str] = {}
    for f in sorted(os.listdir(os.path.join(HERE, "fonts"))):
        if not f.endswith((".ttf", ".otf")) or f.startswith("ChaiEmoji"):
            continue
        t = TTFont(os.path.join(HERE, "fonts", f), lazy=True)
        cm, hm, up = t.getBestCmap(), t["hmtx"].metrics, t["head"].unitsPerEm
        have = {c: hm[cm[c]][0] / up for c in spaces if c in cm}
        best = min((combo for n in range(1, 7) for combo in combinations_with_replacement(sorted(have), n)),
                   key=lambda combo: (abs(sum(have[c] for c in combo) - target), len(combo)))
        # U+0020 becomes the no-break space so the line can't wrap there
        out[t["name"].getDebugName(1)] = "".join(chr(0xA0) if c == 0x20 else chr(c) for c in best)
    with open(os.path.join(HERE, "emoji_spaces.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=True)
    return out


def main(svg_dir: str, emoji_test: str, frontend: str | None):
    files = sorted(f for f in os.listdir(svg_dir) if f.endswith(".svg"))
    glyph_order = [".notdef", "spacer", "fe0f", "zwj"]
    pen = lambda: TTGlyphPen(None)
    glyphs = {g: pen().glyph() for g in glyph_order}
    metrics = {".notdef": (ADVANCE, 0), "spacer": (ADVANCE, 0), "fe0f": (0, 0), "zwj": (0, 0)}
    cmap = {SPACER: "spacer", 0xFE0F: "fe0f", 0x200D: "zwj"}
    colr: dict[str, list[tuple[str, int]]] = {}
    palette: list[tuple[str, float]] = []
    pal_index: dict[tuple[str, float], int] = {}
    layer_map: dict[str, list[list]] = {}
    ligatures: list[tuple[list[str], str]] = []
    next_cp = FIRST_LAYER
    components: set[int] = set()
    skipped = 0

    for f in files:
        cps = [int(x, 16) for x in f[:-4].split("-") if x.lower() != "fe0f"]
        layers = svg_layers(os.path.join(svg_dir, f))
        if not layers:
            skipped += 1
            continue
        base = glyph_name(cps)
        glyph_order.append(base)
        glyphs[base] = pen().glyph()           # the colour comes from COLR (preview) / layers (export)
        metrics[base] = (ADVANCE, 0)
        entry, colr_layers = [], []
        for i, (col, op, rec) in enumerate(layers):
            g = f"{base}.l{i}"
            tt = pen()
            rec.replay(Cu2QuPen(tt, max_err=1.0, reverse_direction=True))
            glyphs[g] = tt.glyph()
            metrics[g] = (ADVANCE, 0)
            glyph_order.append(g)
            cmap[next_cp] = g
            if (col, op) not in pal_index:
                pal_index[(col, op)] = len(palette)
                palette.append((col, op))
            colr_layers.append((g, pal_index[(col, op)]))
            entry.append([next_cp, col] + ([round(op, 3)] if op < 1 else []))
            next_cp += 1
        colr[base] = colr_layers
        layer_map[key_of("".join(chr(c) for c in cps))] = entry
        if len(cps) == 1:
            cmap[cps[0]] = base
        else:
            components.update(cps)
            ligatures.append((cps, base))

    if next_cp > 0xFFFFD:
        raise SystemExit(f"too many layers for plane 15: {next_cp - FIRST_LAYER}")
    # Codepoints only seen inside sequences (e.g. regional indicators, U+20E3 keycap, tag letters)
    for c in sorted(components):
        if c not in cmap:
            g = glyph_name([c])
            glyph_order.append(g)
            glyphs[g] = pen().glyph()
            metrics[g] = (0, 0)
            cmap[c] = g

    fb = FontBuilder(UPEM, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyphs)
    # Each glyph's left side bearing must be its own left edge: renderers place a TrueType glyph by
    # it, so a wrong value shifts that layer away from the others
    glyf = fb.font["glyf"]
    for g in glyph_order:
        gl = glyf[g]
        gl.recalcBounds(glyf)
        metrics[g] = (metrics[g][0], getattr(gl, "xMin", 0))
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=950, descent=-250)
    fb.setupNameTable({"familyName": "Chai Emoji", "styleName": "Regular",
                       "copyright": "Twemoji graphics (c) Twitter, Inc and other contributors, CC-BY 4.0"})
    fb.setupOS2(sTypoAscender=950, sTypoDescender=-250, sTypoLineGap=0, usWinAscent=950, usWinDescent=250)
    fb.setupPost()
    font = fb.font
    # Ligatures for sequences (longest first); FE0F is a mark the lookup skips, so a sequence matches
    # with or without it
    ligatures.sort(key=lambda x: -len(x[0]))
    rules = "\n".join(f"    sub {' '.join(cmap[c] for c in cps)} by {base};" for cps, base in ligatures)
    fea = f"""
@MARKS = [fe0f];
table GDEF {{ GlyphClassDef , , @MARKS, ; }} GDEF;
lookup EMOJI {{
    lookupflag IgnoreMarks;
{rules}
}} EMOJI;
feature ccmp {{ lookup EMOJI; }} ccmp;
feature liga {{ lookup EMOJI; }} liga;
"""
    addOpenTypeFeaturesFromString(font, fea)
    font["COLR"] = buildCOLR(colr, version=0)
    font["CPAL"] = buildCPAL([[(int(c[1:3], 16) / 255, int(c[3:5], 16) / 255, int(c[5:7], 16) / 255, op) for c, op in palette]])

    os.makedirs(os.path.join(HERE, "fonts"), exist_ok=True)
    font.save(os.path.join(HERE, "fonts", "ChaiEmoji.ttf"))
    with open(os.path.join(HERE, "emoji_layers.json"), "w", encoding="utf-8") as fh:
        json.dump(layer_map, fh, separators=(",", ":"))
    build_spaces()
    print(f"{len(layer_map)} emoji, {next_cp - FIRST_LAYER} layers, {len(palette)} colours, "
          f"{len(ligatures)} sequences ({skipped} drawings without shapes skipped)")

    if frontend:
        os.makedirs(os.path.join(frontend, "public", "fonts"), exist_ok=True)
        font.flavor = "woff2"
        font.save(os.path.join(frontend, "public", "fonts", "ChaiEmoji.woff2"))
        # The picker: Unicode's groups and names, only emoji Twemoji draws, skin-tone variants left out
        groups: list[dict] = []
        group = None
        for line in open(emoji_test, encoding="utf-8"):
            if line.startswith("# group:"):
                group = {"g": line.split(":", 1)[1].strip(), "e": []}
                if group["g"] != "Component":
                    groups.append(group)
                continue
            m = re.match(r"^([0-9A-F ]+);\s*fully-qualified\s*#\s*(\S+)\s+E[\d.]+\s+(.+)$", line.strip())
            if not m or group is None or group["g"] == "Component":
                continue
            cps = [int(x, 16) for x in m.group(1).split()]
            if any(0x1F3FB <= c <= 0x1F3FF for c in cps):
                continue
            seq = "".join(chr(c) for c in cps)
            if key_of(seq) in layer_map:
                group["e"].append([seq, m.group(3)])
        groups = [g for g in groups if g["e"]]
        os.makedirs(os.path.join(frontend, "public", "emoji"), exist_ok=True)
        with open(os.path.join(frontend, "public", "emoji", "emoji-list.json"), "w", encoding="utf-8") as fh:
            json.dump(groups, fh, ensure_ascii=False, separators=(",", ":"))
        print(f"picker: {sum(len(g['e']) for g in groups)} emoji in {len(groups)} groups")


if __name__ == "__main__":
    if sys.argv[1:] == ["--spaces"]:      # only the caption fonts' emoji-wide spaces
        print(build_spaces())
        sys.exit(0)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
