"""Writes caption_templates.json (the caption styles) — the same file goes in both repos."""
import json, sys

FONTS = {
    # key: (family libass finds it by, bold, italic, file, em = unitsPerEm / (winAscent + winDescent), css weight)
    "anton":                       ("Anton", 0, 0, "Anton-Regular.ttf", 0.577, 400),
    "bangers":                     ("Bangers", 0, 0, "Bangers-Regular.ttf", 0.569, 400),
    "bebas":                       ("Bebas Neue", 0, 0, "BebasNeue-Regular.ttf", 0.769, 400),
    "kaushan":                     ("Kaushan Script", 0, 0, "KaushanScript-Regular.ttf", 0.689, 400),
    "lilita":                      ("Lilita One", 0, 0, "LilitaOne-Regular.ttf", 0.875, 400),
    "luckiest":                    ("Luckiest Guy", 0, 0, "LuckiestGuy-Regular.ttf", 0.816, 400),
    "merriweather-bold":           ("Merriweather", 1, 0, "Merriweather-Bold.ttf", 0.577, 700),
    "merriweather-bold-italic":    ("Merriweather", 1, 1, "Merriweather-BoldItalic.ttf", 0.577, 700),
    "montserrat-bold":             ("Montserrat Bold", 1, 0, "Montserrat-Bold.ttf", 0.640, 700),
    "montserrat-black":            ("Montserrat Black", 1, 0, "Montserrat-Black.ttf", 0.640, 900),
    "montserrat-extrabold-italic": ("Montserrat ExtraBold", 1, 1, "Montserrat-ExtraBoldItalic.ttf", 0.640, 800),
    "oswald-bold":                 ("Oswald", 1, 0, "Oswald-Bold.ttf", 0.588, 700),
    "pacifico":                    ("Pacifico", 0, 0, "Pacifico-Regular.ttf", 0.517, 400),
    "poppins":                     ("Poppins", 0, 0, "Poppins-Regular.ttf", 0.568, 400),
    "poppins-extrabold":           ("Poppins ExtraBold", 1, 0, "Poppins-ExtraBold.ttf", 0.568, 800),
    "space-mono-bold":             ("Space Mono", 1, 0, "SpaceMono-Bold.ttf", 0.675, 700),
}
# how big each font is drawn (% of the caption size) so they all read about the same size
SIZE = {"anton": 145, "bangers": 150, "bebas": 125, "kaushan": 110, "lilita": 90, "luckiest": 92,
        "merriweather-bold": 105, "merriweather-bold-italic": 105, "montserrat-bold": 100, "montserrat-black": 100,
        "montserrat-extrabold-italic": 105, "oswald-bold": 130, "pacifico": 125, "poppins": 110,
        "poppins-extrabold": 110, "space-mono-bold": 95}

W, Y, K = "#FFFFFF", "#FFE600", "#111111"

def t(group, id, label, base, font, color=W, hl=Y, per=3, **kw):
    d = {"id": f"tpl:{id}", "label": label, "group": group, "base": base, "font": font,
         "perLine": per, "size": kw.pop("size", SIZE[font]), "color": color, "hl": hl}
    d.update(kw)
    return d

T = [
    # ── Bold: the spoken word changes colour (and maybe size)
    t("Bold", "spotlight", "Spotlight", "plain", "poppins-extrabold", hl="#E4F13A", stroke=3),
    t("Bold", "impact", "Impact", "plain", "anton", hl="#7CFC00", upper=True, stroke=3),
    t("Bold", "sunshine", "Sunshine", "plain", "montserrat-black", hl=Y, upper=True, stroke=0, shadow=4),
    t("Bold", "duo", "Duo", "plain", "montserrat-black", hl="#2BFF3A", upper=True, per=2, stroke=4),
    t("Bold", "cyan", "Cyan", "plain", "bebas", hl="#22E3FF", upper=True, stroke=3),
    t("Bold", "alert", "Alert", "plain", "oswald-bold", hl="#FF4D4D", upper=True, stroke=0, shadow=4),
    t("Bold", "bold-yellow", "Blockbuster", "plain", "anton", hl="#FFD400", upper=True, stroke=5, shadow=5),
    t("Bold", "retro", "Retro", "plain", "montserrat-extrabold-italic", hl="#FFC400", stroke=3, shadow=6),
    t("Bold", "fresh", "Fresh", "plain", "lilita", hl="#2BFF6A", stroke=0, shadow=4),
    t("Bold", "ugc", "UGC", "hormozi", "poppins-extrabold", hl="#FFD60A", per=2, scale=130, stroke=4),
    t("Bold", "bigword", "Big word", "hormozi", "montserrat-extrabold-italic", hl=Y, upper=True, per=2, scale=135, stroke=5),
    t("Bold", "headline", "Headline", "hormozi", "oswald-bold", hl="#FFB020", upper=True, per=2, scale=130, stroke=3),
    t("Bold", "comic", "Comic", "pop", "bangers", hl="#3EE6A0", upper=True, stroke=6),
    t("Bold", "punch", "Punch", "pop", "bangers", hl=Y, upper=True, per=2, stroke=6, shadow=5),
    t("Bold", "toxic", "Toxic", "plain", "bangers", color="#2BFF3A", hl=W, upper=True, per=2, stroke=5, shadow=4),
    t("Bold", "big-one", "One word", "pop", "montserrat-extrabold-italic", hl=W, upper=True, per=1, size=170, stroke=0, shadow=6, liveText=True),

    # ── Clean: no colour change, just a good-looking line
    t("Clean", "clean", "Clean", "plain", "poppins-extrabold", liveText=True, stroke=0, shadow=4),
    t("Clean", "speed", "Speed", "plain", "montserrat-extrabold-italic", liveText=True, upper=True, per=2, stroke=0, shadow=4),
    t("Clean", "italic", "Italic", "plain", "montserrat-extrabold-italic", liveText=True, stroke=0, shadow=4),
    t("Clean", "tall", "Tall", "plain", "oswald-bold", liveText=True, stroke=0, shadow=3),
    t("Clean", "sunny", "Sunny", "plain", "lilita", color=Y, hl=W, liveText=True, stroke=0, shadow=5),
    t("Clean", "weights", "Light & bold", "plain", "poppins", liveFont="poppins-extrabold", liveText=True, stroke=0, shadow=3),

    # ── Box: the line (or the spoken word) sits on a colour
    t("Box", "storybook", "Storybook", "box", "merriweather-bold-italic", color="#FFF7AE", hl="#7C6CF0", box="hl", liveText=True),
    t("Box", "blush", "Blush", "box", "merriweather-bold-italic", color=W, hl="#FF4FD8", box="hl", liveText=True),
    t("Box", "cartoon", "Cartoon", "box", "montserrat-bold", hl="#6D28D9", upper=True, per=2, box="hl", liveText=True),
    t("Box", "code", "Code", "box", "space-mono-bold", hl="#FFE14D"),
    t("Box", "breaking", "Breaking", "box", "anton", color="#FFB8B3", hl=W, upper=True, box="#FF3B30"),
    t("Box", "newspaper", "Newspaper", "box", "merriweather-bold", color=K, hl=K, box="#FFFFFF", liveText=True),
    t("Box", "label", "Label", "box", "montserrat-black", color=K, hl=K, box="#FFFFFF", liveText=True),
    t("Box", "blackout", "Blackout", "box", "montserrat-black", hl="#FFD400", upper=True, box="#000000"),
    t("Box", "pink-chip", "Pink chip", "highlight", "poppins-extrabold", hl="#FF2D8A", box="#1A1A1A", stroke=0),
    t("Box", "aqua-chip", "Aqua chip", "highlight", "poppins-extrabold", hl="#22B8FF", per=2, stroke=3),
    t("Box", "red-chip", "Red chip", "highlight", "poppins-extrabold", hl="#FF2A1F", stroke=3),
    t("Box", "royal-chip", "Royal chip", "highlight", "montserrat-black", hl="#6B21A8", upper=True, stroke=3),
    t("Box", "editorial", "Editorial", "highlight", "merriweather-bold", hl="#4F1DFF", stroke=2),
    t("Box", "green-chip", "Green chip", "highlight", "lilita", color="#2BE04A", hl="#1DB954", stroke=3),
    t("Box", "marker", "Marker", "highlight", "luckiest", hl="#2BE04A", upper=True, per=2, stroke=5),

    # ── Glow: neon
    t("Glow", "soft-glow", "Soft glow", "glow", "poppins-extrabold", color=W, hl=W),
    t("Glow", "neon-green", "Neon green", "glow", "montserrat-extrabold-italic", color="#7CFF3A", hl="#39FF14", upper=True, per=2),
    t("Glow", "neon-red", "Neon red", "glow", "montserrat-black", color="#FF3B3B", hl="#FF0000", upper=True, per=2),
    t("Glow", "neon-yellow", "Neon yellow", "glow", "montserrat-black", color="#FFE94A", hl="#FFD400", upper=True, per=2),
    t("Glow", "lime-glow", "Lime glow", "glow", "anton", color="#C8FF3A", hl="#A6FF00", upper=True, per=2),
    t("Glow", "arcade", "Arcade", "glow", "space-mono-bold", color=W, hl="#FF2D2D", upper=True),
    t("Glow", "sunset", "Sunset", "glow", "poppins-extrabold", color="#FFA726", hl="#FF8A00", per=3, stack=True),

    # ── Script: a handwritten word
    t("Script", "signature", "Signature", "hormozi", "poppins-extrabold", liveFont="pacifico", hl="#FFF04A", scale=125, stroke=0, shadow=4),
    t("Script", "glow-script", "Brush", "hormozi", "poppins-extrabold", liveFont="kaushan", hl="#5CFF4A", scale=130, stroke=0, shadow=4),
    t("Script", "emphasis", "Emphasis", "hormozi", "poppins-extrabold", liveFont="montserrat-extrabold-italic", hl="#C5D82D", scale=150, stroke=0, shadow=4),
    t("Script", "romance", "Romance", "hormozi", "merriweather-bold-italic", liveFont="kaushan", hl="#FF4D4D", scale=140, stroke=0, shadow=3),
    t("Script", "script", "Script", "hormozi", "kaushan", hl="#FFF04A", per=2, scale=135, stroke=0, shadow=4),

    # ── Stacked: one word a line
    t("Stacked", "stack-serif", "Stack serif", "hormozi", "merriweather-bold", hl="#FFE94A", stack=True, scale=125, stroke=0, shadow=4),
    t("Stacked", "stack-bold", "Stack bold", "hormozi", "montserrat-black", hl=W, upper=True, stack=True, scale=140, stroke=0, shadow=4, liveText=True),
    t("Stacked", "pink-stack", "Pink stack", "plain", "montserrat-black", hl="#FF7EB6", upper=True, per=2, stack=True, stroke=4),
    t("Stacked", "shout", "Shout", "hormozi", "poppins-extrabold", hl="#6BFF3A", stack=True, scale=150, stroke=0, shadow=4),
    t("Stacked", "stack-script", "Stack script", "hormozi", "poppins-extrabold", liveFont="kaushan", hl="#5CFF4A", stack=True, scale=135, stroke=0, shadow=4),
]

assert len({x["id"] for x in T}) == len(T)
fonts = {k: {"family": v[0], "bold": v[1], "italic": v[2], "file": v[3], "em": v[4], "weight": v[5]} for k, v in FONTS.items()}
for x in T:
    assert x["font"] in fonts and x.get("liveFont", x["font"]) in fonts, x["id"]
doc = {"_": "Caption styles. The SAME file is in both repos (frontend src/modules/captions, backend src/python): keep them identical. See templates.ts / render.py for what each field does.",
       "fonts": fonts, "templates": T}
for path in sys.argv[1:]:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
        f.write("\n")
print(len(T), "templates,", len(fonts), "fonts")
