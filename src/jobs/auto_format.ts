// ── Auto format: which format a place gets (Make my clips) ─────────────────────
// A COPY of the editor's modules/editor/autoFormat.ts (frontend) — keep them in step: the same rules,
// numbers and tests, so Make my clips lays every scene as the Subjects tool's Auto format (Cuts) would.
// Only the imports differ: the slot shapes and the Vertical crop the editor takes from its utils.ts are
// at the end of this file.

type LayoutType = 'vertical' | 'split' | 'trio' | 'frame_blur_fill'
type BoxPosition = { x: number; y: number; w: number; h: number }

// The Rulebook ("Auto Format Framing Spec", 2026-10-25; the user: "follow the rulebook"). For each
// place (one stretch where a scene's shot comes back), the first rule that fits decides:
//   0. No people found: the place is left as it is (the caller never asks).
//   1. Everyone fits one 9:16 crop with FIT_MARGIN room either side (always true for one person): Vertical.
//   2. Exactly 2 people: Split — the speaker at the bottom, the other on top.
//   3. Exactly 3 people: Trio in seat order (left, middle, right = top, middle, bottom).
//   4. 4 or more who form exactly 3 groups (1 + 2 + 1, 2 + 1 + 2…): Trio, one group per view.
//   5. 4 or more, any other seating: Split into two sides — cut at the break between two groups, else at
//      the gap nearest the middle — the speaker's side at the bottom.
// The format comes from the people and how they sit; the speaker only orders a Split's views. Speaker:
// whoever talks most in the place with at least SPEAKER_MIN of it, ahead of everyone else; nobody (no
// transcript, crosstalk): left on top. Groups: the gaps between neighbours' face boxes in face widths — a
// break is at least BREAK_RATIO × the tightest gap and at least BREAK_MIN. Split and Trio stack their
// views top to bottom; each view is framed here (section 5 of the spec); Follow keeps a one-person view
// on its person, a view on 2+ people holds.
// Never lose a subject (the user, 2026-10-25 — over the Rulebook's tiny-face rule, which put a wide shot
// of seven in one Vertical slice and cut four of them out): small faces go by the rules above too, and
// a Split or Trio must hold every face whole in a view within MAX_ZOOM; when it can't (people spread wider
// than its views can show), the place shows the whole picture — Blur fill.

/**
 * Every threshold, under the Rulebook's names and defaults (its section 7): tune here and rerun the test
 * log (autoFormat.test.ts). PRESENCE_MIN is applied by the scan (backend track_subjects.ts findAngles),
 * SHORT_PLACE by Auto format in EditorShell, MIN_HOLD by its Vertical mode's Follow; SWAP_ON_TURN (off)
 * and FOLLOW_SMOOTH are not built yet.
 */
export const AUTO_FORMAT = {
  /** Who counts as in the frame: their face in this share of the place's frames (the scan) */
  PRESENCE_MIN: 0.5,
  /** A group fits one Vertical view with this share of its span as room either side */
  FIT_MARGIN: 0.15,
  /** A gap is a break between groups at this many times the tightest gap… */
  BREAK_RATIO: 1.5,
  /** …and at least this many face widths */
  BREAK_MIN: 1.0,
  /** The speaker has at least this share of the place's talking */
  SPEAKER_MIN: 0.2,
  /** A Split swaps top and bottom when the speaker changes (not built: the order is fixed per place) */
  SWAP_ON_TURN: false,
  /** Seconds the view stays on one person before it cuts to the next speaker (Vertical mode's Follow) */
  MIN_HOLD: 2.0,
  /** A place shorter than this (seconds) takes the format of the place before it */
  SHORT_PLACE: 1.0,
  /** A one-person view's height in face heights: Split, Trio */
  SPLIT_FACE_H: 3.2,
  TRIO_FACE_H: 2.6,
  /** Room around a group in one view: this share of its span either side */
  GROUP_PAD: 0.15,
  /** The face centre this far down its view */
  FACE_Y: 0.4,
  /** The view's top at least this many face heights above the forehead (wins over FACE_Y) */
  HEADROOM_MIN: 0.5,
  /** A view on people one above another (a collage's rows) reaches this many face heights below the lowest chin */
  CHIN_ROOM: 0.3,
  /** The tightest crop: face heights tall, face widths wide */
  MIN_VIEW_H: 1.55,
  MIN_VIEW_W: 1.6,
  /** The largest upscale: a crop at least 1 / MAX_ZOOM of its output view's pixels */
  MAX_ZOOM: 2.0,
  /** How fast a view follows its person (seconds to settle; not wired yet) */
  FOLLOW_SMOOTH: 0.4,
} as const

/** Output view heights in pixels (1080 × 1920): Vertical, Split, Trio */
const VIEW_PX_H: Partial<Record<LayoutType, number>> = { vertical: 1920, split: 960, trio: 640 }
/**
 * The source's height in pixels when not known: the editor only has the editing copy (720p at most) and
 * the export reads the original — the Rulebook's own example is a 1080p source
 */
export const SOURCE_PX_H = 1080

type Face = { x: number; y: number; w: number; h: number }
export interface AutoLayout {
  layout: LayoutType
  framing: BoxPosition[]
  /** Which rule decided it (for the notice and the tests); 'whole': no Split or Trio could hold everyone;
   *  'speaker' / 'biggest': the Vertical mode (decideVertical) on the speaker, or nobody talks */
  why: 'fits' | 'two' | 'three' | 'groups3' | 'sides' | 'whole' | 'speaker' | 'biggest'
}

/** The whole picture, nobody cropped out: the wide video on a blurred copy of itself */
export const WHOLE_PICTURE: LayoutType = 'frame_blur_fill'

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))
const midX = (f: Face) => f.x + f.w / 2
const midY = (f: Face) => f.y + f.h / 2

/** A strip of the picture a view keeps to: [left, right] (fractions of the width) */
type Strip = readonly [number, number]

/** The smallest view height the upscale limit allows for a view of this layout (fractions of the picture) */
const minZoomH = (layout: LayoutType, sourcePxH: number) => (VIEW_PX_H[layout] ?? 0) / AUTO_FORMAT.MAX_ZOOM / sourcePxH

/**
 * How high and how low a view on `group` reaches (fractions of the height): HEADROOM_MIN above the highest
 * forehead, and below the lowest chin — `room`: CHIN_ROOM more, when the strip has space for it — people one
 * above another (the user, 2026-10-26: a collage of four interviews, two rows, got views sized by width only
 * that cut the bottom row's chins, so every Split lost someone and the place fell to the whole picture)
 */
function reach(group: readonly Face[], room = false) {
  const tallest = Math.max(...group.map(f => f.h))
  const top = Math.max(0, Math.min(...group.map(f => f.y)) - AUTO_FORMAT.HEADROOM_MIN * tallest)
  const bottom = Math.min(1, Math.max(...group.map(f => f.y + f.h * (1 + (room ? AUTO_FORMAT.CHIN_ROOM : 0)))))
  return [top, bottom] as const
}

/** The smallest view height for `group` in a view of this layout: the tightness and upscale limits, and the group's own span (across and down; `room`: with CHIN_ROOM) */
function leastH(layout: LayoutType, a: number, group: readonly Face[], sourcePxH: number, room = false) {
  const left = Math.min(...group.map(f => f.x)), right = Math.max(...group.map(f => f.x + f.w))
  const [top, bottom] = reach(group, room)
  return Math.max(Math.max(...group.map(f => f.h)) * AUTO_FORMAT.MIN_VIEW_H, (Math.max(...group.map(f => f.w)) * AUTO_FORMAT.MIN_VIEW_W) / a,
    (right - left) / a, minZoomH(layout, sourcePxH), group.length > 1 ? bottom - top : 0)
}

/**
 * The strips of side-by-side groups (left to right), so the views of a Split or Trio never overlap (the
 * user, 2026-10-25: a close two-shot gave two near full-height views that both showed the middle quarter
 * of the picture). A boundary sits in the gap between two groups — halfway, unless a view needs more room
 * than that gives it (a person at the picture's edge), then as far as the gap allows; the picture's edges at
 * the ends. `minWidths`: the narrowest each view may be. Null when no boundaries give every view its room.
 */
export function strips(groupsLR: readonly (readonly Face[])[], minWidths: readonly number[]): Strip[] | null {
  const k = groupsLR.length
  const ends = groupsLR.map(g => [Math.min(...g.map(f => f.x)), Math.max(...g.map(f => f.x + f.w))] as const)
  // Boundary i (between views i and i + 1) lies in the gap between their groups
  const lo = ends.slice(0, -1).map(e => e[1]), hi = ends.slice(1).map(e => e[0])
  const b = lo.map((l, i) => clamp((l + hi[i]) / 2, l, Math.max(l, hi[i])))
  const at = (j: number) => (j === 0 ? 0 : j === k ? 1 : b[j - 1])
  // Room for each view: left to right, then right to left
  for (let j = 1; j < k; j++) if (at(j) - at(j - 1) < minWidths[j - 1]) b[j - 1] = Math.min(hi[j - 1], at(j - 1) + minWidths[j - 1])
  for (let j = k; j >= 2; j--) if (at(j) - at(j - 1) < minWidths[j - 1]) b[j - 2] = Math.max(lo[j - 2], at(j) - minWidths[j - 1])
  const out = groupsLR.map((_, j) => [at(j), at(j + 1)] as const)
  const ok = out.every(([l, r], j) => r - l >= minWidths[j] - 1e-9 && l <= ends[j][0] + 1e-9 && r >= ends[j][1] - 1e-9)
  return ok ? out : null
}

/**
 * A view on `group` (one person, or several) inside `strip`: as tall as the layout wants (SPLIT_FACE_H /
 * TRIO_FACE_H face heights, or the group's span with GROUP_PAD either side), never shorter than the
 * tightness and upscale limits, never wider than its strip; centred on the group, sliding in at the
 * strip's (or picture's) edge, the face FACE_Y down unless the headroom asks for more. Null when the strip
 * is too narrow for the smallest view the limits allow.
 */
function viewIn(layout: LayoutType, a: number, group: readonly Face[], strip: Strip, sourcePxH: number): BoxPosition | null {
  const left = Math.min(...group.map(f => f.x)), right = Math.max(...group.map(f => f.x + f.w))
  const tallest = Math.max(...group.map(f => f.h))
  const faceH = layout === 'trio' ? AUTO_FORMAT.TRIO_FACE_H : AUTO_FORMAT.SPLIT_FACE_H
  const [above, below] = reach(group, true)
  const want = group.length === 1 ? tallest * faceH : Math.max(((right - left) * (1 + 2 * AUTO_FORMAT.GROUP_PAD)) / a, tallest * faceH, below - above)
  const least = leastH(layout, a, group, sourcePxH)
  // The tallest the strip and the picture allow
  const most = Math.min(1, 1 / a, (strip[1] - strip[0]) / a)
  if (least > most + 1e-9) return null
  const h = clamp(want, least, most), w = a * h
  const cx = (left + right) / 2, cy = group.reduce((t, f) => t + midY(f), 0) / group.length
  const top = Math.min(...group.map(f => f.y))
  // (several people: low enough to reach the lowest chin, with CHIN_ROOM as far as the headroom allows)
  const low = group.length > 1 ? Math.min(below - h, above) : 0
  return {
    x: clamp(cx - w / 2, strip[0], strip[1] - w),
    y: clamp(Math.max(low, Math.min(cy - h * AUTO_FORMAT.FACE_Y, top - AUTO_FORMAT.HEADROOM_MIN * tallest)), 0, 1 - h),
    w, h,
  }
}

/** The gaps between neighbours (left to right), each in face widths: the space between their boxes / their mean width */
export function seatGaps(byX: readonly Face[]): number[] {
  return byX.slice(1).map((g, i) => (g.x - (byX[i].x + byX[i].w)) / ((g.w + byX[i].w) / 2))
}

/** Where the groups break: the indexes i where a break falls between person i - 1 and i */
export function seatBreaks(gaps: readonly number[]): number[] {
  if (!gaps.length) return []
  const tightest = Math.max(0, Math.min(...gaps))
  return gaps.flatMap((g, k) => (g >= AUTO_FORMAT.BREAK_RATIO * tightest && g >= AUTO_FORMAT.BREAK_MIN ? [k + 1] : []))
}

/**
 * The format for one place. `faces`: its people (fractions of the picture; any order); `talk`: how long
 * each talks there (ms, the same order), or null when not known; `videoAR`: the picture's width / height
 * (16:9 when not known); `sourcePxH`: its height in pixels (the upscale limit). Null with no faces (rule 0).
 */
export function decideFormat(faces: readonly Face[], talk: readonly number[] | null, videoAR?: number, sourcePxH = SOURCE_PX_H): AutoLayout | null {
  if (!faces.length) return null
  // Left to right: A, B, C… each with how much they talk
  const order = faces.map((f, i) => ({ f, t: talk?.[i] ?? 0 })).sort((p, q) => midX(p.f) - midX(q.f))
  const byX = order.map(o => o.f)
  const n = byX.length
  const speaker = speakerOf(order.map(o => o.t))
  const aSplit = normalizedSlotAspect('split', videoAR) ?? 1, aTrio = normalizedSlotAspect('trio', videoAR) ?? 1
  const vertical = (cx: number, why: AutoLayout['why']): AutoLayout => ({ layout: 'vertical', framing: [defaultCropForSlot('vertical', 0, videoAR, cx)], why })
  const whole: AutoLayout = { layout: WHOLE_PICTURE, framing: [], why: 'whole' }
  /** Views on side-by-side groups (left to right), each in its own strip; null when one can't be made */
  const viewsFor = (layout: LayoutType, a: number, groupsLR: Face[][]) => {
    // (room under the chins when the gaps allow it, else just the faces)
    const widths = (room: boolean) => groupsLR.map(g => a * Math.min(leastH(layout, a, g, sourcePxH, room), 1))
    const st = strips(groupsLR, widths(true)) ?? strips(groupsLR, widths(false))
    if (!st) return null
    const views = groupsLR.map((g, i) => viewIn(layout, a, g, st[i], sourcePxH))
    return views.every(Boolean) ? (views as BoxPosition[]) : null
  }
  /** Split into two sides (left: byX before `cut`): the speaker's side at the bottom; no speaker: left on top */
  const sides = (cut: number, why: AutoLayout['why']): AutoLayout | null => {
    const views = viewsFor('split', aSplit, [byX.slice(0, cut), byX.slice(cut)])
    if (!views) return null
    return { layout: 'split', why, framing: speaker !== null && speaker < cut ? [views[1], views[0]] : views }
  }
  /** Trio: the groups left to right become the views top to bottom */
  const trio = (groupsLR: Face[][], why: AutoLayout['why']): AutoLayout | null => {
    const views = viewsFor('trio', aTrio, groupsLR)
    return views ? { layout: 'trio', why, framing: views } : null
  }
  /** Every face whole inside some view */
  const keepsEveryone = (r: AutoLayout) => byX.every(f => r.framing.some(v =>
    f.x >= v.x - 1e-6 && f.x + f.w <= v.x + v.w + 1e-6 && f.y >= v.y - 1e-6 && f.y + f.h <= v.y + v.h + 1e-6))
  /** The Split or Trio if it could be made and keeps everyone; else the whole picture */
  const unlessLost = (r: AutoLayout | null): AutoLayout => (r && keepsEveryone(r) ? r : whole)
  /** Two sides cut at the gap nearest the middle (the sides differ by one person at most; with an odd
   *  count, the wider of the two middle gaps) */
  const middleCut = () => {
    const gaps = seatGaps(byX)
    const middles = n % 2 === 0 ? [n / 2] : [(n - 1) / 2, (n + 1) / 2]
    return middles.reduce((b, c) => (gaps[c - 1] > gaps[b - 1] ? c : b))
  }
  /** Two sides at `cut` — or, when that can't keep everyone sharp (a person alone at the picture's edge
   *  with too narrow a strip), at the other cuts, nearest the middle first; only then the whole picture */
  const sidesAt = (cut: number, why: AutoLayout['why']): AutoLayout => {
    const others = Array.from({ length: n - 1 }, (_, i) => i + 1).filter(c => c !== cut).sort((p, q) => Math.abs(p - n / 2) - Math.abs(q - n / 2))
    for (const c of [cut, ...others]) {
      const r = sides(c, why)
      if (r && keepsEveryone(r)) return r
    }
    return whole
  }
  /** A Trio that can't keep everyone sharp in its strips (too close for three views): two sides instead —
   *  the simpler format (the Rulebook, section 6) — and only then the whole picture */
  const trioOrSides = (r: AutoLayout | null, cut: number): AutoLayout => (r && keepsEveryone(r) ? r : sidesAt(cut, 'sides'))

  // 1. Everyone fits one 9:16 crop, with FIT_MARGIN room either side (always true for one person)
  const vw = defaultCropForSlot('vertical', 0, videoAR).w
  const left = Math.min(...byX.map(f => f.x)), right = Math.max(...byX.map(f => f.x + f.w))
  const fits = n === 1 || (right - left) * (1 + 2 * AUTO_FORMAT.FIT_MARGIN) <= vw
  if (fits) return vertical((left + right) / 2, 'fits')
  // 2. Exactly two: the speaker at the bottom
  if (n === 2) return sidesAt(1, 'two')
  // 3. Exactly three: Trio in seat order, wherever the speaker sits
  if (n === 3) return trioOrSides(trio(byX.map(f => [f]), 'three'), middleCut())
  // 4. Four or more in exactly three groups: Trio, one group a view
  const gaps = seatGaps(byX)
  const breaks = seatBreaks(gaps)
  if (breaks.length === 2) {
    // (two sides at the wider of the two breaks, when the Trio can't be made)
    const wider = gaps[breaks[0] - 1] >= gaps[breaks[1] - 1] ? breaks[0] : breaks[1]
    return trioOrSides(trio([byX.slice(0, breaks[0]), byX.slice(breaks[0], breaks[1]), byX.slice(breaks[1])], 'groups3'), wider)
  }
  // 5. Any other seating: two sides — at the break between two groups, else at the gap nearest the
  //    middle (the sides differ by one person at most; with an odd count, the wider of the two middle gaps)
  if (breaks.length === 1) return sidesAt(breaks[0], 'sides')
  return sidesAt(middleCut(), 'sides')
}

/** The speaker (index into `talk`): the most talk, at least SPEAKER_MIN of it, ahead of everyone; null — two level (crosstalk), or nobody talks */
function speakerOf(talk: readonly number[]): number | null {
  const total = talk.reduce((s, t) => s + t, 0)
  const ranked = talk.map((t, i) => ({ i, t })).sort((p, q) => q.t - p.t)
  return total > 0 && ranked[0].t / total >= AUTO_FORMAT.SPEAKER_MIN && (ranked.length === 1 || ranked[0].t > ranked[1].t) ? ranked[0].i : null
}

/**
 * Auto format's Vertical mode (the user, 2026-10-28: "when we select Vertical, the speaker gets the
 * priority rather than the people in it"): every place gets one 9:16 view — on everyone when they fit
 * one (rule 1's test), else on the place's speaker, else (nobody talks there) on the biggest face. Follow
 * then cuts the view to whoever talks, turn by turn, each held at least MIN_HOLD (EditorShell
 * frameClipOnSubjects). Listeners may be out of the picture here: the user's choice of this mode, over
 * never ignore anyone (which the Cuts mode, decideFormat, keeps). Null with no faces.
 */
export function decideVertical(faces: readonly Face[], talk: readonly number[] | null, videoAR?: number): AutoLayout | null {
  if (!faces.length) return null
  const vertical = (cx: number, why: AutoLayout['why']): AutoLayout => ({ layout: 'vertical', framing: [defaultCropForSlot('vertical', 0, videoAR, cx)], why })
  const vw = defaultCropForSlot('vertical', 0, videoAR).w
  const left = Math.min(...faces.map(f => f.x)), right = Math.max(...faces.map(f => f.x + f.w))
  if (faces.length === 1 || (right - left) * (1 + 2 * AUTO_FORMAT.FIT_MARGIN) <= vw) return vertical((left + right) / 2, 'fits')
  const speaker = talk ? speakerOf(faces.map((_, i) => talk[i] ?? 0)) : null
  if (speaker !== null) return vertical(midX(faces[speaker]), 'speaker')
  const biggest = faces.reduce((b, f) => (f.w * f.h > b.w * b.h ? f : b))
  return vertical(midX(biggest), 'biggest')
}

const within = (f: Face, v: BoxPosition) =>
  f.x >= v.x - 1e-6 && f.x + f.w <= v.x + v.w + 1e-6 && f.y >= v.y - 1e-6 && f.y + f.h <= v.y + v.h + 1e-6

/**
 * Whether a format keeps everyone (the user, 2026-10-26: "never ever ignore any person in the view"):
 * every face whole in a view — in a Vertical, a face wider than the 9:16 crop itself has its middle in it;
 * the whole picture keeps everyone
 */
export function keepsAll(layout: LayoutType, framing: readonly BoxPosition[], faces: readonly Face[]): boolean {
  if (layout === WHOLE_PICTURE) return true
  if (!framing.length) return false
  return faces.every(f => (layout === 'vertical' && f.w > framing[0].w
    ? f.x + f.w / 2 >= framing[0].x && f.x + f.w / 2 <= framing[0].x + framing[0].w
    : framing.some(v => within(f, v))))
}

/** Whether any two views share part of the picture (the user, 2026-10-26: views never overlap, in any format) */
export function viewsOverlap(framing: readonly BoxPosition[]): boolean {
  return framing.some((p, a) => framing.some((q, b) => a < b
    && Math.min(p.x + p.w, q.x + q.w) - Math.max(p.x, q.x) > 1e-6 && Math.min(p.y + p.h, q.y + q.h) - Math.max(p.y, q.y) > 1e-6))
}

// ── From the editor's modules/editor/utils.ts (slot shapes for an export of 1080 × 1920) ──

const DEFAULT_SOURCE_AR = 16 / 9
/** One slot's shape (w / h) in fractions of the source picture: Vertical 9:16, Split 9:8, Trio 27:16; the whole picture: none */
function normalizedSlotAspect(layout: LayoutType, videoAR = DEFAULT_SOURCE_AR): number | null {
  const px = layout === 'split' ? 1080 / 960 : layout === 'trio' ? 1080 / 640 : layout === 'vertical' ? 1080 / 1920 : null
  return px === null ? null : px / (videoAR || DEFAULT_SOURCE_AR)
}
/** A full-height Vertical crop centred on `centerX` (the editor's defaultCropForSlot for 'vertical'), inside the picture */
function defaultCropForSlot(_layout: 'vertical', _slot: number, videoAR = DEFAULT_SOURCE_AR, centerX?: number): BoxPosition {
  const a = normalizedSlotAspect('vertical', videoAR)!
  let w = a, h = 1
  if (w > 1) { w = 1; h = 1 / a }
  return { x: clamp((centerX ?? 0.5) - w / 2, 0, 1 - w), y: clamp(0.5 - h / 2, 0, 1 - h), w, h }
}
