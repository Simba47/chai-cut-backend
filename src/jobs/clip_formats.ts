// ── Make my clips: formats by the Rulebook (Auto format, Cuts) ────────────────
// The user, 2026-10-29: "i want the same mechanism as auto format (cuts) in make my clips". A test on
// two 3-minute QA clips (7 and 9 people) had Auto format keep everyone whole 98.5–99.8% of the time
// against 75–82% for the old framing (buildSegments), with the talker on screen 100% against 74–95%.
// So a clip is framed as the Subjects tool's Auto format would frame it, then followed:
//   1. The scan (track_subjects.ts findAngles, after withoutPictures takes out posters and photos): the clip's scenes, each with its people (faces read in
//      people mode, who is who) and how long each talks in every place the scene comes back.
//   2. Every place gets its format (auto_format.ts decideFormat, the editor's rules); a place under
//      SHORT_PLACE takes the one before it when that still keeps everyone (EditorShell autoFormat).
//   3. Each place is its own section, cut on the camera cuts; stretches no scene covers (no faces) get
//      a still centred Vertical, as the old framing's cutaways.
//   4. Follow (the editor's Cuts mode): a view on one person keeps on them as they move, inside its strip
//      of the picture (never over a neighbour's view); a view on two or more holds. Motion off: all hold.

import { trackFaces, stabilise, FRAME_INTERVAL_MS, type FrameDetection, type CropBox, type SpokenWord } from './ai_edit.js'
import { findAngles, whoIs, withoutPictures, FRAME_SHOWS_MS } from './track_subjects.js'
import { decideFormat, keepsAll, AUTO_FORMAT, type AutoLayout } from './auto_format.js'

interface Kf { t_ms: number; x: number; y: number; w: number; h: number }
export interface FormattedPart {
  start_ms: number; end_ms: number
  layout: 'vertical' | 'split' | 'trio' | 'frame_blur_fill'
  slotKfs: Kf[][]
  /** How many people its views show (the old framing's panels, visuals and B-roll go only over one) */
  people: number
  /** Which rule gave the format ('cutaway': no faces there) */
  why: AutoLayout['why'] | 'cutaway'
}

/** A stretch shorter than this isn't a place of its own (angles.ts MIN_SHOWN_MS) */
const MIN_PLACE_MS = 200
/** A followed face is the one nearest where the person was, at most this far across (fractions of the width)… */
const FOLLOW_REACH = 0.15
/** …and of about their size */
const FOLLOW_SIZE = [0.6, 1.6] as const

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))
const within = (f: CropBox, v: Omit<Kf, 't_ms'>) => f.x >= v.x - 1e-6 && f.x + f.w <= v.x + v.w + 1e-6 && f.y >= v.y - 1e-6 && f.y + f.h <= v.y + v.h + 1e-6

/**
 * The clip's sections in the Rulebook's formats. `frames`: faces 4× a second in people mode (detectFaces
 * with people = true); `cuts`: camera cuts (ms from the clip's
 * start); `speech`: the clip's words (clip ms); `size`: the source's pixels.
 */
export function formatClip(
  frames: FrameDetection[], cuts: number[], durationMs: number, speech: SpokenWord[],
  size: { width: number; height: number }, opts: { motion?: boolean } = {},
): FormattedPart[] {
  const ar = size.width / size.height
  frames = withoutPictures(frames, cuts)
  const angles = findAngles(frames, cuts, durationMs, speech)
  // Who each face is: a scene's people are where it was first seen, and someone can sit elsewhere in a
  // place it comes back (another take of the same angle): Follow finds them by who they are
  const who = opts.motion === false ? new Map<CropBox, number>() : whoIs(frames, cuts)
  type Decided = { a: number; b: number; faces: CropBox[]; plan: AutoLayout }
  const decided: Decided[] = angles.filter(g => g.people > 0).flatMap(g => g.spans.flatMap(([s, e], i): Decided[] => {
    const a = Math.max(0, Math.round(s)), b = Math.min(durationMs, Math.round(e))
    if (b - a < MIN_PLACE_MS) return []
    const faces = g.faces as CropBox[]
    const plan = decideFormat(faces, g.talk?.[i] ?? null, ar, size.height)
    return plan ? [{ a, b, faces, plan }] : []
  })).sort((p, q) => p.a - q.a)
  // A very short place takes the format of the place right before it (no flicker on fast cutting), when
  // that format holds everyone here too
  const places = decided.map((x, i) => {
    const before = i > 0 ? decided[i - 1] : null
    const short = x.b - x.a < AUTO_FORMAT.SHORT_PLACE * 1000
    return short && before && Math.abs(before.b - x.a) <= 150 && keepsAll(before.plan.layout, before.plan.framing, x.faces)
      ? { ...x, plan: before.plan } : x
  })

  const reelW = Math.min(1, (9 / 16) / ar)
  const cutaway = (a: number, b: number): FormattedPart => ({
    start_ms: a, end_ms: b, layout: 'vertical', people: 0, why: 'cutaway',
    slotKfs: [[{ t_ms: a, x: (1 - reelW) / 2, y: 0, w: reelW, h: 1 }]],
  })
  const out: FormattedPart[] = []
  let at = 0
  for (const p of places) {
    const a = Math.max(p.a, at)
    if (a - at >= 50) out.push(cutaway(at, a))
    if (p.b - a < 50) continue
    const views = p.plan.layout === 'frame_blur_fill' ? [{ x: 0, y: 0, w: 1, h: 1 }] : p.plan.framing
    const people = p.faces.length
    // (the first section starts the clip, a gap under 50 ms joins the section after it)
    const start = out.length ? a : 0
    out.push({
      start_ms: start, end_ms: p.b, layout: p.plan.layout, people, why: p.plan.why,
      slotKfs: views.map((v, slot) => opts.motion === false || p.plan.layout === 'frame_blur_fill'
        ? [{ t_ms: start, ...v }]
        : follow(frames, who, p.faces, v, views.filter((_, k) => k !== slot), start, p.b)),
    })
    at = p.b
  }
  if (durationMs - at >= 50) out.push(cutaway(at, durationMs))
  else if (out.length) out[out.length - 1].end_ms = durationMs
  if (!out.length) out.push(cutaway(0, durationMs))
  return out
}

/**
 * A view's keyframes over [a, b): on one person it keeps them in its middle as they move (across only, as
 * the editor's Follow; height and size as laid), never past the edge of a neighbour's view (`others`); on
 * two or more people, or nobody, it holds. The person is found by who they are (`who`), else — too small
 * to tell — as the face nearest where they were, of their size; alone in their scene, as its only face.
 */
function follow(frames: FrameDetection[], who: Map<CropBox, number>, faces: CropBox[], view: Omit<Kf, 't_ms'>, others: Omit<Kf, 't_ms'>[], a: number, b: number): Kf[] {
  const held = [{ t_ms: a, ...view }]
  const mine = faces.filter(f => within(f, view))
  if (mine.length !== 1) return held
  // Its strip: between the neighbours' views on either side (the views of a Split or Trio sit side by side in the picture)
  const mid = view.x + view.w / 2
  const lo = Math.max(0, ...others.filter(o => o.x + o.w / 2 < mid).map(o => o.x + o.w))
  const hi = Math.min(1, ...others.filter(o => o.x + o.w / 2 >= mid).map(o => o.x))
  const minX = Math.min(view.x, lo), maxX = Math.max(view.x, hi - view.w)
  const fr = [...frames].sort((p, q) => p.frame_index - q.frame_index)
    .filter(f => { const t = f.frame_index * FRAME_INTERVAL_MS + FRAME_SHOWS_MS; return t >= a && t < b })
  if (fr.length < 2) return held
  // Their face frame by frame: the face of the same track, else theirs by who it is, else (alone in the
  // scene) the only face, else the face nearest where they were, of their size
  const ids = trackFaces(fr, [])
  const person = (mine[0] as CropBox & { person?: number }).person
  const alone = faces.length === 1
  let cx = mine[0].x + mine[0].w / 2, w = mine[0].w, id: number | undefined
  const seen: Array<CropBox | null> = []
  const raw = fr.map((f, fi) => {
    const real = f.faces.map((face, k) => ({ face, id: ids[fi][k] })).filter(x => (x.face.score ?? 0) > 0)
    const same = (id !== undefined ? real.find(x => x.id === id) : undefined)
      ?? (person !== undefined ? real.find(x => who.get(x.face) === person) : undefined)
      ?? (alone && real.length === 1 ? real[0] : undefined)
    const near = same ?? real
      .filter(x => Math.abs(x.face.x + x.face.w / 2 - cx) <= FOLLOW_REACH && x.face.w >= w * FOLLOW_SIZE[0] && x.face.w <= w * FOLLOW_SIZE[1])
      .sort((p, q) => Math.abs(p.face.x + p.face.w / 2 - cx) - Math.abs(q.face.x + q.face.w / 2 - cx))[0]
    if (near) { cx = near.face.x + near.face.w / 2; w = near.face.w; id = near.id }
    seen.push(near?.face ?? null)
    return cx
  })
  // Smoothed (it lags someone moving fast) — but never so far behind that their face leaves the view:
  // then it moves just enough to keep it whole (never lose a subject), inside its strip
  const xs = stabilise(raw).map((c, i) => {
    let x = c - view.w / 2
    const f = seen[i]
    if (f && f.w <= view.w) x = clamp(x, f.x + f.w - view.w, f.x)
    return clamp(x, minX, maxX)
  })
  // A keyframe where the view moves, and the last still one before it (the renderer glides between keyframes)
  const kfs: Kf[] = [{ t_ms: a, ...view, x: xs[0] }]
  for (let i = 1; i < xs.length; i++) {
    const moved = Math.abs(xs[i] - xs[i - 1]) > 1e-4
    const moves = i + 1 < xs.length && Math.abs(xs[i + 1] - xs[i]) > 1e-4
    if (moved || moves) kfs.push({ t_ms: Math.max(a, fr[i].frame_index * FRAME_INTERVAL_MS + FRAME_SHOWS_MS), ...view, x: xs[i] })
  }
  return kfs
}
