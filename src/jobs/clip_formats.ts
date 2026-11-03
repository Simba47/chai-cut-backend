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
//      a still centred Vertical, as the old framing's cutaways — one under SHORT_PLACE goes on in the
//      section before it (a centred view there showed the wall between two people: 2026-10-31).
//   4. Follow (the editor's Cuts mode): a view on one person keeps on them as they move, inside its strip
//      of the picture (never over a neighbour's view); a view on two or more holds. Motion off: all hold.
//   5. Brief appearances (findAngles from the close looks: someone on screen ~0.5 s): the format stays and
//      the view nearest them widens to take them in for that moment, cutting out and back (user, 2026-11-01).

import { trackFaces, stabilise, FRAME_INTERVAL_MS, type FrameDetection, type CropBox, type SpokenWord } from './ai_edit.js'
import { findAngles, whoIs, withoutPictures, faceCuts, FRAME_SHOWS_MS, type Brief } from './track_subjects.js'
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
  // (the close looks only find brief appearances: the rest — pictures, who is who, Follow — reads the 4-a-second frames)
  const close = frames.filter(f => f.close)
  frames = withoutPictures(frames.filter(f => !f.close), cuts)
  // (with the camera cuts FFmpeg missed: wipes, soft cuts — none when the caller added them already)
  cuts = [...cuts, ...faceCuts(frames, cuts)].sort((x, y) => x - y)
  const angles = findAngles([...frames, ...close], cuts, durationMs, speech)
  // Who each face is: a scene's people are where it was first seen, and someone can sit elsewhere in a
  // place it comes back (another take of the same angle): Follow finds them by who they are
  const who = opts.motion === false ? new Map<CropBox, number>() : whoIs(frames, cuts)
  type Decided = { a: number; b: number; faces: CropBox[]; plan: AutoLayout }
  const decided: Decided[] = angles.filter(g => g.people > 0).flatMap(g => g.spans.flatMap(([s, e], i): Decided[] => {
    const a = Math.max(0, Math.round(s)), b = Math.min(durationMs, Math.round(e))
    if (b - a < MIN_PLACE_MS) return []
    // (the people in this place: a scene's places can show fewer of them — 2026-11-03)
    const here = g.present?.[i] ?? g.faces.map((_, k) => k)
    const faces = here.map(k => g.faces[k]) as CropBox[]
    const talk = g.talk?.[i] ? here.map(k => g.talk![i][k] ?? 0) : null
    const plan = decideFormat(faces, talk, ar, size.height)
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
  /** No faces through [a, b): a cutaway — or, under SHORT_PLACE, the section before it goes on (its views hold) */
  const between = (a: number, b: number) => {
    if (b - a < AUTO_FORMAT.SHORT_PLACE * 1000 && out.length) out[out.length - 1].end_ms = b
    else out.push(cutaway(a, b))
  }
  let at = 0
  for (const p of places) {
    const a = Math.max(p.a, at)
    if (a - at >= 50) between(at, a)
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
  if (durationMs - at >= 50) between(at, durationMs)
  else if (out.length) out[out.length - 1].end_ms = durationMs
  if (!out.length) out.push(cutaway(0, durationMs))
  // Someone on screen briefly: the section keeps its format, the view nearest them takes them in
  for (const g of angles) for (const b of g.brief ?? []) {
    const part = out.find(p => p.why !== 'cutaway' && p.layout !== 'frame_blur_fill' && p.start_ms < b.to && p.end_ms > b.from)
    if (part) widenFor(part, b, g.faces as CropBox[])
  }
  return out
}

/** Room kept around a newcomer's face when a view widens to take them in (face heights above and below, face widths beside) */
const WIDEN_ROOM_TOP = 0.5
const WIDEN_ROOM_BOTTOM = 0.6
const WIDEN_ROOM_SIDE = 0.25
/** A brief moment shorter than this (once inside its section) isn't widened for */
const WIDEN_MIN_MS = 150

/** Where a view is at `t`: the renderer glides between keyframes (before the first: the first) */
export function viewAt(kfs: readonly Kf[], t: number): Omit<Kf, 't_ms'> {
  const pick = ({ x, y, w, h }: Kf) => ({ x, y, w, h })
  if (t <= kfs[0].t_ms) return pick(kfs[0])
  for (let i = 1; i < kfs.length; i++) {
    if (t > kfs[i].t_ms) continue
    const p = kfs[i - 1], q = kfs[i], f = q.t_ms === p.t_ms ? 1 : (t - p.t_ms) / (q.t_ms - p.t_ms)
    return { x: p.x + (q.x - p.x) * f, y: p.y + (q.y - p.y) * f, w: p.w + (q.w - p.w) * f, h: p.h + (q.h - p.h) * f }
  }
  return pick(kfs[kfs.length - 1])
}

/**
 * A view taking in `face` as well as the people it shows (`keep`): the same shape, never smaller, about
 * the middle of them all, inside the picture and its strip (lo..hi across — never over a neighbour's
 * view); everyone in it whole. With the room around the faces if that fits, else the faces alone; null
 * when even that can't be done (the view stays as it was).
 */
export function widened(view: Omit<Kf, 't_ms'>, keep: readonly CropBox[], face: CropBox, lo = 0, hi = 1): Omit<Kf, 't_ms'> | null {
  const shape = view.w / view.h
  const all = [...keep, face]
  for (const room of [true, false]) {
    const x0 = Math.min(...all.map(f => f.x - (room ? WIDEN_ROOM_SIDE * f.w : 0)))
    const x1 = Math.max(...all.map(f => f.x + f.w + (room ? WIDEN_ROOM_SIDE * f.w : 0)))
    const y0 = Math.max(0, Math.min(...all.map(f => f.y - (room ? WIDEN_ROOM_TOP * f.h : 0))))
    const y1 = Math.min(1, Math.max(...all.map(f => f.y + f.h + (room ? WIDEN_ROOM_BOTTOM * f.h : 0))))
    let h = Math.min(1, Math.max(view.h, y1 - y0, (x1 - x0) / shape))
    let w = shape * h
    if (w > hi - lo) { w = hi - lo; h = w / shape }
    if (w < view.w - 1e-6) return null
    const v = { x: clamp((x0 + x1) / 2 - w / 2, lo, hi - w), y: clamp((y0 + y1) / 2 - h / 2, 0, 1 - h), w, h }
    if (all.every(f => within(f, v))) return v
  }
  return null
}

/**
 * A brief appearance in a section: the view whose strip holds them (else the nearest) widens to take
 * them in with the scene's people it shows (`people`) over [from, to) — cut out and back, as the editor
 * cuts (a keyframe 1 ms before each change); its moves within that moment give way. The section's format
 * and other views stay as they are.
 */
function widenFor(part: FormattedPart, b: Brief, people: readonly CropBox[]) {
  const from = Math.max(part.start_ms, b.from), to = Math.min(part.end_ms, b.to)
  if (to - from < WIDEN_MIN_MS) return
  const face: CropBox = { x: b.x, y: b.y, w: b.w, h: b.h }
  const views = part.slotKfs.map(kfs => viewAt(kfs, from))
  const cx = b.x + b.w / 2
  const strip = (s: number) => {
    const mid = views[s].x + views[s].w / 2
    const others = views.filter((_, k) => k !== s)
    return [Math.max(0, ...others.filter(o => o.x + o.w / 2 < mid).map(o => o.x + o.w)), Math.min(1, ...others.filter(o => o.x + o.w / 2 >= mid).map(o => o.x))]
  }
  // (a view that already shows them whole needs nothing)
  if (views.some(v => within(face, v))) return
  const order = views.map((v, s) => ({ s, inStrip: cx >= strip(s)[0] && cx <= strip(s)[1], d: Math.abs(v.x + v.w / 2 - cx) }))
    .sort((p, q) => Number(q.inStrip) - Number(p.inStrip) || p.d - q.d)
  for (const { s } of order) {
    const [lo, hi] = strip(s)
    const v = widened(views[s], people.filter(f => within(f, views[s])), face, lo, hi)
    if (!v) continue
    const kfs = part.slotKfs[s]
    const back = viewAt(kfs, to)
    const held = from > part.start_ms ? [{ t_ms: from - 1, ...viewAt(kfs, from - 1) }] : []
    const next: Kf[] = [
      ...kfs.filter(k => k.t_ms < from - 1 || k.t_ms >= to + 1),
      ...held, { t_ms: from, ...v }, { t_ms: to - 1, ...v },
      ...(to < part.end_ms ? [{ t_ms: to, ...back }] : []),
    ].sort((p, q) => p.t_ms - q.t_ms)
    part.slotKfs[s] = next.filter((k, i) => i === next.length - 1 || next[i + 1].t_ms !== k.t_ms)
    return
  }
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
