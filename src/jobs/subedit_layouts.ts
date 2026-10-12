import {
  subjectBoxAt, cutsOnTimeline, followChanges, SUBJECT_STEP_MS,
  type SubjectPoint, type Cuts, type ViewChange, type BoxPosition,
} from './subject_framing.js'

// ── Sub-edit layouts ──────────────────────────────────────────────────────────
// Sub-edit picks the layout moment by moment from how many of the people are in the picture:
//   • one   → Vertical, locked on them (following them as they move);
//   • two   → Split: one in each half (left person on top);
//   • three → Trio: the one talking in the middle, the other two above and below;
//   • four or more → Split by halves: the top shows the left half of the picture, the bottom the
//     right half, so everyone stays in view.
// A layout holds at least MIN_PART_MS (a face missed for a moment never flips it); changes land on
// camera cuts and, in a trio, on the first word of whoever starts talking.

export type Person = { id: string; path: readonly SubjectPoint[]; speaking?: ReadonlyArray<readonly [number, number]> | null }
export type PartLayout = 'vertical' | 'split' | 'trio'
export interface LayoutPart {
  from: number
  to: number
  layout: PartLayout
  /** Top to bottom: who each slot follows, or a fixed view (the halves) */
  slots: Array<{ person: string } | { fixed: BoxPosition }>
}

/** A layout shorter than this joins the one before it */
const MIN_PART_MS = 1200
/** A turn shorter than this (a "yeah") doesn't move the middle of a trio */
const MIN_TURN_MS = 500
/** How much of the picture's height a slot's view takes: tighter in a split or a trio, so each shows one person */
const VIEW_H: Record<PartLayout, number> = { vertical: 1, split: 0.75, trio: 0.55 }
/** Each layout's slot shape in the reel (width / height, px) */
const SLOT_ASPECT: Record<PartLayout, number> = { vertical: 1080 / 1920, split: 1080 / 960, trio: 1080 / 640 }

/** A slot's view size on the source (fractions of its width and height), `h` of its height when it fits */
export function slotView(layout: PartLayout, videoAR: number, h = VIEW_H[layout]): { w: number; h: number } {
  const a = SLOT_ASPECT[layout] / videoAR   // the slot's shape as a share of the source's
  let w = a * h
  if (w > 1) { w = 1; h = 1 / a }
  return { w, h: Math.min(1, h) }
}

/**
 * The layouts over [from, to) of the timeline (`toSource` / `toTimeline`: timeline ↔ source ms;
 * `cuts`: the camera cuts, source ms).
 */
export function planLayouts(
  people: readonly Person[], from: number, to: number, toSource: (t: number) => number,
  toTimeline: (s: number) => number, cuts: Cuts,
): Array<{ from: number; to: number; key: string; members: string[] }> {
  const turns = people
    .flatMap(p => (p.speaking ?? []).map(([a, b]) => ({ id: p.id, a: Math.round(toTimeline(a)), b: Math.round(toTimeline(b)) })))
    .filter(t => t.b > from && t.a < to && t.b - t.a >= MIN_TURN_MS)
    .sort((x, y) => x.a - y.a)
  /** Who has been talking last at t (held through the silence after) */
  const speakerAt = (t: number) => {
    let who: string | null = turns[0]?.id ?? null
    for (const tr of turns) { if (tr.a <= t) who = tr.id; else break }
    return who
  }
  const points = new Set<number>([from])
  for (let t = from; t < to; t += SUBJECT_STEP_MS) points.add(Math.round(t))
  for (const c of cutsOnTimeline(cuts, from, to, toSource)) points.add(c)
  for (const tr of turns) if (tr.a > from && tr.a < to) points.add(tr.a)
  const at = [...points].filter(p => p >= from && p < to).sort((a, b) => a - b)

  /** The layout at a moment: its key (layout + who where) and its people, top to bottom */
  const stateAt = (p: number): { key: string; members: string[] } | null => {
    const src = toSource(p + 1)
    const seen = people
      .map(x => ({ x, box: subjectBoxAt(x.path, src, cuts) }))
      .filter((v): v is { x: Person; box: BoxPosition } => !!v.box)
      .sort((a, b) => (a.box.x + a.box.w / 2) - (b.box.x + b.box.w / 2))
      .map(v => v.x.id)
    if (!seen.length) return null
    if (seen.length === 1) return { key: `vertical:${seen[0]}`, members: seen }
    if (seen.length === 2) return { key: `split:${seen.join(',')}`, members: seen }
    if (seen.length === 3) {
      const talking = speakerAt(p)
      const middle = talking && seen.includes(talking) ? talking : seen[1]
      const others = seen.filter(id => id !== middle)
      return { key: `trio:${others[0]},${middle},${others[1]}`, members: [others[0], middle, others[1]] }
    }
    return { key: 'halves', members: seen }
  }

  const runs: Array<{ from: number; to: number; key: string; members: string[] }> = []
  for (const p of at) {
    const s = stateAt(p)
    const last = runs[runs.length - 1]
    if (!s) continue                       // nobody seen: what was showing holds
    if (last && last.key === s.key) continue
    runs.push({ from: p, to, ...s })
  }
  if (!runs.length) return []
  runs[0].from = from
  for (let i = 0; i < runs.length - 1; i++) runs[i].to = runs[i + 1].from
  // Short ones join the one before — except one starting on a camera cut, which joins the one after
  // (it's the new shot: the layout must change on the cut, not a moment later); the first joins the
  // one after. Then equal neighbours join
  const cutAt = new Set(cutsOnTimeline(cuts, from, to, toSource))
  for (let i = runs.length - 1; i >= 0 && runs.length > 1; i--) {
    if (runs[i].to - runs[i].from >= MIN_PART_MS) continue
    const intoNext = i < runs.length - 1 && (i === 0 || (cutAt.has(runs[i].from) && !cutAt.has(runs[i + 1].from)))
    if (intoNext) { runs[i + 1].from = runs[i].from; runs.splice(i, 1) }
    else { runs[i - 1].to = runs[i].to; runs.splice(i, 1) }
  }
  for (let i = runs.length - 1; i > 0; i--) if (runs[i].key === runs[i - 1].key) { runs[i - 1].to = runs[i].to; runs.splice(i, 1) }
  return runs
}

/**
 * The layouts as sections, each slot's view changes included: a person's slot follows them
 * (smoothly, keeping the view's size); the halves are still views of the left and right half.
 */
export function layoutSections(
  people: readonly Person[], from: number, to: number, toSource: (t: number) => number,
  toTimeline: (s: number) => number, cuts: Cuts, videoAR: number,
): Array<{ from: number; to: number; layout: PartLayout; slots: ViewChange[][] }> {
  const byId = new Map(people.map(p => [p.id, p]))
  return planLayouts(people, from, to, toSource, toTimeline, cuts).map(run => {
    if (run.key === 'halves') {
      // Everyone's faces, their middle height: both halves look there
      const mid = toSource((run.from + run.to) / 2)
      const ys = run.members.flatMap(id => { const b = subjectBoxAt(byId.get(id)!.path, mid, cuts); return b ? [b.y + b.h / 2] : [] })
      const cy = ys.length ? ys.reduce((a, b) => a + b, 0) / ys.length : 0.4
      const h = Math.min(1, (0.5 * videoAR) / SLOT_ASPECT.split)
      const y = Math.max(0, Math.min(1 - h, cy - h * 0.4))
      const half = (x: number): ViewChange[] => [{ t_ms: run.from, x, y, w: 0.5, h, cut: false }]
      return { from: run.from, to: run.to, layout: 'split' as const, slots: [half(0), half(0.5)] }
    }
    const layout = run.key.split(':')[0] as PartLayout
    const view = slotView(layout, videoAR)
    const slots = run.members.map(id => {
      const follow = followChanges(byId.get(id)!.path, run.from, run.to, toSource, view, 'smooth', cuts)
      // Not seen at all in this stretch (shouldn't happen: they were seen to be here): the middle
      return follow?.length ? follow : [{ t_ms: run.from, x: (1 - view.w) / 2, y: 0, w: view.w, h: view.h, cut: false }]
    })
    return { from: run.from, to: run.to, layout, slots }
  })
}
