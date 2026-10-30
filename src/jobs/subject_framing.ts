// ── Subject framing (the worker's copy) ───────────────────────────────────────
// A copy of chai-cut-frontend/src/modules/editor/subjects.ts (without spliceFollow) and the bits of
// views.ts it needs, so Sub-edit on the clip board frames a clip exactly as the editor's Subjects
// tool does. KEEP THE TWO IN STEP.

export type BoxPosition = { x: number; y: number; w: number; h: number }
export type ViewKeyframe = { t_ms: number; x: number; y: number; w: number; h: number }
export type ViewChange = ViewKeyframe & { cut: boolean }
/** A keyframe this close before the next one only holds the old view up to a cut (views.ts) */
export const HOLD_GAP_MS = 1

/** Keyframes for a list of view changes: each cut gets a hold keyframe just before it (views.ts) */
export function buildKeyframes(changes: readonly ViewChange[]): ViewKeyframe[] {
  const pos = (k: ViewKeyframe): BoxPosition => ({ x: k.x, y: k.y, w: k.w, h: k.h })
  const sorted = [...changes].sort((a, b) => a.t_ms - b.t_ms)
  const out: ViewKeyframe[] = []
  sorted.forEach((c, i) => {
    const prev = sorted[i - 1]
    if (c.cut && prev && c.t_ms - prev.t_ms > HOLD_GAP_MS + 1) out.push({ t_ms: c.t_ms - HOLD_GAP_MS, ...pos(prev) })
    out.push({ t_ms: c.t_ms, ...pos(c) })
  })
  return out
}

// ── Subjects: following a marked person with a view ───────────────────────────
// The worker saves where each subject's face is, 4× a second, in source-video time (only where
// they were seen: server/services/subjects.ts). Here that becomes a view's keyframes over a
// stretch of the timeline the user picks — the view keeps its size and only moves to frame them:
//   • smooth: it follows them gently (a small wobble of the face doesn't move it);
//   • still: it holds one place while they stay there and cuts when they go somewhere else.
// Either way a camera cut (they jump across the picture) is a cut, never a slide. Everything
// stays plain view keyframes (views.ts), so the preview and the export play it exactly alike and
// the user can still change any of it by hand.

/**
 * [source ms, x, y, w, h, talking] of the subject's face (fractions of the frame). talking (tracked
 * since 2026-10-12): 1 while they are the one talking (their mouth moves with the speech), else 0
 */
export type SubjectPoint = [number, number, number, number, number, number?]
export type FollowStyle = 'smooth' | 'still'

/** The path is sampled this often (the worker looks 4× a second) */
export const SUBJECT_STEP_MS = 250
/** A face this long from the nearest sighting counts as not seen there */
const SEEN_WITHIN_MS = 375
/** The face jumping this far between two samples (a quarter second): a camera cut, or found again elsewhere */
const JUMP = 0.12
/** Still: a place held for less than this (from a jump) joins the stretch next to it, so a blip never flicks the view */
const MIN_HOLD_MS = 2000
/** Smooth: how quickly the view catches up (per sample; about 30% a second), and moves smaller than this are ignored */
const EASE = 1 - Math.pow(1 - 0.3, SUBJECT_STEP_MS / 1000)
const DEAD_ZONE = 0.03

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))
/** Eased toward each value, then held until it has moved more than the dead zone (backend ai_edit's _ema + _deadZone) */
function steady(values: number[]): number[] {
  const out: number[] = []
  let ema = values[0], held = values[0]
  values.forEach((v, i) => {
    if (i > 0) ema = EASE * v + (1 - EASE) * ema
    if (i > 0 && Math.abs(ema - held) > DEAD_ZONE) held = ema
    out.push(held)
  })
  return out
}
const median = (v: number[]) => { const s = [...v].sort((a, b) => a - b); return s.length ? s[Math.floor(s.length / 2)] : 0 }

/** Camera cuts (source ms, in order), from the tracker */
export type Cuts = readonly number[]

/** Whether a camera cut lies between two moments of the source (a < cut ≤ b, either order) */
export function cutBetween(cuts: Cuts | undefined, a: number, b: number): boolean {
  if (!cuts?.length || a === b) return false
  const lo = Math.min(a, b), hi = Math.max(a, b)
  // The first cut after lo
  let i = 0, j = cuts.length
  while (i < j) { const m = (i + j) >> 1; if (cuts[m] <= lo) i = m + 1; else j = m }
  return i < cuts.length && cuts[i] <= hi
}

/**
 * The camera cuts inside [from, to) of the timeline, in timeline ms (`toSource`: timeline ms →
 * source ms; a cut is placed between the two quarter-second steps it falls between)
 */
export function cutsOnTimeline(cuts: Cuts | undefined, from: number, to: number, toSource: (t: number) => number): number[] {
  if (!cuts?.length) return []
  const out: number[] = []
  for (let t = from; t < to; t += SUBJECT_STEP_MS) {
    const s0 = toSource(t), s1 = toSource(Math.min(to, t + SUBJECT_STEP_MS))
    for (const c of cuts) {
      if (c <= s0 || c > s1) continue
      // Within a stretch that plays straight through: where it falls; across a removed part: its end
      const k = s1 - s0 > 0 && s1 - s0 < 2 * SUBJECT_STEP_MS ? (c - s0) / (s1 - s0) : 1
      out.push(Math.round(t + k * (Math.min(to, t + SUBJECT_STEP_MS) - t)))
    }
  }
  return out
}

/** The subject's face at a moment of the source video, or null when they weren't seen around then (never across a camera cut) */
export function subjectBoxAt(path: readonly SubjectPoint[], sourceMs: number, cuts?: Cuts): BoxPosition | null {
  if (!path.length) return null
  // The first point at or after sourceMs
  let lo = 0, hi = path.length
  while (lo < hi) { const mid = (lo + hi) >> 1; if (path[mid][0] < sourceMs) lo = mid + 1; else hi = mid }
  // Only sightings in the same shot as this moment
  const sameShot = (p: SubjectPoint | undefined) => (p && !cutBetween(cuts, p[0], sourceMs) ? p : undefined)
  const a = sameShot(path[lo - 1]), b = sameShot(path[lo])
  const box = (p: SubjectPoint): BoxPosition => ({ x: p[1], y: p[2], w: p[3], h: p[4] })
  // Between two sightings a frame or so apart: in between them
  if (a && b && b[0] - a[0] <= 2 * SEEN_WITHIN_MS) {
    const f = (sourceMs - a[0]) / Math.max(1, b[0] - a[0])
    return { x: a[1] + (b[1] - a[1]) * f, y: a[2] + (b[2] - a[2]) * f, w: a[3] + (b[3] - a[3]) * f, h: a[4] + (b[4] - a[4]) * f }
  }
  const near = [a, b].filter((p): p is SubjectPoint => !!p && Math.abs(p[0] - sourceMs) <= SEEN_WITHIN_MS)
    .sort((p, q) => Math.abs(p[0] - sourceMs) - Math.abs(q[0] - sourceMs))[0]
  return near ? box(near) : null
}

/** Whether the subject is the one talking at a moment of the source (the tracker's reading nearest it, in the same shot) */
export function subjectTalksAt(path: readonly SubjectPoint[], sourceMs: number, cuts?: Cuts): boolean {
  if (!path.length) return false
  let lo = 0, hi = path.length
  while (lo < hi) { const mid = (lo + hi) >> 1; if (path[mid][0] < sourceMs) lo = mid + 1; else hi = mid }
  const near = [path[lo - 1], path[lo]].filter((p): p is SubjectPoint => !!p && Math.abs(p[0] - sourceMs) <= SEEN_WITHIN_MS && !cutBetween(cuts, p[0], sourceMs))
    .sort((p, q) => Math.abs(p[0] - sourceMs) - Math.abs(q[0] - sourceMs))[0]
  return !!near && near[5] === 1
}

/** A switch to someone else holds at least this long (unless they leave the picture) */
const SWITCH_HOLD_MS = 1500
/** A stretch shorter than this joins the one before it (no flicker) */
const MIN_STRETCH_MS = 1000
/** Someone else must be talking this long before the view goes to them (the switch is placed where they started) */
const SWITCH_CONFIRM_MS = 750

/**
 * Who the view shows through [from, to) of the timeline, as stretches of one subject each: whoever
 * is talking; while nobody is (or it's unclear) whoever it shows now, as long as they are in the
 * picture; when they leave it, someone who is (the one talking, else the first marked). A switch
 * holds at least SWITCH_HOLD_MS. Empty when none of them is seen there.
 */
export function directStretches(
  subjects: ReadonlyArray<{ id: string; path: readonly SubjectPoint[] }>, from: number, to: number, toSource: (t: number) => number,
  cuts?: Cuts,
): Array<{ from: number; to: number; id: string }> {
  const picks: Array<{ t: number; id: string | null }> = []
  let current: string | null = null, since = -Infinity
  // Someone else talking, and since when (picks index)
  let pending: string | null = null, pendingFrom = 0
  for (let t = from; t < to; t += SUBJECT_STEP_MS) {
    const src = toSource(t)
    const seen = subjects.filter(s => subjectBoxAt(s.path, src, cuts))
    const talking = seen.filter(s => subjectTalksAt(s.path, src, cuts))
    const here = current !== null && seen.some(s => s.id === current)
    let want: string | null = current
    if (talking.length) want = talking.some(s => s.id === current) ? current : talking[0].id
    else if (!here && seen.length) want = seen[0].id
    if (want === current) { pending = null; picks.push({ t, id: current }); continue }
    // The one shown has left the picture (or nobody is shown yet): at once
    if (!here || current === null) { current = want; since = t; pending = null; picks.push({ t, id: current }); continue }
    // Someone else is talking: once they have kept on for a moment (and the hold is over), the view
    // goes to them from where they started
    if (pending !== want) { pending = want; pendingFrom = picks.length }
    picks.push({ t, id: current })
    const startedAt = picks[pendingFrom].t
    if (t + SUBJECT_STEP_MS - startedAt >= SWITCH_CONFIRM_MS && startedAt - since >= SWITCH_HOLD_MS) {
      current = want; since = startedAt
      for (let k = pendingFrom; k < picks.length; k++) picks[k].id = current
      pending = null
    }
  }
  // Before anyone is chosen: the first one chosen
  const first = picks.find(p => p.id)?.id
  if (!first) return []
  const out: Array<{ from: number; to: number; id: string }> = []
  for (const p of picks) {
    const id = p.id ?? first
    const last = out[out.length - 1]
    if (last && last.id === id) last.to = Math.min(to, p.t + SUBJECT_STEP_MS)
    else out.push({ from: p.t, to: Math.min(to, p.t + SUBJECT_STEP_MS), id })
  }
  out[0].from = from
  out[out.length - 1].to = to
  // Blips join the stretch before (or after, at the start)
  for (let i = out.length - 1; i >= 0 && out.length > 1; i--) {
    if (out[i].to - out[i].from >= MIN_STRETCH_MS) continue
    if (i > 0) { out[i - 1].to = out[i].to; out.splice(i, 1) }
    else { out[1].from = out[0].from; out.splice(0, 1) }
  }
  for (let i = out.length - 1; i > 0; i--) if (out[i].id === out[i - 1].id) { out[i - 1].to = out[i].to; out.splice(i, 1) }
  // A switch next to a camera cut happens exactly on the cut (the steps are a quarter second apart)
  const onTimeline = cutsOnTimeline(cuts, from, to, toSource)
  for (let i = 1; i < out.length; i++) {
    const b = out[i].from
    const c = onTimeline.filter(x => x > b - SUBJECT_STEP_MS - 1 && x < b + SUBJECT_STEP_MS && x > out[i - 1].from && x < out[i].to)
      .sort((x, y) => Math.abs(x - b) - Math.abs(y - b))[0]
    if (c !== undefined) { out[i - 1].to = c; out[i].from = c }
  }
  return out
}

/** [start, end] source ms while a subject is talking — the tracker's turns, to the word */
export type Turn = readonly [number, number]
type Speaker = { id: string; path: readonly SubjectPoint[]; speaking?: readonly Turn[] | null }
/** A turn shorter than this (a "yeah", a laugh) doesn't take the view */
const MIN_TURN_MS = 500
/** A stretch shorter than this joins the one before it (no flicker) */
const MIN_TURN_STRETCH_MS = 300

/**
 * Who the view shows through [from, to), from the speakers' turns: from the first word of a turn
 * the view is on whoever says it — cut exactly there, no delay — and stays on them through the
 * silence after, until the next person starts talking. Someone who isn't in the picture (another
 * camera's shot) can't be shown: then whoever is in it, back to the speaker as soon as the picture
 * cuts back to them. Before the first turn: the first speaker. `toTimeline`: source ms → timeline
 * ms. Empty when nobody has turns or nobody is seen.
 */
export function turnStretches(
  subjects: readonly Speaker[], from: number, to: number, toSource: (t: number) => number,
  toTimeline: (s: number) => number, cuts?: Cuts,
): Array<{ from: number; to: number; id: string }> {
  const turns = subjects
    .flatMap(s => (s.speaking ?? []).map(([a, b]) => ({ id: s.id, a: Math.round(toTimeline(a)), b: Math.round(toTimeline(b)) })))
    .filter(t => t.b > from && t.a < to && t.b - t.a >= MIN_TURN_MS)
    .sort((x, y) => x.a - y.a)
  if (!turns.length) return []
  const speakerAt = (t: number) => {
    let who = turns[0].id
    for (const tr of turns) { if (tr.a <= t) who = tr.id; else break }
    return who
  }
  // Moments the view may change: each turn's first word, each camera cut, and a quarter-second
  // step (someone leaving the picture without a cut)
  const points = new Set<number>([from])
  for (const tr of turns) if (tr.a > from) points.add(tr.a)
  for (const c of cutsOnTimeline(cuts, from, to, toSource)) points.add(c)
  for (let t = from; t < to; t += SUBJECT_STEP_MS) points.add(Math.round(t))
  const at = [...points].filter(p => p >= from && p < to).sort((a, b) => a - b)
  // Seen just after the moment (on a cut: in the new shot)
  const seenAt = (s: Speaker, t: number) => !!subjectBoxAt(s.path, toSource(t + 1), cuts)
  const out: Array<{ from: number; to: number; id: string }> = []
  let current: string | null = null
  for (const p of at) {
    const want = speakerAt(p)
    const seen = subjects.filter(s => seenAt(s, p))
    let next: string | null = current
    if (seen.some(s => s.id === want)) next = want
    else if (current === null || !seen.some(s => s.id === current)) next = seen[0]?.id ?? current
    if (next === null) continue
    if (next !== current) { out.push({ from: p, to, id: next }); current = next }
  }
  if (!out.length) return []
  out[0].from = from
  for (let i = 0; i < out.length - 1; i++) out[i].to = out[i + 1].from
  // Blips (a person seen for a moment between two others) join the stretch before
  for (let i = out.length - 1; i > 0; i--) {
    if (out[i].to - out[i].from >= MIN_TURN_STRETCH_MS) continue
    out[i - 1].to = out[i].to
    out.splice(i, 1)
  }
  for (let i = out.length - 1; i > 0; i--) if (out[i].id === out[i - 1].id) { out[i - 1].to = out[i].to; out.splice(i, 1) }
  return out
}

/**
 * The view changes that switch between the subjects over [from, to): each stretch follows its
 * subject (smoothly, the view keeping its size), with a cut where the subject changes. With the
 * speakers' turns (and `toTimeline`), on the turns — to the word; else on the talking readings.
 */
export function autoDirectChanges(
  subjects: readonly Speaker[], from: number, to: number, toSource: (t: number) => number,
  view: { w: number; h: number }, cuts?: Cuts, toTimeline?: (s: number) => number,
): ViewChange[] | null {
  const byTurns = toTimeline && subjects.some(s => s.speaking?.length) ? turnStretches(subjects, from, to, toSource, toTimeline, cuts) : []
  const stretches = byTurns.length ? byTurns : directStretches(subjects, from, to, toSource, cuts)
  if (!stretches.length) return null
  const out: ViewChange[] = []
  for (const st of stretches) {
    const s = subjects.find(x => x.id === st.id)!
    const part = followChanges(s.path, st.from, st.to, toSource, view, 'smooth', cuts)
    if (!part?.length) continue
    part.forEach((c, i) => out.push({ ...c, t_ms: i === 0 ? st.from : c.t_ms, cut: i === 0 ? out.length > 0 : c.cut }))
  }
  return out.length ? out : null
}

/** Where a view of this size looks to frame a face: centred on it; zoomed in, the face a little above the middle */
export function viewOnFace(face: BoxPosition, view: { w: number; h: number }): BoxPosition {
  const cx = face.x + face.w / 2, cy = face.y + face.h / 2
  return {
    x: clamp(cx - view.w / 2, 0, Math.max(0, 1 - view.w)),
    // (and its top at least half a face above the face's top: the face box ends at the forehead and
    // a tight view put the hair outside it — user, 2026-10-24; as autoFormat.ts HEAD_ROOM)
    y: view.h >= 0.999 ? 0 : clamp(Math.min(cy - view.h * 0.4, face.y - 0.5 * face.h), 0, Math.max(0, 1 - view.h)),
    w: view.w, h: view.h,
  }
}

/**
 * The view changes that follow a subject over [from, to) of the timeline (`toSource`: timeline ms
 * → source ms), keeping the view's size. Before they are first seen it waits where they appear;
 * while they are out of the picture it stays where they were last seen. Null when they aren't seen
 * in that stretch at all.
 */
export function followChanges(
  path: readonly SubjectPoint[], from: number, to: number, toSource: (t: number) => number,
  view: { w: number; h: number }, style: FollowStyle, cuts?: Cuts,
): ViewChange[] | null {
  const ts: number[] = []
  for (let t = from; t < to; t += SUBJECT_STEP_MS) ts.push(Math.round(t))
  if (!ts.length) return null
  const raw = ts.map(t => subjectBoxAt(path, toSource(t), cuts))
  if (!raw.some(Boolean)) return null
  // Not seen: the nearest sighting before (else after) holds
  const filled: BoxPosition[] = []
  let last: BoxPosition | null = null
  for (const b of raw) { if (b) last = b; filled.push((b ?? last)!) }
  const firstSeen = raw.find(Boolean)!
  for (let i = 0; i < filled.length && !filled[i]; i++) filled[i] = firstSeen
  const target = filled.map(b => viewOnFace(b, view))
  const centre = filled.map(b => b.x + b.w / 2)

  // Stretches between jumps (a camera cut, or the person found again somewhere else)
  const starts = [0]
  for (let i = 1; i < ts.length; i++) if (Math.abs(centre[i] - centre[i - 1]) > JUMP) starts.push(i)
  const parts = starts.map((s, k) => ({ from: s, to: starts[k + 1] ?? ts.length }))
  // Where each stretch starts: a jump right after a camera cut starts exactly on the cut
  const onTimeline = cutsOnTimeline(cuts, from, to, toSource)
  const startOf = (i: number) => {
    if (i === 0) return ts[0]
    const c = onTimeline.find(x => x > ts[i - 1] && x <= ts[i])
    return c ?? ts[i]
  }

  const changes: ViewChange[] = []
  if (style === 'still') {
    const minLen = Math.ceil(MIN_HOLD_MS / SUBJECT_STEP_MS)
    const placeOf = (p: { from: number; to: number }) => {
      const xs = target.slice(p.from, p.to)
      return { x: median(xs.map(v => v.x)), y: median(xs.map(v => v.y)) }
    }
    const apart = (a: { x: number; y: number }, b: { x: number; y: number }) => Math.abs(a.x - b.x) + Math.abs(a.y - b.y)
    const join = (i: number) => { parts[i - 1].to = parts[i].to; parts.splice(i, 1) }
    for (;;) {
      // Neighbours that frame the same place are one hold
      for (let i = parts.length - 1; i > 0; i--) if (apart(placeOf(parts[i - 1]), placeOf(parts[i])) < DEAD_ZONE) join(i)
      // The shortest stretch under MIN_HOLD_MS joins the neighbour framing the nearest place
      // (equally near: the longer one), so a blip never flicks the view away and back
      const i = parts.length > 1
        ? parts.map((p, k) => ({ k, len: p.to - p.from })).filter(x => x.len < minLen).sort((a, b) => a.len - b.len)[0]?.k
        : undefined
      if (i === undefined) break
      const here = placeOf(parts[i])
      const score = (k: number) => (k < 0 || k >= parts.length ? Infinity : apart(here, placeOf(parts[k])) - (parts[k].to - parts[k].from) * 1e-6)
      if (score(i - 1) <= score(i + 1)) join(i)
      else join(i + 1)
    }
    for (const p of parts) changes.push({ t_ms: startOf(p.from), ...placeOf(p), w: view.w, h: view.h, cut: changes.length > 0 })
    return changes
  }
  for (const p of parts) {
    // Eased, then held while they barely move (as Make my clips frames people); each stretch
    // starts where they are, with a cut
    const xs = steady(target.slice(p.from, p.to).map(v => v.x))
    const ys = steady(target.slice(p.from, p.to).map(v => v.y))
    for (let k = 0; k < xs.length; k++) {
      const i = p.from + k
      const prev = changes[changes.length - 1]
      // Only where the view moves: the renderer glides in a straight line between points anyway
      const moved = k === 0 || !prev || xs[k] !== prev.x || ys[k] !== prev.y
      const holdsTill = k > 0 && moved && prev && ts[i - 1] !== prev.t_ms
      // A point where it set off, so it doesn't glide all the way from the last change
      if (holdsTill) changes.push({ t_ms: ts[i - 1], x: prev.x, y: prev.y, w: view.w, h: view.h, cut: false })
      if (moved) changes.push({ t_ms: k === 0 ? startOf(i) : ts[i], x: xs[k], y: ys[k], w: view.w, h: view.h, cut: k === 0 && changes.length > 0 })
    }
  }
  return changes
}
