import { test } from 'node:test'
import assert from 'node:assert/strict'
import type { FrameDetection, CropBox } from './ai_edit.js'

// Close looks and brief appearances (user, 2026-11-01: "frame even if 0.5 sec" — keep the format, widen the view)
// track_subjects and clip_formats read the database and storage settings when they load; the functions tested here use neither
process.env.DATABASE_URL ??= 'postgres://test:test@127.0.0.1:1/test'
process.env.R2_ENDPOINT ??= 'http://127.0.0.1:1'
process.env.R2_ACCESS_KEY_ID ??= 'test'
process.env.R2_SECRET_ACCESS_KEY ??= 'test'
process.env.R2_BUCKET ??= 'test'
const { findAngles, closeLookWindows } = await import('./track_subjects.js')
const { formatClip, widened, viewAt } = await import('./clip_formats.js')

const SIZE = { width: 1920, height: 1080 }
const sig = (d: number) => [0.5 - d / 2, 0.1 + d / 2, ...Array.from({ length: 22 }, () => 0.4 / 22)]
const face = (x: number, d: number, w = 0.08, extra: Partial<CropBox> = {}): CropBox => ({ x: x - w / 2, y: 0.3, w, h: w * 1.6, score: 0.95, sig: sig(d), ...extra })
const within = (f: CropBox, v: { x: number; y: number; w: number; h: number }) => f.x >= v.x - 1e-6 && f.x + f.w <= v.x + v.w + 1e-6 && f.y >= v.y - 1e-6 && f.y + f.h <= v.y + v.h + 1e-6

/** 4 frames a second over [0, dur) from `at(t)`; and the close looks, 10 a second over `windows` */
function framesOf(dur: number, at: (t: number) => CropBox[]): FrameDetection[] {
  const out: FrameDetection[] = []
  for (let n = 0; n * 250 + 100 < dur; n++) { const faces = at(n * 250 + 100); out.push({ frame_index: n, person_count: faces.length, faces }) }
  return out
}
function closeOf(windows: Array<[number, number]>, at: (t: number) => CropBox[]): FrameDetection[] {
  return windows.flatMap(([a, b]) => {
    const out: FrameDetection[] = []
    for (let t = a + 40; t < b; t += 100) { const faces = at(t); out.push({ frame_index: (t - 100) / 250, person_count: faces.length, faces, close: true }) }
    return out
  })
}

// ── A one-shot clip, 10 s: A talks throughout at 0.42; B (at 0.60) comes in for half a second, 4.0–4.5 s ──
const DUR = 10000
const A = (t: number) => face(0.42, 0.05)
const withB = (bx: number, extra: Partial<CropBox> = {}) => (t: number) => [A(t), ...(t >= 4000 && t < 4500 ? [face(bx, 0.9, 0.08, extra)] : [])]
const scene = withB(0.60)
const regular = framesOf(DUR, scene)

test('close looks: around a face seen in only two frames, not around someone seen throughout; a short shot whole', () => {
  const w = closeLookWindows(regular, [], DUR)
  assert.equal(w.length, 1, JSON.stringify(w))
  assert.ok(w[0][0] <= 4000 && w[0][1] >= 4500, JSON.stringify(w))
  assert.ok(w[0][1] - w[0][0] <= 1500, 'a short look, not the whole clip')
  // a 0.6 s shot between two cuts is read whole
  const shots = closeLookWindows(framesOf(DUR, t => [A(t)]), [6000, 6600], DUR)
  assert.deepEqual(shots, [[6000, 6600]])
})

test('close looks: at most a quarter of the clip', () => {
  // a face flickering in and out all clip long (seen once every second)
  const flicker = framesOf(DUR, t => [A(t), ...(Math.floor(t / 250) % 4 === 0 ? [face(0.8, 0.9)] : [])])
  const w = closeLookWindows(flicker, [], DUR)
  assert.ok(w.reduce((n, [a, b]) => n + b - a, 0) <= 0.25 * DUR + 1, JSON.stringify(w))
})

test('brief appearance: B (0.5 s) is one with the close looks; without them only A, as before', () => {
  const before = findAngles(regular, [], DUR)
  assert.equal(before[0].people, 1)
  assert.equal(before[0].brief, undefined, 'two 4-a-second frames: a misread to the old rule')
  const close = closeOf(closeLookWindows(regular, [], DUR), scene)
  const after = findAngles([...regular, ...close], [], DUR)
  // the scene is what it was: one person, the same face, the same spans
  assert.equal(after[0].people, 1)
  assert.deepEqual(after[0].faces, before[0].faces)
  assert.deepEqual(after[0].spans, before[0].spans)
  assert.equal(after[0].brief?.length, 1, JSON.stringify(after[0].brief))
  const b = after[0].brief![0]
  assert.ok(b.from <= 4050 && b.to >= 4450 && b.to - b.from <= 900, JSON.stringify(b))
  assert.ok(Math.abs(b.x + b.w / 2 - 0.60) < 0.01)
})

test('no brief appearance for a misread (one or two frames even in the close look), or a picture (still)', () => {
  const misread = (t: number) => [A(t), ...(t >= 4000 && t < 4200 ? [face(0.8, 0.9)] : [])]
  const reg = framesOf(DUR, misread)
  const close = closeOf([[3700, 4600]], misread)
  assert.equal(findAngles([...reg, ...close], [], DUR)[0].brief, undefined)
  const poster = withB(0.60, { still: true })
  assert.equal(findAngles([...framesOf(DUR, poster), ...closeOf([[3550, 4950]], poster)], [], DUR)[0].brief, undefined)
  // an unsure face (a blur read 0.7) isn't one
  const blur = withB(0.60, { score: 0.7 })
  assert.equal(findAngles([...framesOf(DUR, blur), ...closeOf([[3550, 4950]], blur)], [], DUR)[0].brief, undefined)
})

test('Make my clips: the Vertical stays, its view widens to take B in for that moment, and back', () => {
  const close = closeOf(closeLookWindows(regular, [], DUR), scene)
  const plain = formatClip(regular, [], DUR, [], SIZE)
  const parts = formatClip([...regular, ...close], [], DUR, [], SIZE)
  assert.deepEqual(parts.map(p => [p.start_ms, p.end_ms, p.layout]), plain.map(p => [p.start_ms, p.end_ms, p.layout]), 'the same sections and formats')
  assert.equal(parts[0].layout, 'vertical')
  const kfs = parts[0].slotKfs[0], before = plain[0].slotKfs[0]
  const B = face(0.60, 0.9)
  // inside the moment the view holds A and B whole; outside it, as without the close looks
  for (const t of [4100, 4300]) { const v = viewAt(kfs, t); assert.ok(within(B, v) && within(A(t), v), `${t}: ${JSON.stringify(v)}`) }
  for (const t of [1000, 3700, 4900, 8000]) assert.deepEqual(viewAt(kfs, t), viewAt(before, t), String(t))
  // a cut out and back (a keyframe 1 ms before each change), the same 9:16 shape
  const v = viewAt(kfs, 4200), v0 = viewAt(before, 4200)
  assert.ok(Math.abs(v.w / v.h - v0.w / v0.h) < 1e-6, 'same shape')
  assert.ok(Math.abs(viewAt(kfs, 3950).w - v0.w) < 1e-6 || viewAt(kfs, 3950).w === v.w, 'no slow zoom before it')
})

test('too far to take in with one 9:16 view: the view stays as it was (no other format for half a second)', () => {
  const far = withB(0.95)
  const reg = framesOf(DUR, far), close = closeOf(closeLookWindows(reg, [], DUR), far)
  const plain = formatClip(reg, [], DUR, [], SIZE)
  const parts = formatClip([...reg, ...close], [], DUR, [], SIZE)
  assert.deepEqual(parts.map(p => p.layout), plain.map(p => p.layout))
  assert.deepEqual(parts[0].slotKfs, plain[0].slotKfs)
})

test('a Split: the view whose strip holds B widens inside its strip, never over the other view', () => {
  const two = (t: number) => [face(0.25, 0.05), face(0.72, 0.9), ...(t >= 4000 && t < 4500 ? [face(0.86, 0.6)] : [])]
  const reg = framesOf(DUR, two), close = closeOf(closeLookWindows(reg, [], DUR), two)
  const plain = formatClip(reg, [], DUR, [], SIZE)
  const parts = formatClip([...reg, ...close], [], DUR, [], SIZE)
  assert.equal(parts[0].layout, plain[0].layout)
  assert.equal(parts[0].layout, 'split')
  const at = 4250, views = parts[0].slotKfs.map(k => viewAt(k, at)), old = plain[0].slotKfs.map(k => viewAt(k, at))
  const B = face(0.86, 0.6)
  const s = views.findIndex(v => within(B, v))
  assert.ok(s >= 0, JSON.stringify(views))
  assert.deepEqual(views[1 - s], old[1 - s], 'the other view is untouched')
  const [l, r] = [views[0], views[1]].sort((p, q) => p.x - q.x)
  assert.ok(l.x + l.w <= r.x + 1e-6, 'no overlap')
})

test('widened(): same shape, the face whole, inside the picture and the strip; null when it can’t', () => {
  const view = { x: 0.3, y: 0.1, w: 0.2, h: 0.632 }
  const f = face(0.58, 0.9)
  const v = widened(view, [face(0.40, 0.05)], f)!
  assert.ok(v && within(f, v) && v.x >= 0 && v.y >= 0 && v.x + v.w <= 1 && v.y + v.h <= 1)
  assert.ok(Math.abs(v.w / v.h - view.w / view.h) < 1e-9)
  assert.equal(widened(view, [face(0.40, 0.05)], f, 0, 0.55), null, 'the strip ends before the face')
})
