import { test } from 'node:test'
import assert from 'node:assert/strict'
import type { FrameDetection, CropBox } from './ai_edit.js'

// clip_formats loads the worker's jobs, which read the database and storage settings when they load; the
// functions tested here use neither
process.env.DATABASE_URL ??= 'postgres://test:test@127.0.0.1:1/test'
process.env.R2_ENDPOINT ??= 'http://127.0.0.1:1'
process.env.R2_ACCESS_KEY_ID ??= 'test'
process.env.R2_SECRET_ACCESS_KEY ??= 'test'
process.env.R2_BUCKET ??= 'test'
const { formatClip } = await import('./clip_formats.js')

// ── A made-up interview, 4 frames a second, 16:9 1080p ──
const SIZE = { width: 1920, height: 1080 }
const sig = (d: number) => [0.5 - d / 2, 0.1 + d / 2, ...Array.from({ length: 22 }, () => 0.4 / 22)]
const face = (x: number, d: number, w = 0.08): CropBox => ({ x: x - w / 2, y: 0.25, w, h: w * 1.6, score: 0.95, sig: sig(d) })
const DUR = 20000
const cuts = [6000, 12000, 15000]
/** 0–6 s: two people far apart; 6–12 s: one man walking from 0.4 to 0.6; 12–15 s: nobody (a slide); 15–20 s: the two again */
function clip(t: number): CropBox[] {
  if (t < 6000 || t >= 15000) return [face(0.25, 0.05), face(0.75, 0.9)]
  if (t < 12000) return [face(0.4 + (0.2 * (t - 6000)) / 6000, 0.05, 0.12)]
  return []
}
const frames: FrameDetection[] = []
for (let n = 0; n * 250 + 100 < DUR; n++) {
  const faces = clip(n * 250 + 100)
  frames.push({ frame_index: n, person_count: faces.length, faces })
}
const within = (f: CropBox, v: { x: number; y: number; w: number; h: number }) => f.x >= v.x - 1e-6 && f.x + f.w <= v.x + v.w + 1e-6 && f.y >= v.y - 1e-6 && f.y + f.h <= v.y + v.h + 1e-6
/** A view's place at t (the keyframe at or before it) */
const at = (kfs: Array<{ t_ms: number; x: number; y: number; w: number; h: number }>, t: number) => [...kfs].reverse().find(k => k.t_ms <= t) ?? kfs[0]

test('the clip is covered end to end, each format on its camera shot', () => {
  const parts = formatClip(frames, cuts, DUR, [], SIZE)
  assert.equal(parts[0].start_ms, 0)
  assert.equal(parts[parts.length - 1].end_ms, DUR)
  parts.slice(1).forEach((p, i) => assert.equal(p.start_ms, parts[i].end_ms))
  assert.deepEqual(parts.map(p => [p.start_ms, p.end_ms, p.layout]), [[0, 6000, 'split'], [6000, 12000, 'vertical'], [12000, 15000, 'vertical'], [15000, 20000, 'split']])
})

test('two people far apart: a Split with each whole in a view, nobody lost', () => {
  const [two] = formatClip(frames, cuts, DUR, [], SIZE)
  assert.equal(two.people, 2)
  for (const f of clip(1000)) assert.ok(two.slotKfs.some(kfs => within(f, kfs[0])), JSON.stringify(two.slotKfs))
})

test('the speaker goes at the bottom of the Split', () => {
  // The LEFT one talks the whole time: their lips move, the other's don't (with nobody talking, left is on top)
  const speech = Array.from({ length: 20 }, (_, k) => ({ start_ms: k * 300, end_ms: k * 300 + 250, speaker_id: 'L' }))
  // (who talks is known by who they are: each face with its identity, as the detector reads it)
  const emb = (k: number, n: number) => {
    const v = Array.from({ length: 16 }, (_, i) => (i === k ? 1 : 0) + 0.15 * Math.sin(n * 7 + i * 3))
    const len = Math.hypot(...v)
    return v.map(x => x / len)
  }
  const lipped = frames.map(f => ({ ...f, faces: f.faces.map((x, k) => ({ ...x, emb: emb(k, f.frame_index), lip: k === 0 ? 0.02 + 0.015 * Math.sin(f.frame_index) : 0.02 })) }))
  const [two] = formatClip(lipped, cuts, DUR, speech, SIZE)
  const left = clip(1000)[0]
  assert.ok(within(left, two.slotKfs[1][0]), 'the left person (the speaker) is the bottom view')
  // …and without anyone talking, the left one is on top
  const [quiet] = formatClip(frames, cuts, DUR, [], SIZE)
  assert.ok(within(left, quiet.slotKfs[0][0]), 'nobody talks: the left person on top')
})

test('a man walking: his one-person view follows him and keeps him whole', () => {
  const walk = formatClip(frames, cuts, DUR, [], SIZE)[1]
  assert.equal(walk.why, 'fits')
  for (const t of [6600, 9000, 11600]) {
    const v = at(walk.slotKfs[0], t)
    const f = clip(t)[0]
    // (the smoothing lags him by up to a second or so: his face stays in the view all along)
    assert.ok(f.x >= v.x && f.x + f.w <= v.x + v.w, `${t}: face ${f.x.toFixed(3)} view ${v.x.toFixed(3)}–${(v.x + v.w).toFixed(3)}`)
  }
  assert.ok(at(walk.slotKfs[0], 11600).x > at(walk.slotKfs[0], 6600).x + 0.08, 'the view moved with him')
})

test('Motion off: every view holds where it was laid', () => {
  const parts = formatClip(frames, cuts, DUR, [], SIZE, { motion: false })
  for (const p of parts) for (const kfs of p.slotKfs) assert.equal(kfs.length, 1)
})

test('no faces: a still centred Vertical (a cutaway)', () => {
  const slide = formatClip(frames, cuts, DUR, [], SIZE)[2]
  assert.equal(slide.why, 'cutaway')
  assert.equal(slide.slotKfs[0].length, 1)
  assert.ok(Math.abs(slide.slotKfs[0][0].x + slide.slotKfs[0][0].w / 2 - 0.5) < 1e-6)
})

test('a following view in a Split never crosses into its neighbour’s view', () => {
  // The left one drifts right toward the middle over the first shot
  const drift = frames.map(f => {
    const t = f.frame_index * 250 + 100
    return t < 6000 ? { ...f, faces: [face(0.25 + (0.2 * t) / 6000, 0.05), face(0.75, 0.9)] } : f
  })
  const [two] = formatClip(drift, cuts, DUR, [], SIZE)
  const [p, q] = two.slotKfs
  for (let t = 0; t < 6000; t += 250) {
    const a = at(p, t), b = at(q, t)
    const [l, r] = a.x < b.x ? [a, b] : [b, a]
    assert.ok(l.x + l.w <= r.x + 1e-6, `${t}: views overlap ${JSON.stringify([a, b])}`)
  }
})
