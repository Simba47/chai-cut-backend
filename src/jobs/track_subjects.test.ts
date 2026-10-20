import { test } from 'node:test'
import assert from 'node:assert/strict'
import type { FrameDetection, CropBox } from './ai_edit.js'

// track_subjects reads the database and storage settings when it loads; the functions tested here use neither
process.env.DATABASE_URL ??= 'postgres://test:test@127.0.0.1:1/test'
process.env.R2_ENDPOINT ??= 'http://127.0.0.1:1'
process.env.R2_ACCESS_KEY_ID ??= 'test'
process.env.R2_SECRET_ACCESS_KEY ??= 'test'
process.env.R2_BUCKET ??= 'test'
const { shotSpans, settleGroups } = await import('./track_subjects.js')

// ── A made-up clip, 4 frames a second ──
// A colour signature (24 numbers summing to 1) `d` away (L1) from person A's
const sig = (d: number) => [0.5 - d / 2, 0.1 + d / 2, ...Array.from({ length: 22 }, () => 0.4 / 22)]
const face = (x: number, d: number, w = 0.11): CropBox => ({ x: x - w / 2, y: 0.3, w, h: w * 1.4, score: 0.95, sig: sig(d) })
const DUR = 25000
const cuts = [5000, 10000, 15000, 20000]
/** The faces at a moment (ms) */
function clip(t: number): CropBox[] {
  if (t < 5000) return [face(0.5, 0.05)]                                  // A, still
  if (t < 10000) return [face(0.3, 0.9), face(0.7, 0.85)]                 // a two-shot of other people
  if (t < 15000) return [face(t < 12000 ? 0.5 : 0.43, 0.08)]              // A again: leans 7% for the last 3 s
  if (t < 20000) return [face(0.5, Math.floor(t / 250) % 2 ? 0.47 : 0.53)] // B: someone else, in black too, in A's place
  return [face(0.5, 0.05), ...(t >= 21000 && t < 24000 ? [face(0.2, 0.9)] : [])] // A; someone walks in for 3 s
}
const frames: FrameDetection[] = []
for (let n = 0; n * 250 + 100 < DUR; n++) {
  const faces = clip(n * 250 + 100)
  frames.push({ frame_index: n, person_count: faces.length, faces })
}

test('someone leaning inside a still shot keeps the whole shot (the group’s format reaches the cut)', () => {
  const { spans } = shotSpans(frames, cuts, 1000, DUR)
  assert.ok(spans.some(([a, b]) => a === 10000 && b === 15000), JSON.stringify(spans))
})

test('another person in the same place, reading near the line frame to frame, is not the group (not even part of the shot)', () => {
  const { spans } = shotSpans(frames, cuts, 1000, DUR)
  assert.ok(!spans.some(([a, b]) => a < 20000 && b > 15000), JSON.stringify(spans))
})

test('someone walking in for a while still splits a shot', () => {
  const { spans } = shotSpans(frames, cuts, 1000, DUR)
  const late = spans.filter(([a]) => a >= 20000)
  // (the edges fall within a frame of the walk-in: frames are a quarter second apart)
  assert.ok(late.length === 2 && late.every(([a, b]) => b <= 21250 || a >= 23750), JSON.stringify(spans))
})

test('the group starts and ends on camera cuts only', () => {
  const { spans } = shotSpans(frames, cuts, 1000, DUR)
  const edges = new Set([0, ...cuts, DUR])
  for (const [a, b] of spans.filter(([x]) => x < 20000)) assert.ok(edges.has(a) && edges.has(b), JSON.stringify(spans))
})

test('two groups claiming one shot: it goes to the group whose person it is nearer to', () => {
  // Shot 2 shows a man 0.30 from A's colours and 0.36 from B's (both under the line on their own)
  const f2: FrameDetection[] = []
  const at = (t: number) => (t < 5000 ? [face(0.5, 0)] : t < 10000 ? [face(0.5, 0.66)] : [face(0.5, 0.30)])
  for (let n = 0; n * 250 + 100 < 15000; n++) f2.push({ frame_index: n, person_count: 1, faces: at(n * 250 + 100) })
  const c2 = [5000, 10000]
  const alone = [shotSpans(f2, c2, 1000, 15000), shotSpans(f2, c2, 6000, 15000)]
  assert.ok(alone.every(r => r.spans.some(([a, b]) => a <= 10000 && b >= 15000)), 'on their own, both groups take the third shot')
  const [a, b] = settleGroups(alone, c2, 15000)
  assert.deepEqual(a.spans, [[0, 5000], [10000, 15000]])
  assert.deepEqual(b.spans, [[5000, 10000]])
  assert.ok(b.path.every(([t]) => t >= 5000 && t <= 10000), 'its people’s boxes go with the stretch')
})

test('a panel: a face more where someone sits in the marked shot is not another shot (heads turning)', () => {
  // Three people marked; a fourth sits at 0.65, read only now and then. The shot comes back at 10–15 s,
  // the fourth read in most frames there
  const fp: FrameDetection[] = []
  const at = (t: number, n: number) => {
    const fourth = t < 5000 ? n % 4 === 0 : t >= 10000 && n % 4 !== 0
    if (t >= 5000 && t < 10000) return [face(0.4, 0.9)]
    return [face(0.2, 0.05, 0.06), face(0.5, 0.05, 0.06), face(0.8, 0.05, 0.06), ...(fourth ? [face(0.65, 0.05, 0.06)] : [])]
  }
  for (let n = 0; n * 250 + 100 < 15000; n++) fp.push({ frame_index: n, person_count: 0, faces: at(n * 250 + 100, n) })
  const { spans } = shotSpans(fp, [5000, 10000], 1100, 15000)
  assert.deepEqual(spans, [[0, 5000], [10000, 15000]])
})
