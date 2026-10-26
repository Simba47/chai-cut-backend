import { test } from 'node:test'
import assert from 'node:assert/strict'
import type { FrameDetection, CropBox } from './ai_edit.js'

// track_subjects reads the database and storage settings when it loads; the functions tested here use neither
process.env.DATABASE_URL ??= 'postgres://test:test@127.0.0.1:1/test'
process.env.R2_ENDPOINT ??= 'http://127.0.0.1:1'
process.env.R2_ACCESS_KEY_ID ??= 'test'
process.env.R2_SECRET_ACCESS_KEY ??= 'test'
process.env.R2_BUCKET ??= 'test'
const { shotSpans, settleGroups, whoIs, followSubjects, findAngles, findPeople } = await import('./track_subjects.js')

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

// ── Who is who: faces' identities (the detector's SFace `emb`) ──
// Person k's identity, read a little differently in each frame n (unit length; the same person ~0.9 alike, others ~0)
const emb = (k: number, n: number) => {
  const v = Array.from({ length: 16 }, (_, i) => (i === k ? 1 : 0) + 0.15 * Math.sin(n * 7 + i * 3))
  const len = Math.hypot(...v)
  return v.map(x => x / len)
}
const person = (k: number, x: number, d: number, n: number, w = 0.14): CropBox => ({ ...face(x, d, w), emb: emb(k, n) })
/** A made-up clip: `at(t, n)` the faces at a moment (ms) and frame */
function made(dur: number, at: (t: number, n: number) => CropBox[]): FrameDetection[] {
  const out: FrameDetection[] = []
  for (let n = 0; n * 250 + 100 < dur; n++) { const faces = at(n * 250 + 100, n); out.push({ frame_index: n, person_count: faces.length, faces }) }
  return out
}
// P in a close-up; Q in the same seat, colours alike (a man in black, a woman in white: 0.1 apart);
// P again from another camera (another place, other colours: the light)
const pq = made(15000, (t, n) => (t < 5000 ? [person(0, 0.5, 0.05, n)] : t < 10000 ? [person(1, 0.52, 0.15, n)] : [person(0, 0.3, 0.45, n)]))
const pqCuts = [5000, 10000]

test('who is who: the same person in another camera angle is one person; someone else whose colours read alike is not', () => {
  const who = whoIs(pq, pqCuts)
  const at = (t: number) => who.get(pq[Math.floor((t - 100) / 250)].faces[0])
  assert.equal(at(1100), at(11100))
  assert.notEqual(at(1100), at(6100))
  assert.notEqual(at(1100), undefined)
})

test('marking the same person again, in another angle, is caught (no duplicate subject)', () => {
  const [a, b] = followSubjects(pq, pqCuts, [{ id: 'a', t: 1100, x: 0.5, y: 0.4 }, { id: 'b', t: 11100, x: 0.3, y: 0.4 }])
  assert.equal(a.error, undefined)
  assert.match(b.error ?? '', /same person/)
})

test('a subject is followed into their other angles, and never onto someone else whose colours read alike', () => {
  const [a] = followSubjects(pq, pqCuts, [{ id: 'a', t: 1100, x: 0.5, y: 0.4 }])
  assert.ok(a.path.some(([t]) => t > 10000), 'seen in their other angle')
  assert.ok(!a.path.some(([t]) => t > 5000 && t < 10000), 'not on the other person in their seat')
})

test('the scenes: someone else in the same seat is another scene; each face says who it is, the same in every scene', () => {
  const angles = findAngles(pq, pqCuts, 15000)
  assert.equal(angles.length, 3)
  const [p1, q, p2] = angles.map(a => a.faces[0].person)
  assert.notEqual(p1, undefined)
  assert.equal(p1, p2)
  assert.notEqual(p1, q)
})

test('sub-edit registers each person once, whichever angles they are in', () => {
  assert.equal(findPeople(pq, pqCuts).length, 2)
})

test('a trio with one face never read through a camera shot is still the group', () => {
  // Marked with all three read; the shot comes back at 10–15 s with the middle one's head turned all through
  const trio = made(15000, t => t >= 5000 && t < 10000 ? [face(0.4, 0.9)]
    : [face(0.2, 0.05, 0.06), ...(t < 10000 ? [face(0.5, 0.05, 0.06)] : []), face(0.8, 0.05, 0.06)])
  const { spans } = shotSpans(trio, [5000, 10000], 1100, 15000)
  assert.deepEqual(spans, [[0, 5000], [10000, 15000]])
})

test('a short wide shot between close-ups is not the close-up, even with a frame on the cut showing the close-up', () => {
  // B's close-up; a 1.5 s wide shot of four (faces far smaller) at 5–6.5 s whose last frame, right on the
  // cut, already shows B again; then B
  const fw = made(10000, t => (t >= 5000 && t < 6340 ? [0.11, 0.34, 0.65, 0.9].map(x => face(x, 0.4, 0.036)) : [face(0.5, 0.05, 0.1)]))
  const { spans } = shotSpans(fw, [5000, 6500], 1100, 10000)
  assert.deepEqual(spans, [[0, 5000], [6500, 10000]])
})

test('who is in an angle: everyone seen in it counts, even someone there only part of the time (never ignore anyone)', () => {
  // One camera shot of P; Q sits in for 3 s, then R for 3 s (each 30% of the shot)
  const fw = made(10000, (t, n) => [person(0, 0.5, 0.05, n, 0.12),
    ...(t < 3000 ? [person(1, 0.15, 0.6, n, 0.12)] : []), ...(t >= 4000 && t < 7000 ? [person(2, 0.85, 0.9, n, 0.12)] : [])])
  const [angle] = findAngles(fw, [], 10000)
  assert.equal(angle.people, 3)
  assert.deepEqual(angle.faces.map(f => Math.round((f.x + f.w / 2) * 100) / 100), [0.15, 0.5, 0.85])
})

test('who is in an angle: a misread seen in only 1–2 frames is nobody (PRESENCE_MIN_FRAMES)', () => {
  // A close-up of P; for two frames the detector also "sees" a big face over the background
  const fw = made(10000, (t, n) => [person(0, 0.4, 0.05, n, 0.13), ...(n === 10 || n === 11 ? [{ ...face(0.8, 0.9, 0.3), score: 0.8 }] : [])])
  const [angle] = findAngles(fw, [], 10000)
  assert.equal(angle.people, 1)
})

test('who is in an angle: one person in two places, never seen together, counts once', () => {
  // A close-up of B in the middle; for 4 s the video's own edit shows him on the right (faces too small
  // for who-is-who: by place and timing only)
  const fw = made(12000, (t) => [face(t >= 4000 && t < 8000 ? 0.72 : 0.5, 0.05, 0.1)])
  const [angle] = findAngles(fw, [], 12000)
  assert.equal(angle.people, 1)
})
