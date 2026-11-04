import { test } from 'node:test'
import assert from 'node:assert/strict'
import type { FrameDetection, CropBox } from './ai_edit.js'

// track_subjects reads the database and storage settings when it loads; the functions tested here use neither
process.env.DATABASE_URL ??= 'postgres://test:test@127.0.0.1:1/test'
process.env.R2_ENDPOINT ??= 'http://127.0.0.1:1'
process.env.R2_ACCESS_KEY_ID ??= 'test'
process.env.R2_SECRET_ACCESS_KEY ??= 'test'
process.env.R2_BUCKET ??= 'test'
const { shotSpans, settleGroups, whoIs, followSubjects, findAngles, findPeople, withoutPictures, faceCuts } = await import('./track_subjects.js')

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

// ── Pictures on the wall vs people sitting still (face_detect.py people mode marks a still face; the
// scan drops it only when who it is is never seen moving: withoutPictures) ──
const still = (f: CropBox): CropBox => ({ ...f, still: true })

test('a listener sitting still in a two-shot stays when they are seen moving elsewhere in the clip (2026-10-29)', () => {
  // L's close-up, talking (0–5 s); then the two-shot: L still, listening, beside C who talks (5–10 s)
  const fw = made(10000, (t, n) => (t < 5000
    ? [person(0, 0.5, 0.05, n, 0.14)]
    : [still(person(0, 0.29, 0.3, n, 0.09)), person(1, 0.77, 0.6, n, 0.11)]))
  const kept = withoutPictures(fw, [5000])
  assert.equal(kept[30].faces.length, 2, 'L is kept in the two-shot')
  const angles = findAngles(kept, [5000], 10000)
  assert.equal(angles.find(a => a.faces.some(f => Math.abs(f.x + f.w / 2 - 0.77) < 0.02))!.people, 2)
})

test('a picture on the wall (a face that never moves anywhere) is still dropped', () => {
  // P talks all through; a poster's face on the wall behind, still the whole clip
  const fw = made(10000, (t, n) => [person(0, 0.6, 0.05, n, 0.12), still(person(5, 0.15, 0.7, n, 0.08))])
  const kept = withoutPictures(fw, [])
  assert.ok(kept.every(f => f.faces.length === 1 && Math.abs(f.faces[0].x + f.faces[0].w / 2 - 0.6) < 0.01))
  assert.equal(findAngles(kept, [], 10000)[0].people, 1)
})

test('a still face whose identity can’t be read is judged on its stillness, as before', () => {
  const fw = made(8000, (t, n) => [person(0, 0.6, 0.05, n, 0.12), still(face(0.15, 0.7, 0.08))])
  assert.ok(withoutPictures(fw, []).every(f => f.faces.length === 1))
})

test('nothing still: the scan is unchanged', () => {
  assert.equal(withoutPictures(pq, pqCuts), pq)
})

test('a blurred thing read as a face (unsure), "moving" only in stretches too short to judge, is still dropped', () => {
  // P talks all through; behind her a blur the detector half-reads as a face (0.72): still for 4 s, with
  // 1-s stretches between that were never judged
  const fw = made(12000, (t, n) => [person(0, 0.4, 0.05, n, 0.12),
    { ...(t % 5000 < 4000 ? still(person(5, 0.62, 0.7, n, 0.08)) : person(5, 0.62, 0.7, n, 0.08)), score: 0.72 }])
  assert.ok(withoutPictures(fw, []).filter((_, i) => (i * 250 + 100) % 5000 < 4000).every(f => f.faces.length === 1))
})

test('a listener seen only in one shot, side-on (matching no one), is a person: their head moves a little', () => {
  // C talks; L sits still beside him, read as still, his face's box moving a few thousandths (breathing, nods)
  const wobble = (n: number) => 0.004 * Math.sin(n * 1.7)
  const fw = made(6000, (t, n) => [still({ ...person(6, 0.3, 0.3, n, 0.09), x: 0.3 - 0.045 + wobble(n), y: 0.3 + wobble(n + 3) }), person(1, 0.77, 0.6, n, 0.11)])
  assert.ok(withoutPictures(fw, []).every(f => f.faces.length === 2))
  assert.equal(findAngles(withoutPictures(fw, []), [], 6000)[0].people, 2)
})

test('a still face the user marked is a person, whatever it reads (+ Add)', () => {
  // The poster case — a face that never moves anywhere — but the user clicked it at 2.1 s
  const fw = made(8000, (t, n) => [person(0, 0.6, 0.05, n, 0.12), still(person(5, 0.15, 0.7, n, 0.08))])
  assert.ok(withoutPictures(fw, []).every(f => f.faces.length === 1), 'dropped unmarked')
  const kept = withoutPictures(fw, [], [{ t: 2100, x: 0.15, y: 0.36 }])
  assert.ok(kept.every(f => f.faces.length === 2), 'kept once marked')
  const [r] = followSubjects(kept, [], [{ id: 'l', t: 2100, x: 0.15, y: 0.36 }])
  assert.equal(r.error, undefined)
})

test('scenes: a 0.76 s shot of people seen nowhere else is a scene of its own (not the other shots); a flash is not', () => {
  const fw = made(15000, (t, n) => t < 9723 ? [person(0, 0.31, 0.05, n, 0.1), person(1, 0.74, 0.9, n, 0.1)]
    : t < 10483 ? [person(2, 0.17, 0.6, n, 0.096), person(3, 0.71, 0.7, n, 0.076)]
    : t < 10700 ? [person(4, 0.4, 0.2, n, 0.1)] : [person(5, 0.5, 0.3, n, 0.12)])
  const angles = findAngles(fw, [9723, 10483, 10700], 15000)
  const short = angles.find(a => a.spans.some(([s, e]) => s === 9723 && e === 10483))!
  assert.equal(short.people, 2)
  assert.ok(angles.find(a => a.spans.some(([s, e]) => s === 10483 && e === 10700))!.people === 0, 'a 0.2 s flash goes with the other shots')
  assert.deepEqual(angles.filter(a => a.people).map(a => a.spans[0][0]), [0, 9723, 10700], 'scenes in the order they come up')
})

// ── Cuts FFmpeg misses (animated wipes, soft cuts), found from the faces; each place's own people (2026-11-03:
// an audition show's clapperboard wipe put the judges' panel into the man's stage shot — his places got a Split for six) ──
/** The man on stage (0–9.5 s, 10–20 s); between, a wipe to a panel of four (9.5–10.5 s) with no cut FFmpeg saw */
const stage = made(20000, (t, n) => t >= 8500 && t < 9500 ? [0.2, 0.39, 0.61, 0.77].map((x, k) => person(4 + k, x, 0.3 + k * 0.1, n, 0.035))
  : [person(0, 0.5 + 0.04 * Math.sin(t / 700), 0.05, n, 0.055)])

test('faceCuts: a wipe from one man to a panel of four and back are cuts; the man moving about is not', () => {
  const found = faceCuts(stage, [])
  assert.equal(found.length, 2, JSON.stringify(found))
  assert.ok(Math.abs(found[0] - 8475) <= 130 && Math.abs(found[1] - 9475) <= 130, JSON.stringify(found))
})

test('faceCuts: one man turning side-on (his track lost, read as someone else) in the same place is not a cut', () => {
  // A new identity every second (side-on reads), always in about the same place
  const turning = made(10000, (t, n) => [person(Math.floor(t / 1000) % 2 ? 1 : 0, 0.5 + 0.07 * Math.sin(t / 300), 0.05, n, 0.06)])
  assert.deepEqual(faceCuts(turning, []), [])
})

test('each place has its own people: the man\'s places show only him, though the scene took in a panel', () => {
  // The panel stretch sits inside the man's shot (no cut at all): the scene has five people, his places one
  const angles = findAngles(stage, [], 20000)
  const all = [...faceCuts(stage, [])]
  const split = findAngles(stage, all, 20000)
  const man = split.find(a => a.spans.some(([s]) => s === 0))!
  assert.equal(man.people, 1, 'with the faces\' cuts, the man is a scene of one')
  assert.ok(angles[0].people >= 1)
})

test('two people sitting close together are two people (one face a place in a frame)', () => {
  const close = made(8000, (t, n) => [person(0, 0.12, 0.05, n, 0.03), person(1, 0.165, 0.6, n, 0.03), person(2, 0.6, 0.9, n, 0.05)])
  assert.equal(findAngles(close, [], 8000)[0].people, 3)
})

test('present: a scene\'s place shows only the people seen there (even with a cut missed elsewhere)', () => {
  // The man (0–5 s); someone else (5–7 s); the man again (7–12 s) with a panel of four wiped in at 8.5–9.5 s —
  // that cut not found (only FFmpeg's): the scene takes the panel in, its first place shows only him
  const man = (n: number) => person(0, 0.5, 0.05, n, 0.055)
  const fw = made(12000, (t, n) => t < 5000 ? [man(n)] : t < 7000 ? [person(9, 0.3, 0.8, n, 0.12)]
    : t >= 8500 && t < 9500 ? [0.2, 0.39, 0.61, 0.77].map((x, k) => person(4 + k, x, 0.3 + k * 0.1, n, 0.035)) : [man(n)])
  const scene = findAngles(fw, [5000, 7000], 12000).find(a => a.spans.some(([s]) => s === 0))!
  assert.equal(scene.spans.length, 2, JSON.stringify(scene.spans))
  assert.equal(scene.people, 5)
  const m = scene.faces.findIndex(f => Math.abs(f.x + f.w / 2 - 0.5) < 0.02)
  assert.deepEqual(scene.present?.[0], [m], 'the first place: only the man')
  assert.equal(scene.present?.[1].length, 5, 'the second place: the man and the panel')
})

// ── Who talks: each stretch of a voice by the lips, one unbroken piece of talk one person, a voice that is
// someone's own not given to them elsewhere on lip flicker (2026-11-04: Follow the speaker showed the wrong man) ──
const { speakingTurns } = await import('./track_subjects.js')
/** Frames 4 a second over `dur`; `at(t)` → the people seen then: [who, lip-gap moving?] */
function talkScene(dur: number, at: (t: number) => Array<[string, boolean]>) {
  const n = Math.ceil(dur / 250), timeOf = (fi: number) => fi * 250 + 100
  const faceOf = (s: string, fi: number) => {
    const here = at(timeOf(fi)).find(([w]) => w === s)
    if (!here) return undefined
    return { x: 0.4, y: 0.3, w: 0.1, h: 0.16, score: 0.95, lip: here[1] ? 0.02 + 0.03 * (fi % 2) : 0.02 } as CropBox
  }
  return { n, timeOf, faceOf }
}
const wordsOf = (a: number, b: number, label: string) => Array.from({ length: Math.floor((b - a) / 400) }, (_, k) => ({ start_ms: a + k * 400, end_ms: a + k * 400 + 300, speaker_id: label }))
const turnsSummary = (m: Map<string, Array<[number, number]>>) => Object.fromEntries([...m].map(([s, ts]) => [s, ts.map(([a, b]) => [Math.round(a / 1000), Math.round(b / 1000)])]))

test('who talks: one voice label over two men (the transcript merged them) — each close-up\'s talk goes to its own man', () => {
  // P's close-up talking 0–5 s, Q's close-up talking 6–11 s; every word labelled spk1
  const sc = talkScene(11000, t => (t < 5500 ? [['P', true]] : [['Q', true]]))
  const turns = speakingTurns(sc.n, sc.timeOf, ['P', 'Q'], sc.faceOf, [...wordsOf(0, 5000, 'spk1'), ...wordsOf(6000, 11000, 'spk1')], [5500])
  const sum = turnsSummary(turns)
  assert.ok(sum.P.length && sum.P.every(([, b]) => b <= 6), JSON.stringify(sum))
  assert.ok(sum.Q.length && sum.Q.every(([a]) => a >= 5), JSON.stringify(sum))
})

test('who talks: one unbroken piece of talk over a wide shot, his close-up and a wide shot is all his', () => {
  // A wide shot (both small, lips unreadable) 0–3 s, Q's close-up talking 3–6 s, the wide shot again 6–9 s — Q talks on
  const sc = talkScene(9000, t => (t >= 3000 && t < 6000 ? [['Q', true]] : [['P', false], ['Q', false]]))
  const turns = speakingTurns(sc.n, sc.timeOf, ['P', 'Q'], sc.faceOf, wordsOf(0, 9000, 'spk1'), [3000, 6000])
  assert.deepEqual(turnsSummary(turns).P, [])
  assert.equal(turnsSummary(turns).Q.reduce((n, [a, b]) => n + b - a, 0) >= 8, true, JSON.stringify(turnsSummary(turns)))
})

test('who talks: a man whose own voice is another label doesn\'t take this voice on a flicker of his lips', () => {
  // P talks (spk1) in his close-up 0–5 s; Q (spk2) talks in his close-up 6–10 s; then a wide shot 11–15 s where
  // Q talks on (spk2) but only P's lips flicker
  const sc = talkScene(15000, t => (t < 5500 ? [['P', true]] : t < 10500 ? [['Q', true]] : [['P', true], ['Q', false]]))
  const turns = speakingTurns(sc.n, sc.timeOf, ['P', 'Q'], sc.faceOf, [...wordsOf(0, 5000, 'spk1'), ...wordsOf(6000, 10000, 'spk2'), ...wordsOf(11000, 15000, 'spk2')], [5500, 10500])
  const sum = turnsSummary(turns)
  assert.ok(sum.P.every(([, b]) => b <= 6), `P has no turn in the wide shot: ${JSON.stringify(sum)}`)
  assert.ok(sum.Q.some(([a, b]) => a <= 11 && b >= 14), JSON.stringify(sum))
})
