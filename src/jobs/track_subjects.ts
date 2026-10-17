import { mkdtemp, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { tmpdir } from 'node:os'
import { r2DownloadToFile, r2SignedUrl } from '../r2.js'
import db from '../db.js'
import type { TransactionSql } from 'postgres'
import type { Job, TrackSubjectsJobPayload } from '../types.js'
import { detectFaces, detectCuts, trackFaces, probeSize, FRAME_INTERVAL_MS, type FrameDetection, type CropBox, type SpokenWord } from './ai_edit.js'
import { autoDirectChanges, followChanges, buildKeyframes, type SubjectPoint } from './subject_framing.js'
import { snapSectionBoundaries } from '../lib/snapSections.js'

// ── Subjects: people the user marked in a clip, followed through it ────────────
// In the editor the user pauses on a person and clicks them (a "subject"). This job looks at the
// clip 4× a second with the same face detection Make my clips uses, takes the face under each
// click, and follows that person: frame to frame by where their face is (trackFaces), and after a
// camera cut or when they come back into the picture by the colours of their face and clothes
// (the detector's signature). The result is each subject's face, frame by frame, wherever they
// were seen. It also works out who is talking, moment to moment — the person whose mouth moves
// with the speech (the transcript's words), as Make my clips decides it — so the editor can switch
// the view to whoever is talking through the whole clip.

/** Two signatures closer than this (L1 distance; each sums to 1) are the same person (as personShots) */
const SAME_PERSON = 0.5
/** A track this much closer to one subject than to another goes to it; nearer than that is unclear */
const CLEAR_MARGIN = 0.05
/** Marked from the view: a face whose middle is at most this far across from the view's middle (fraction of the width) */
const VIEW_MARK_REACH = 0.25
/** The face under a click is looked for this many frames either side of the moment (half a second) */
const MARK_SEARCH_FRAMES = 2
/**
 * Detection frame n shows the picture from about n × 250 ms + this: FFmpeg's fps filter keeps a
 * frame from late in each quarter second (measured 2026-10-12 on 24, 25 and 30 fps video: +50 to
 * +160 ms, about +100). Without it a face would show ~0.1 s early — on the other side of a camera cut.
 */
export const FRAME_SHOWS_MS = 100

/** Where the user clicked: ms from the start of the scan, and fractions of the frame */
export interface SubjectMark { id: string; t: number; x: number; y: number }
/**
 * [ms from the start of the scan, x, y, w, h, talking] of the subject's face (fractions of the
 * frame); talking: 1 while they are the one talking, else 0
 */
export type PathPoint = [number, number, number, number, number, number]
/** [start, end] ms (from the start of the scan) while the subject is talking */
export type Turn = [number, number]
export interface SubjectResult { id: string; path: PathPoint[]; speaking: Turn[]; error?: string }

// Who talks when. The transcript's words carry a speaker label: each label is matched to the person
// whose mouth moves with that label's words (the most words where their lips clearly lead), so the
// talking turns start and end on the words themselves — the view can cut on the first word.
/** A person's lips "lead" on a word when they move at least this much (std of lip gap / face height over ±0.5 s) */
const MIN_LIP_MOVE = 0.0015
/** …and this much more than anyone else's */
const LIP_LEAD = 1.3
/** A label belongs to a person who led on at least this many of its words (and this share of them) */
const LABEL_MIN_WORDS = 2
const LABEL_MIN_SHARE = 0.25
/** A pause shorter than this inside one person's talking doesn't end their turn */
const TURN_GAP_MS = 700

const pstdev = (v: number[]) => {
  const m = v.reduce((a, b) => a + b, 0) / v.length
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length)
}

/**
 * Each subject's talking turns. `faceOf(s, fi)`: the subject's face in frame fi (if seen there);
 * `timeOf(fi)`: that frame's time. A word goes to its label's person; a word without one (or whose
 * label matched nobody) to the person whose lips clearly lead on it; failing that, when only one of
 * them is in the picture, to them.
 */
export function speakingTurns(
  frameCount: number, timeOf: (fi: number) => number, subjects: string[],
  faceOf: (s: string, fi: number) => CropBox | undefined, speech: SpokenWord[],
): Map<string, Turn[]> {
  const out = new Map<string, Turn[]>(subjects.map(s => [s, []]))
  if (!speech.length || !subjects.length || !frameCount) return out
  // How much each subject's mouth moves around a frame
  const motion = (s: string, fi: number): number | null => {
    const v: number[] = []
    for (let k = fi - 2; k <= fi + 2; k++) {
      if (k < 0 || k >= frameCount) continue
      const lip = faceOf(s, k)?.lip
      if (typeof lip === 'number') v.push(lip)
    }
    return v.length >= 3 ? pstdev(v) : null
  }
  // The frames over each word (a quarter second either side: frames are 4 a second)
  const framesOf = (w: SpokenWord) => {
    const out: number[] = []
    for (let fi = 0; fi < frameCount; fi++) { const t = timeOf(fi); if (t >= w.start_ms - 125 && t <= w.end_ms + 125) out.push(fi) }
    return out
  }
  const words = speech.map(w => {
    const fis = framesOf(w)
    const seen = subjects.filter(s => fis.some(fi => faceOf(s, fi)))
    const moves = seen.map(s => ({ s, m: Math.max(0, ...fis.map(fi => motion(s, fi) ?? 0)) })).sort((a, b) => b.m - a.m)
    const lead = moves[0] && moves[0].m >= MIN_LIP_MOVE && (!moves[1] || moves[0].m >= LIP_LEAD * moves[1].m) ? moves[0].s : null
    return { w, seen, lead }
  })
  // Each label: the person who leads on most of its words
  const labelTo = new Map<string, string>()
  const byLabel = new Map<string, typeof words>()
  for (const x of words) if (x.w.speaker_id) byLabel.set(x.w.speaker_id, [...(byLabel.get(x.w.speaker_id) ?? []), x])
  for (const [label, list] of byLabel) {
    const votes = new Map<string, number>()
    for (const x of list) if (x.lead) votes.set(x.lead, (votes.get(x.lead) ?? 0) + 1)
    const best = [...votes].sort((a, b) => b[1] - a[1])[0]
    if (best && best[1] >= LABEL_MIN_WORDS && best[1] >= LABEL_MIN_SHARE * list.length) labelTo.set(label, best[0])
  }
  const whos = words.map(x => (x.w.speaker_id ? labelTo.get(x.w.speaker_id) : undefined) ?? x.lead ?? (x.seen.length === 1 ? x.seen[0] : null))
  // A word with no clear talker right before someone's words is theirs: at a hand-over both mouths
  // move within the lip readings' half second, so a turn's first word is often unclear by lips alone
  for (let i = whos.length - 2; i >= 0; i--) {
    if (!whos[i] && whos[i + 1] && words[i + 1].w.start_ms - words[i].w.end_ms < TURN_GAP_MS) whos[i] = whos[i + 1]
  }
  for (const [i, x] of words.entries()) {
    const who = whos[i]
    if (!who) continue
    const turns = out.get(who)!
    const last = turns[turns.length - 1]
    if (last && x.w.start_ms - last[1] < TURN_GAP_MS) last[1] = Math.max(last[1], x.w.end_ms)
    else turns.push([x.w.start_ms, x.w.end_ms])
  }
  // Two people never talk at once in the turns: where they overlap, the one who started later has the floor
  const all = [...out].flatMap(([s, ts]) => ts.map(t => ({ s, t })))
  for (const a of all) for (const b of all) {
    if (a === b || a.s === b.s) continue
    if (b.t[0] > a.t[0] && b.t[0] < a.t[1]) a.t[1] = b.t[0]
  }
  for (const [s, ts] of out) out.set(s, ts.filter(t => t[1] - t[0] > 0))
  return out
}

const l1 = (a: number[], b: number[]) => a.reduce((t, v, i) => t + Math.abs(v - (b[i] ?? 0)), 0)
const meanOf = (sigs: number[][]) => sigs[0].map((_, k) => sigs.reduce((t, v) => t + v[k], 0) / sigs.length)
const round4 = (v: number) => Math.round(v * 10000) / 10000

/**
 * How well a click points at a face: the face itself, or the person under it (their body is below
 * their head, a little wider). Lower is better; null when the click is not on that person.
 */
function clickScore(face: CropBox, x: number, y: number): number | null {
  const cx = face.x + face.w / 2
  const left = face.x - face.w * 0.9, right = face.x + face.w * 1.9
  if (x < left || x > right || y < face.y - face.h * 0.6) return null
  const onFace = x >= face.x && x <= face.x + face.w && y >= face.y && y <= face.y + face.h
  return Math.abs(x - cx) / face.w + (onFace ? 0 : 0.5)
}

/**
 * Each mark's person, frame by frame. `frames` are 4 a second from the start of the scan, `cuts`
 * the camera cuts (ms from the start). A subject is never two faces in one frame, and a face
 * belongs to one subject at most. `speech` (the transcript's words, ms from the start; with the
 * frames' lip readings) tells who is talking when; without it nobody is marked as talking.
 */
export function followSubjects(frames: FrameDetection[], cuts: number[], marks: SubjectMark[], speech: SpokenWord[] = [], durationMs = 0): SubjectResult[] {
  // Real faces only (motion boxes have no face to recognise and wander)
  const fr: FrameDetection[] = [...frames]
    .sort((a, b) => a.frame_index - b.frame_index)
    .map(f => ({ ...f, faces: f.faces.filter(face => (face.score ?? 0) > 0) }))
  // (trackFaces times frames as frame_index × 250 ms: the cuts are moved to match)
  const ids = trackFaces(fr, cuts.map(c => c - FRAME_SHOWS_MS))
  // Every track: the frames it is in (frame position → face) and its colours
  const tracks = new Map<number, Map<number, CropBox>>()
  fr.forEach((f, fi) => ids[fi].forEach((id, k) => {
    const m = tracks.get(id) ?? new Map<number, CropBox>()
    m.set(fi, f.faces[k])
    tracks.set(id, m)
  }))
  const sigOf = new Map<number, number[]>()
  for (const [id, m] of tracks) {
    const sigs = [...m.values()].map(f => f.sig).filter((s): s is number[] => !!s && s.length > 0)
    if (sigs.length) sigOf.set(id, meanOf(sigs))
  }

  // The track under each click
  const timeOf = (fi: number) => fr[fi].frame_index * FRAME_INTERVAL_MS + FRAME_SHOWS_MS
  // A camera cut between two moments: a face there is in another shot
  const acrossCut = (a: number, b: number) => cuts.some(c => (a < c) !== (b < c))
  const marked = new Map<string, number>()
  const errors = new Map<string, string>()
  for (const mk of marks) {
    if (!fr.length) { errors.set(mk.id, 'No faces could be read in this part of the video.'); continue }
    let near = 0
    fr.forEach((_, fi) => { if (Math.abs(timeOf(fi) - mk.t) < Math.abs(timeOf(near) - mk.t)) near = fi })
    let best: { id: number; score: number; off: number } | null = null
    for (let off = 0; off <= MARK_SEARCH_FRAMES; off++) {
      for (const fi of off ? [near - off, near + off] : [near]) {
        if (fi < 0 || fi >= fr.length || acrossCut(mk.t, timeOf(fi))) continue
        fr[fi].faces.forEach((face, k) => {
          const s = clickScore(face, mk.x, mk.y)
          if (s === null) return
          // Nearer in time wins over a slightly better aim further away
          const score = s + off * 0.3
          if (!best || score < best.score) best = { id: ids[fi][k], score, off }
        })
      }
      if (best) break
    }
    // Marked from the view (its middle, not a click on the face): the face nearest that middle
    // across the picture, at the moment (or just around it)
    if (!best) {
      for (let off = 0; off <= MARK_SEARCH_FRAMES && !best; off++) {
        for (const fi of off ? [near - off, near + off] : [near]) {
          if (fi < 0 || fi >= fr.length || acrossCut(mk.t, timeOf(fi))) continue
          fr[fi].faces.forEach((face, k) => {
            const d = Math.abs(face.x + face.w / 2 - mk.x)
            if (d > VIEW_MARK_REACH) return
            const score = d + off * 0.1
            if (!best || score < best.score) best = { id: ids[fi][k], score, off }
          })
        }
      }
    }
    if (!best) { errors.set(mk.id, 'No face in the view here. Put the view on their face (where it can be seen) and mark them again.'); continue }
    const id = (best as { id: number }).id
    const other = [...marked].find(([, t]) => t === id)
    if (other) { errors.set(mk.id, 'This is the same person as another subject.'); continue }
    marked.set(mk.id, id)
  }

  // The other tracks, given to the subject whose colours they match (clearly), nearest first
  const subjects = [...marked.keys()]
  const owned = new Map<string, Set<number>>(subjects.map(s => [s, new Set([marked.get(s)!])]))
  const taken = new Set(marked.values())
  const framesOf = (s: string) => {
    const set = new Set<number>()
    for (const id of owned.get(s)!) for (const fi of tracks.get(id)!.keys()) set.add(fi)
    return set
  }
  const offers: Array<{ id: number; s: string; d: number }> = []
  for (const [id, sig] of sigOf) {
    if (taken.has(id)) continue
    const ds = subjects
      .map(s => ({ s, d: sigOf.has(marked.get(s)!) ? l1(sig, sigOf.get(marked.get(s)!)!) : Infinity }))
      .sort((a, b) => a.d - b.d)
    const [first, second] = ds
    if (!first || first.d >= SAME_PERSON) continue
    if (second && second.d < SAME_PERSON && second.d - first.d < CLEAR_MARGIN) continue
    offers.push({ id, s: first.s, d: first.d })
  }
  offers.sort((a, b) => a.d - b.d)
  for (const o of offers) {
    // One person is never two faces at once: a track seen at the same time as theirs is someone else
    const busy = framesOf(o.s)
    if ([...tracks.get(o.id)!.keys()].some(fi => busy.has(fi))) continue
    owned.get(o.s)!.add(o.id)
  }

  // Each subject's face, frame by frame
  const faces = new Map<string, Map<number, CropBox>>()
  for (const s of subjects) {
    const byFrame = new Map<number, CropBox>()
    for (const id of owned.get(s)!) for (const [fi, face] of tracks.get(id)!) byFrame.set(fi, face)
    faces.set(s, byFrame)
  }
  // Who is talking when, to the word (durationMs: kept for the callers; the words carry the times)
  void durationMs
  const turns = speakingTurns(fr.length, timeOf, subjects, (s, fi) => faces.get(s)?.get(fi), speech)
  const talksAt = (s: string, t: number) => (turns.get(s) ?? []).some(([a, b]) => t >= a && t <= b)

  return marks.map(mk => {
    const err = errors.get(mk.id)
    if (err) return { id: mk.id, path: [], speaking: [], error: err }
    const byFrame = faces.get(mk.id)!
    const path = [...byFrame.entries()].sort((a, b) => a[0] - b[0])
      .map(([fi, f]): PathPoint => [timeOf(fi), round4(f.x), round4(f.y), round4(f.w), round4(f.h), talksAt(mk.id, timeOf(fi)) ? 1 : 0])
    return { id: mk.id, path, speaking: turns.get(mk.id) ?? [] }
  })
}

// ── Group subjects: a camera shot marked with its own format ────────────────────
// The user marks a shot (say Split, its top view on two people and its bottom view on the third):
// the shot is the faces in the picture at that moment — how many, where, and whose (the colours of
// face and clothes, the detector's signature). Wherever the clip shows the same shot again (the same
// camera, the same people in the same places; one more or one fewer is another shot, and so are
// other people sitting in the same places), the editor gives it that format and those views.

/** A face is the marked shot's face when its middle is this close (fractions of the frame) and its size about the same */
const SHOT_REACH_X = 0.06
const SHOT_REACH_Y = 0.08
const SHOT_SIZE_RATIO = 1.45
/**
 * The same people: the faces' signatures, on average over the shot's faces, this close to the marked
 * ones (L1; each sums to 1). Two interviews are often shot alike — three people sitting left, middle
 * and right — so the places alone can't tell them apart. Measured 2026-10-13 on such a clip: the
 * same people 0.13–0.28 a frame (never over 0.45), other people 0.7–0.9. One face alone is too noisy
 * to judge (a turned head reads 0.5 and more), the average of the shot's faces isn't.
 */
const SHOT_SAME_PEOPLE = 0.5
/** A break in the shot shorter than this, inside one camera shot, isn't one (someone leaned out of place for a moment) */
const SHOT_JOIN_MS = 1000
/** A face smaller than this share of the marked shot's smallest face is background (a poster, a face far behind) */
const SHOT_BACKGROUND = 0.5
/** Inside one camera shot, something else for this long is another shot (shorter: the face reader's misreads) */
const SHOT_CHANGE_MS = 2000
/** A stretch of the shot shorter than this isn't one */
const SHOT_MIN_MS = 800

export interface ShotResult {
  spans: Turn[]; path: PathPoint[]; error?: string
  /** The marked shot itself: the frame it was read from (ms from the start of the scan) and its faces */
  ref?: { t: number; faces: CropBox[] }
}

/**
 * Where the clip shows the shot marked at `markT` (ms from the start of the scan): [start, end]
 * stretches, and the people's box frame by frame. `cuts`: the camera cuts; `durationMs`: the scan.
 */
export function shotSpans(frames: FrameDetection[], cuts: number[], markT: number, durationMs: number): ShotResult {
  const fr: FrameDetection[] = [...frames]
    .sort((a, b) => a.frame_index - b.frame_index)
    .map(f => ({ ...f, faces: f.faces.filter(face => (face.score ?? 0) > 0) }))
  const timeOf = (fi: number) => fr[fi].frame_index * FRAME_INTERVAL_MS + FRAME_SHOWS_MS
  const bounds = [0, ...cuts.filter(c => c > 0 && c < durationMs), durationMs]
  const stretchOf = (t: number) => { let k = 0; while (k < bounds.length - 2 && t >= bounds[k + 1]) k++; return k }
  // The marked shot: the faces at the mark — the face count seen most in the second around it (one
  // frame can miss a face), in the frame nearest the mark with that count
  const home = stretchOf(markT)
  const around = fr.map((f, fi) => ({ f, t: timeOf(fi) })).filter(x => Math.abs(x.t - markT) <= 1000 && stretchOf(x.t) === home)
  const counts = new Map<number, number>()
  for (const x of around) counts.set(x.f.faces.length, (counts.get(x.f.faces.length) ?? 0) + 1)
  const want = [...counts].sort((a, b) => b[1] - a[1] || b[0] - a[0])[0]?.[0] ?? 0
  const ref = around.filter(x => x.f.faces.length === want).sort((a, b) => Math.abs(a.t - markT) - Math.abs(b.t - markT))[0]
  if (!ref || !want) return { spans: [], path: [], error: 'No faces in the picture here. Mark a shot where the people’s faces can be seen.' }
  const refFaces = ref.f.faces
  const smallest = Math.min(...refFaces.map(f => f.w))
  const mid = (f: CropBox) => [f.x + f.w / 2, f.y + f.h / 2]
  const near = (face: CropBox, r: CropBox) => {
    const [x, y] = mid(face), [rx, ry] = mid(r)
    const ratio = Math.max(face.w / r.w, r.w / face.w)
    return Math.abs(x - rx) <= SHOT_REACH_X && Math.abs(y - ry) <= SHOT_REACH_Y && ratio <= SHOT_SIZE_RATIO
      ? Math.abs(x - rx) + Math.abs(y - ry) : null
  }
  // Whose faces they are: each marked face's signature, averaged over the second around the mark
  const refSigs = refFaces.map(r => {
    const sigs = around.filter(x => x.f.faces.length === want)
      .flatMap(x => x.f.faces.filter(face => near(face, r) !== null && face.sig?.length).slice(0, 1).map(face => face.sig!))
    return sigs.length ? meanOf(sigs) : r.sig?.length ? r.sig : null
  })

  // Each frame: the marked shot (every face where it was, nobody else), part of it (some faces not
  // read, nobody else), another shot, or no faces read
  type State = 'match' | 'part' | 'other' | 'none'
  const states = fr.map((f): { s: State; box?: CropBox } => {
    const faces = f.faces.filter(face => face.w >= SHOT_BACKGROUND * smallest)
    if (!faces.length) return { s: 'none' }
    const used = new Set<number>()
    let found = 0
    const apart: number[] = []
    refFaces.forEach((r, ri) => {
      let best = -1, bestD = Infinity
      faces.forEach((face, k) => {
        if (used.has(k)) return
        const d = near(face, r)
        if (d !== null && d < bestD) { bestD = d; best = k }
      })
      if (best < 0) return
      used.add(best); found++
      const sig = faces[best].sig, refSig = refSigs[ri]
      if (sig?.length && refSig) apart.push(l1(sig, refSig))
    })
    if (used.size < faces.length) return { s: 'other' }     // someone who isn't in the marked shot
    // The same places, other people (another interview shot the same way)
    if (apart.length && apart.reduce((a, b) => a + b, 0) / apart.length >= SHOT_SAME_PEOPLE) return { s: 'other' }
    if (found < refFaces.length) return { s: 'part' }
    const xs = faces.flatMap(face => [face.x, face.x + face.w]), ys = faces.flatMap(face => [face.y, face.y + face.h])
    const x = Math.min(...xs), y = Math.min(...ys)
    return { s: 'match', box: { x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y } }
  })

  // The stretches. A camera shot (cut to cut) is one shot: when most of its frames are the marked
  // shot, all of it is — the face reader misses or misreads a face now and then, mostly as a shot
  // starts or ends (it used to leave a second of the shot unmarked there). Only something else for
  // SHOT_CHANGE_MS or longer (someone walks in, the camera turns to others) splits a camera shot.
  const spans: Turn[] = []
  const half = FRAME_INTERVAL_MS / 2
  for (let k = 0; k < bounds.length - 1; k++) {
    const lo = bounds[k], hi = bounds[k + 1]
    const idx = fr.map((_, fi) => fi).filter(fi => stretchOf(timeOf(fi)) === k)
    let from = lo
    let piece: number[] = []
    const flush = (to: number) => {
      const match = piece.filter(fi => states[fi].s === 'match').length
      const other = piece.filter(fi => states[fi].s === 'other').length
      if (match && match >= other && to - from >= SHOT_MIN_MS) spans.push([Math.round(from), Math.round(to)])
      piece = []
    }
    let i = 0
    while (i < idx.length) {
      if (states[idx[i]].s !== 'other') { piece.push(idx[i++]); continue }
      let j = i
      while (j < idx.length && states[idx[j]].s === 'other') j++
      if (timeOf(idx[j - 1]) - timeOf(idx[i]) + FRAME_INTERVAL_MS >= SHOT_CHANGE_MS) {
        // Another shot for a while: what came before ends here, what comes after starts after it
        flush(Math.max(from, timeOf(idx[i]) - half))
        from = Math.min(hi, timeOf(idx[j - 1]) + half)
      } else {
        piece.push(...idx.slice(i, j))
      }
      i = j
    }
    flush(hi)
  }
  // The people's box, frame by frame, where the shot is
  const path: PathPoint[] = []
  fr.forEach((_, fi) => {
    const t = timeOf(fi), st = states[fi]
    if (st.s !== 'match' || !spans.some(([x, y]) => t >= x && t <= y)) return
    const bx = st.box!
    path.push([t, round4(bx.x), round4(bx.y), round4(bx.w), round4(bx.h), 0])
  })
  // Joined where two stretches meet (the same shot either side of a cut that isn't one), or are a
  // moment apart inside one camera shot
  const joined: Turn[] = []
  for (const s of spans) {
    const last = joined[joined.length - 1]
    if (last && (s[0] - last[1] <= 1 || (s[0] - last[1] <= SHOT_JOIN_MS && stretchOf(last[1] - 1) === stretchOf(s[0])))) last[1] = s[1]
    else joined.push([s[0], s[1]])
  }
  return { spans: joined, path, ref: { t: ref.t, faces: refFaces } }
}

// ── Camera angles ──────────────────────────────────────────────────────────────
// The editor's Subjects tool lists every camera angle of the clip, so the user can go through them
// one by one and give each its subject or its group — none is missed. An angle is a camera shot and
// everywhere it comes back: found with shotSpans, so it is exactly what marking that shot as a group finds.

/** At most this many angles are listed; the least shown beyond that go with the other shots */
const MAX_ANGLES = 12
/**
 * Someone leaning, shifting or moving a little is not another angle. Two shots with the same number
 * of people, each in about the same place and size, are listed as one: a face this close (fractions
 * of the frame; wider than a shot's own match, which gives group formats their exact places) and
 * within this size ratio, and the same people by their signatures. Another camera (people in other
 * places) or other people in the same places still list apart.
 */
const ANGLE_MOTION_X = 0.15
const ANGLE_MOTION_Y = 0.15
const ANGLE_MOTION_SIZE = 2.2
/** A frame this close to a camera cut may already show the next shot: a short stretch is judged without it when it has others */
const SHORT_EDGE_MS = 150

export interface Angle {
  /** How many people it shows; 0: the other shots (no faces, or too short to be an angle) */
  people: number
  /** A moment that shows it (ms from the start of the scan), and the faces there, left to right */
  at: number
  faces: Array<{ x: number; y: number; w: number; h: number }>
  /** Wherever the clip shows it: [start, end] ms from the start of the scan */
  spans: Turn[]
}

/**
 * The clip's camera angles, in the order they first come up. Every moment of the scan is in exactly
 * one: the stretches between camera cuts, put together where they show the same shot (the same
 * people in the same places); stretches with no faces, or too short, are the last one (people: 0).
 */
export function findAngles(frames: FrameDetection[], cuts: number[], durationMs: number): Angle[] {
  const fr: FrameDetection[] = [...frames]
    .sort((a, b) => a.frame_index - b.frame_index)
    .map(f => ({ ...f, faces: f.faces.filter(face => (face.score ?? 0) > 0) }))
  const timeOf = (fi: number) => fr[fi].frame_index * FRAME_INTERVAL_MS + FRAME_SHOWS_MS
  const bounds = [0, ...cuts.filter(c => c > 0 && c < durationMs).sort((a, b) => a - b), durationMs]
  const stretches = bounds.slice(0, -1).map((a, k) => ({ a, b: bounds[k + 1], angle: -1 }))
  const overlap = (s: { a: number; b: number }, spans: Turn[]) => spans.reduce((n, [a, b]) => n + Math.max(0, Math.min(b, s.b) - Math.max(a, s.a)), 0)

  const found: Array<{ at: number; faces: CropBox[] }> = []
  stretches.forEach((st, k) => {
    if (st.angle >= 0 || st.b - st.a < SHOT_MIN_MS) return
    // The stretch's own picture: the face count seen most in it, at the middle frame showing that many
    const inside = fr.map((f, fi) => ({ n: f.faces.length, t: timeOf(fi) })).filter(x => x.t >= st.a && x.t < st.b)
    const counts = new Map<number, number>()
    for (const x of inside) counts.set(x.n, (counts.get(x.n) ?? 0) + 1)
    const want = [...counts].sort((p, q) => q[1] - p[1] || q[0] - p[0])[0]?.[0] ?? 0
    if (!want) return
    const withAll = inside.filter(x => x.n === want)
    const shot = shotSpans(fr, cuts, withAll[Math.floor(withAll.length / 2)].t, durationMs)
    if (shot.error || !shot.ref) return
    const id = found.push({ at: shot.ref.t, faces: [...shot.ref.faces].sort((p, q) => p.x - q.x) }) - 1
    st.angle = id
    // Everywhere else the clip shows this shot (for most of the stretch, or a good while of it)
    stretches.forEach((other, j) => {
      if (j <= k || other.angle >= 0) return
      const seen = overlap(other, shot.spans)
      if (seen >= SHOT_MIN_MS && seen >= 0.4 * (other.b - other.a)) other.angle = id
    })
  })

  // Light motion isn't another angle: an angle that is only the same people moved a little goes into the earlier one
  const sameAngle = (p: { faces: CropBox[] }, q: { faces: CropBox[] }) => {
    if (p.faces.length !== q.faces.length) return false
    const apart: number[] = []
    for (let i = 0; i < p.faces.length; i++) {
      const f = p.faces[i], g = q.faces[i]
      const dx = Math.abs(f.x + f.w / 2 - (g.x + g.w / 2)), dy = Math.abs(f.y + f.h / 2 - (g.y + g.h / 2))
      if (dx > ANGLE_MOTION_X || dy > ANGLE_MOTION_Y || Math.max(f.w / g.w, g.w / f.w) > ANGLE_MOTION_SIZE) return false
      if (f.sig?.length && g.sig?.length) apart.push(l1(f.sig, g.sig))
    }
    return !apart.length || apart.reduce((a, b) => a + b, 0) / apart.length < SHOT_SAME_PEOPLE
  }
  const into = found.map((_, id) => id)
  found.forEach((a, id) => {
    for (let e = 0; e < id; e++) if (into[e] === e && sameAngle(found[e], a)) { into[id] = e; break }
  })
  stretches.forEach(st => { if (st.angle >= 0) st.angle = into[st.angle] })

  // A stretch too short to be an angle of its own (or whose shot couldn't be told) joins the angle its
  // picture shows — the marked faces where they are, about their size, the same people — instead of
  // the other shots: most of its frames must show that angle. Frames right on a cut are left out
  // when there are others (the picture may already be the next shot's).
  const fits = (faces: CropBox[], ref: CropBox[]) => {
    if (!ref.length || faces.length < ref.length) return false
    const used = new Set<number>()
    const apart: number[] = []
    for (const r of ref) {
      let best = -1, bestD = Infinity
      faces.forEach((f, k) => {
        if (used.has(k)) return
        const dx = Math.abs(f.x + f.w / 2 - (r.x + r.w / 2)), dy = Math.abs(f.y + f.h / 2 - (r.y + r.h / 2))
        if (dx > SHOT_REACH_X || dy > SHOT_REACH_Y || Math.max(f.w / r.w, r.w / f.w) > SHOT_SIZE_RATIO) return
        if (dx + dy < bestD) { bestD = dx + dy; best = k }
      })
      if (best < 0) return false
      used.add(best)
      if (faces[best].sig?.length && r.sig?.length) apart.push(l1(faces[best].sig!, r.sig!))
    }
    return !apart.length || apart.reduce((x, y) => x + y, 0) / apart.length < SHOT_SAME_PEOPLE
  }
  for (const st of stretches) {
    if (st.angle >= 0) continue
    const inside = fr.map((f, fi) => ({ faces: f.faces, t: timeOf(fi) })).filter(x => x.t >= st.a && x.t < st.b)
    // (the clip's own start and end aren't cuts: a frame there shows this stretch)
    const away = inside.filter(x => (st.a <= 0 || x.t - st.a >= SHORT_EDGE_MS) && (st.b >= durationMs || st.b - x.t >= SHORT_EDGE_MS))
    const use = away.length ? away : inside
    const votes = new Map<number, number>()
    // Only the angles left after the light-motion join above
    for (const x of use) found.forEach((a, id) => { if (into[id] === id && fits(x.faces, a.faces)) votes.set(id, (votes.get(id) ?? 0) + 1) })
    const best = [...votes].sort((p, q) => q[1] - p[1])[0]
    if (best && best[1] * 2 >= use.length) st.angle = best[0]
  }

  // Past the most there can be, the least shown go with the other shots
  const total = (id: number) => stretches.filter(s => s.angle === id).reduce((n, s) => n + s.b - s.a, 0)
  const kept = new Set(found.map((_, id) => id).filter(id => into[id] === id).sort((p, q) => total(q) - total(p)).slice(0, MAX_ANGLES))
  const spansOf = (pick: (s: { angle: number }) => boolean) => {
    const out: Turn[] = []
    for (const s of stretches) {
      if (!pick(s)) continue
      const last = out[out.length - 1]
      if (last && s.a - last[1] <= 1) last[1] = Math.round(s.b)
      else out.push([Math.round(s.a), Math.round(s.b)])
    }
    return out
  }
  const angles: Angle[] = found.flatMap((a, id) => (kept.has(id) ? [{
    people: a.faces.length, at: Math.round(a.at),
    faces: a.faces.map(f => ({ x: round4(f.x), y: round4(f.y), w: round4(f.w), h: round4(f.h) })),
    spans: spansOf(s => s.angle === id),
  }] : []))
  const rest = spansOf(s => !kept.has(s.angle))
  if (rest.length) angles.push({ people: 0, at: Math.round((rest[0][0] + rest[0][1]) / 2), faces: [], spans: rest })
  return angles
}

/**
 * Where FFmpeg reads the clip's video from: a temporary link to the editing copy (at most 720p,
 * its frames keep the original's times), else the original — read through range requests, so only
 * the clip's own seconds are fetched, never the whole video (a long video's original can be
 * gigabytes). If the link can't be read, the file is downloaded as before.
 */
async function clipVideo(clip: { proxy_path: string | null; storage_path: string | null }, tmp: string, signal?: AbortSignal): Promise<string> {
  for (const key of [clip.proxy_path, clip.storage_path]) {
    if (!key) continue
    try {
      const url = await r2SignedUrl(key)
      // Can FFmpeg read it? (half a minute at most: a link that hangs falls back to the download)
      await promisify(execFile)('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width', '-of', 'csv=p=0', url], { timeout: 30_000, signal })
      return url
    } catch { /* the next one */ }
  }
  const videoPath = join(tmp, 'source.mp4')
  try {
    if (!clip.proxy_path) throw new Error('no editing copy')
    await r2DownloadToFile(clip.proxy_path, videoPath, signal)
  } catch {
    if (!clip.storage_path) throw new Error('The clip has no video')
    await r2DownloadToFile(clip.storage_path, videoPath, signal)
  }
  return videoPath
}

/** The video's transcript words in [from, to) of the source, ms from `from` (empty when it has none) */
async function spokenWords(videoId: string, from: number, to: number): Promise<SpokenWord[]> {
  try {
    const [row] = await db<{ id: string }[]>`SELECT id FROM transcripts WHERE video_id = ${videoId} ORDER BY created_at DESC LIMIT 1`
    if (!row) return []
    const words = await db<{ start_ms: number; end_ms: number; speaker_id: string | null }[]>`
      SELECT start_ms, end_ms, speaker_id FROM transcript_words
      WHERE transcript_id = ${row.id} AND end_ms > ${from} AND start_ms < ${to} ORDER BY start_ms
    `
    return words.map(w => ({ start_ms: w.start_ms - from, end_ms: w.end_ms - from, speaker_id: w.speaker_id ?? null }))
  } catch (e) {
    console.warn('[track_subjects] words could not be read:', e instanceof Error ? e.message : e)
    return []
  }
}

/** A group subject's result: where its shot is (source ms), or why it couldn't be found */
async function saveShot(id: string, r: ShotResult, from: number, to: number, sourceCuts: number[]) {
  if (r.error) {
    await db`UPDATE clip_subjects SET status = 'failed', error = ${r.error}, progress = 100, path = NULL, spans = NULL WHERE id = ${id}`
    return
  }
  const path = r.path.map(([t, x, y, w, h]) => [Math.round(t + from), x, y, w, h, 0])
  const spans = r.spans.map(([a, b]) => [Math.round(a + from), Math.round(b + from)])
  await db`
    UPDATE clip_subjects SET status = 'done', error = NULL, progress = 100,
      path = ${db.json(path as never)}, spans = ${db.json(spans as never)}, cuts = ${db.json(sourceCuts as never)},
      speaking = NULL, scan_start_ms = ${from}, scan_end_ms = ${to}
    WHERE id = ${id}
  `
}

async function setSubjects(ids: string[], patch: { status?: string; progress?: number; error?: string | null }) {
  if (!ids.length) return
  await db`
    UPDATE clip_subjects SET
      status = COALESCE(${patch.status ?? null}, status),
      progress = COALESCE(${patch.progress ?? null}, progress),
      error = ${patch.error === undefined ? db`error` : patch.error}
    WHERE id = ANY(${ids})
  `
}

export async function handleTrackSubjectsJob(job: Job, signal?: AbortSignal) {
  const { clip_id, start_ms, end_ms, auto, angles } = job.payload as unknown as TrackSubjectsJobPayload
  if (auto) return subEditClip(job.id, clip_id, signal)
  if (angles) return scanAngles(clip_id, start_ms, end_ms, signal)
  // The subjects waiting when this job starts (ones marked later get a job of their own)
  const subjects = await db<{ id: string; mark_ms: number; mark_x: number; mark_y: number; shot: unknown }[]>`
    UPDATE clip_subjects SET status = 'running', progress = 5, error = NULL
    WHERE clip_id = ${clip_id} AND status = 'queued'
    RETURNING id, mark_ms, mark_x, mark_y, shot
  `
  if (!subjects.length) return
  const ids = subjects.map(s => s.id)
  const t0 = Date.now()
  const tmp = await mkdtemp(join(tmpdir(), 'subjects-'))
  try {
    const [clip] = await db<{ storage_path: string | null; proxy_path: string | null; video_id: string }[]>`
      SELECT v.storage_path, v.proxy_path, v.id AS video_id FROM clips c JOIN videos v ON v.id = c.video_id WHERE c.id = ${clip_id}
    `
    if (!clip?.storage_path) throw new Error('The video of this clip could not be found')
    // Every mark inside the scan, whatever range the editor sent
    const from = Math.max(0, Math.min(start_ms, ...subjects.map(s => s.mark_ms)))
    const to = Math.max(end_ms, ...subjects.map(s => s.mark_ms + 1000))

    const videoPath = await clipVideo(clip, tmp, signal)
    await setSubjects(ids, { progress: 25 })

    // Faces with their lips (who is talking), the camera cuts, and the words said in this stretch
    const [info, cuts, speech] = await Promise.all([
      detectFaces(tmp, videoPath, from, to, signal, true),
      detectCuts(videoPath, from, to, signal),
      spokenWords(clip.video_id, from, to),
    ])
    signal?.throwIfAborted()
    await setSubjects(ids, { progress: 90 })

    // People are followed; a group subject (a marked shot) is found wherever the clip shows that shot
    const people = subjects.filter(s => !s.shot), shots = subjects.filter(s => s.shot)
    const results = people.length
      ? followSubjects(info.frames, cuts, people.map(s => ({ id: s.id, t: s.mark_ms - from, x: s.mark_x, y: s.mark_y })), speech, to - from)
      : []
    // The camera cuts, in the source video's time (exact to the frame: the editor switches on them)
    const sourceCuts = cuts.map(c => Math.round(c + from))
    for (const s of shots) await saveShot(s.id, shotSpans(info.frames, cuts, s.mark_ms - from, to - from), from, to, sourceCuts)
    for (const r of results) {
      if (r.error) {
        await db`UPDATE clip_subjects SET status = 'failed', error = ${r.error}, progress = 100, path = NULL WHERE id = ${r.id}`
        continue
      }
      // Times in the source video
      const path = r.path.map(([t, x, y, w, h, talks]) => [Math.round(t + from), x, y, w, h, talks])
      await db`
        UPDATE clip_subjects SET status = 'done', error = NULL, progress = 100,
          path = ${db.json(path as never)}, cuts = ${db.json(sourceCuts as never)},
          speaking = ${db.json(r.speaking.map(([a, b]) => [Math.round(a + from), Math.round(b + from)]) as never)},
          scan_start_ms = ${from}, scan_end_ms = ${to}
        WHERE id = ${r.id}
      `
    }
    const seen = results.map(r => `${r.error ? 'failed' : `${r.path.length} frames, ${r.speaking.length} turn(s)`}`).join(', ')
    console.log(`[track_subjects] clip ${clip_id}: ${info.frames.length} frames, ${cuts.length} cut(s), subjects: ${seen || 'none'}, shots: ${shots.length} (${((Date.now() - t0) / 1000).toFixed(0)}s)`)
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    await setSubjects(ids, { status: 'failed', error: `Tracking failed: ${msg}`, progress: 100 }).catch(() => {})
    throw err
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
}

/**
 * List the clip's camera angles (its clip_angles row, queued by the editor's Subjects tool): the
 * faces 4× a second and the camera cuts through [start_ms, end_ms] of the source video, put
 * together into angles (findAngles). Times are saved in the source video's time.
 */
async function scanAngles(clipId: string, from: number, to: number, signal?: AbortSignal) {
  const [row] = await db<{ clip_id: string }[]>`
    UPDATE clip_angles SET status = 'running', progress = 5, error = NULL, updated_at = now()
    WHERE clip_id = ${clipId} AND status = 'queued' RETURNING clip_id
  `
  if (!row) return
  const t0 = Date.now()
  const tmp = await mkdtemp(join(tmpdir(), 'angles-'))
  try {
    const [clip] = await db<{ storage_path: string | null; proxy_path: string | null }[]>`
      SELECT v.storage_path, v.proxy_path FROM clips c JOIN videos v ON v.id = c.video_id WHERE c.id = ${clipId}
    `
    if (!clip?.storage_path) throw new Error('The video of this clip could not be found')
    const videoPath = await clipVideo(clip, tmp, signal)
    await db`UPDATE clip_angles SET progress = 25, updated_at = now() WHERE clip_id = ${clipId}`
    const [info, cuts] = await Promise.all([
      // Every face in the clip, not the ones a crop would follow: the list is who is in it
      detectFaces(tmp, videoPath, from, to, signal, false, true),
      detectCuts(videoPath, from, to, signal),
    ])
    signal?.throwIfAborted()
    const angles = findAngles(info.frames, cuts, to - from).map(a => ({
      ...a, at: a.at + from, spans: a.spans.map(([s, e]) => [s + from, e + from]),
    }))
    await db`
      UPDATE clip_angles SET status = 'done', progress = 100, error = NULL, angles = ${db.json(angles as never)},
        scan_start_ms = ${from}, scan_end_ms = ${to}, updated_at = now()
      WHERE clip_id = ${clipId}
    `
    console.log(`[track_subjects] clip ${clipId}: ${angles.length} angle(s) from ${info.frames.length} frames, ${cuts.length} cut(s) (${((Date.now() - t0) / 1000).toFixed(0)}s)`)
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    await db`UPDATE clip_angles SET status = 'failed', progress = 100, error = ${msg}, updated_at = now() WHERE clip_id = ${clipId}`.catch(() => {})
    throw err
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
}

// ── Sub-edit: the clip board's one-button framing ──────────────────────────────
// Everything the Subjects tool does, without the user marking anyone: the worker finds the people
// in the clip, marks each (a frame where their face is clearest), tracks them with who is talking,
// and frames the clip on them — every section's view switches to whoever is talking — written into
// the clip itself, so the clip board's Download exports it and the editor opens with it.

const SUBJECT_COLORS = ['#f472b6', '#38bdf8', '#facc15', '#a78bfa', '#fb923c', '#34d399']
/** At most this many people are marked (the editor allows 6) */
const MAX_PEOPLE = 6
/** Everyone seen at least this long is registered (4 frames a second: 1 s), so a passing face isn't */
const MIN_PERSON_FRAMES = 4
/** Frame slots that show the main video whole (not cropped): their view isn't moved (frames.ts "contain") */
const CONTAIN_SLOTS: Record<string, number[]> = { frame_blur_fill: [0], frame_screenshot: [0], frame_screen_face: [0] }

/**
 * The people in the clip, as marks: the face tracks, grouped into people by the colours of face and
 * clothes (a track clearly matching a person already found is that person again), the ones seen long
 * enough, most seen first. Each mark is the frame where that person's face is biggest.
 */
export function findPeople(frames: FrameDetection[], cuts: number[]): Array<{ t: number; x: number; y: number }> {
  const fr: FrameDetection[] = [...frames]
    .sort((a, b) => a.frame_index - b.frame_index)
    .map(f => ({ ...f, faces: f.faces.filter(face => (face.score ?? 0) > 0) }))
  if (!fr.length) return []
  const ids = trackFaces(fr, cuts.map(c => c - FRAME_SHOWS_MS))
  const tracks = new Map<number, Array<{ fi: number; face: CropBox }>>()
  fr.forEach((f, fi) => ids[fi].forEach((id, k) => {
    const list = tracks.get(id) ?? []
    list.push({ fi, face: f.faces[k] })
    tracks.set(id, list)
  }))
  const people: Array<{ sig: number[] | null; frames: number; best: { fi: number; face: CropBox } }> = []
  for (const list of [...tracks.values()].sort((a, b) => b.length - a.length)) {
    const sigs = list.map(x => x.face.sig).filter((v): v is number[] => !!v && v.length > 0)
    const sig = sigs.length ? meanOf(sigs) : null
    const same = sig ? people.find(p => p.sig && l1(sig, p.sig) < SAME_PERSON) : undefined
    const biggest = list.reduce((a, b) => (b.face.w * b.face.h > a.face.w * a.face.h ? b : a))
    if (same) {
      same.frames += list.length
      if (biggest.face.w * biggest.face.h > same.best.face.w * same.best.face.h) same.best = biggest
      continue
    }
    people.push({ sig, frames: list.length, best: biggest })
  }
  return people
    .filter(p => p.frames >= MIN_PERSON_FRAMES)
    .sort((a, b) => b.frames - a.frames)
    .slice(0, MAX_PEOPLE)
    .map(p => ({
      t: fr[p.best.fi].frame_index * FRAME_INTERVAL_MS + FRAME_SHOWS_MS,
      x: round4(p.best.face.x + p.best.face.w / 2),
      y: round4(p.best.face.y + p.best.face.h / 2),
    }))
}

/** What the clip board's Stop writes on the job (the frontend's stopSubEdit) */
export const STOPPED = 'Stopped by the user'

async function subEditClip(jobId: string, clipId: string, queueSignal?: AbortSignal) {
  const t0 = Date.now()
  // Stopped from the clip board: the job row says so. Checked every 2 s; everything running (the
  // download, FFmpeg, the face reading) is stopped through the signal
  const stop = new AbortController()
  const signal = queueSignal ? AbortSignal.any([queueSignal, stop.signal]) : stop.signal
  const watch = setInterval(async () => {
    try {
      const [row] = await db<{ status: string; error: string | null }[]>`SELECT status, error FROM jobs WHERE id = ${jobId}`
      if (row && row.status === 'failed' && row.error === STOPPED) stop.abort(new Error(STOPPED))
    } catch { /* the next check */ }
  }, 2000)
  let created: string[] = []
  try {
    await subEditSteps(clipId, signal, ids => { created = ids }, t0)
  } catch (e) {
    // Stopped: nothing half-done stays (the people it had just registered go; the clip's framing was never touched)
    if (stop.signal.aborted) {
      if (created.length) await db`DELETE FROM clip_subjects WHERE id = ANY(${created})`.catch(() => {})
      console.log(`[track_subjects] sub-edit clip ${clipId}: stopped by the user (${((Date.now() - t0) / 1000).toFixed(0)}s)`)
      throw new Error(STOPPED)
    }
    throw e
  } finally {
    clearInterval(watch)
  }
}

async function subEditSteps(clipId: string, signal: AbortSignal, onCreated: (ids: string[]) => void, t0: number) {
  const [clip] = await db<{ start_ms: number; end_ms: number; storage_path: string | null; proxy_path: string | null; video_id: string }[]>`
    SELECT c.start_ms, c.end_ms, v.storage_path, v.proxy_path, v.id AS video_id FROM clips c JOIN videos v ON v.id = c.video_id WHERE c.id = ${clipId}
  `
  if (!clip?.storage_path) throw new Error('The video of this clip could not be found')
  const from = Number(clip.start_ms), to = Number(clip.end_ms)
  const tmp = await mkdtemp(join(tmpdir(), 'subedit-'))
  try {
    const videoPath = await clipVideo(clip, tmp, signal)
    const [info, cuts, speech, size] = await Promise.all([
      detectFaces(tmp, videoPath, from, to, signal, true),
      detectCuts(videoPath, from, to, signal),
      spokenWords(clip.video_id, from, to),
      probeSize(videoPath),
    ])
    signal?.throwIfAborted()
    const found = findPeople(info.frames, cuts)
    if (!found.length) throw new Error('No people could be found in this clip')

    // They become the clip's subjects (the Subjects tool shows them), replacing the people marked
    // before. Group subjects (marked shots) stay — the user's own choice — found again in this clip
    const groups = await db<{ id: string; mark_ms: number }[]>`SELECT id, mark_ms FROM clip_subjects WHERE clip_id = ${clipId} AND shot IS NOT NULL`
    for (const g of groups) await saveShot(g.id, shotSpans(info.frames, cuts, Number(g.mark_ms) - from, to - from), from, to, cuts.map(c => Math.round(c + from)))
    const marks = await db.begin(async tx => {
      await tx`DELETE FROM clip_subjects WHERE clip_id = ${clipId} AND shot IS NULL`
      const rows = await tx<{ id: string }[]>`
        INSERT INTO clip_subjects ${tx(found.map((m, i) => ({
          clip_id: clipId, label: String.fromCharCode(65 + i), color: SUBJECT_COLORS[i % SUBJECT_COLORS.length],
          mark_ms: Math.round(m.t + from), mark_x: m.x, mark_y: m.y, status: 'running', progress: 60,
        })))}
        RETURNING id
      `
      return rows.map((r, i) => ({ id: r.id, t: found[i].t, x: found[i].x, y: found[i].y }))
    })
    onCreated(marks.map(m => m.id))
    signal.throwIfAborted()
    const results = followSubjects(info.frames, cuts, marks, speech, to - from)
    const sourceCuts = cuts.map(c => Math.round(c + from))
    const tracked: Array<{ id: string; path: SubjectPoint[]; speaking: Array<[number, number]> }> = []
    for (const r of results) {
      if (r.error) {
        await db`UPDATE clip_subjects SET status = 'failed', error = ${r.error}, progress = 100, path = NULL WHERE id = ${r.id}`
        continue
      }
      const path = r.path.map(([t, x, y, w, h, talks]) => [Math.round(t + from), x, y, w, h, talks] as SubjectPoint)
      const speaking = r.speaking.map(([a, b]) => [Math.round(a + from), Math.round(b + from)] as [number, number])
      tracked.push({ id: r.id, path, speaking })
      await db`
        UPDATE clip_subjects SET status = 'done', error = NULL, progress = 100,
          path = ${db.json(path as never)}, cuts = ${db.json(sourceCuts as never)}, speaking = ${db.json(speaking as never)},
          scan_start_ms = ${from}, scan_end_ms = ${to}
        WHERE id = ${r.id}
      `
    }
    if (!tracked.length) throw new Error('No one could be followed through this clip')
    // Last chance to stop before the clip itself changes
    signal.throwIfAborted()
    const framed = await frameClip(clipId, from, to, tracked, sourceCuts, size)
    console.log(`[track_subjects] sub-edit clip ${clipId}: ${found.length} people, ${framed} view(s) framed (${((Date.now() - t0) / 1000).toFixed(0)}s)`)
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
}

/**
 * Frame the clip on the tracked people (the editor's frameClipOnSubjects): in every section, a view
 * showing the main video switches to whoever is talking; a section with two or three such views
 * (split, trio, a frame) puts one person in each. Views keep their size. Locked and Horizontal
 * sections, photos, other videos and slots showing the video whole are left alone. A clip never
 * opened (no sections yet) gets one Vertical section. The clip's old export no longer matches.
 */
async function frameClip(
  clipId: string, clipStart: number, clipEnd: number, people: Array<{ id: string; path: SubjectPoint[]; speaking: Array<[number, number]> }>,
  cuts: number[], size: { width: number; height: number } | null,
): Promise<number> {
  // Sections and keyframes are kept in clip time (ms from the clip's start)
  const toSource = (t: number) => clipStart + t
  const toTimeline = (s: number) => s - clipStart
  const length = clipEnd - clipStart
  // A 9:16 view at full height, as a share of the source's width
  const reelW = size ? Math.min(1, (9 / 16) * (size.height / size.width)) : 81 / 256
  return db.begin(async tx => {
    const segs = await tx<{ id: string; start_ms: number; end_ms: number; layout: string; frame: { main_slots?: number[] } | null; locked?: boolean | null }[]>`
      SELECT * FROM segments WHERE clip_id = ${clipId} ORDER BY start_ms
    `
    if (!segs.length) {
      // Never opened: one Vertical section over the whole clip (as the editor would open it)
      const changes = autoDirectChanges(people, 0, length, toSource, { w: reelW, h: 1 }, cuts, toTimeline)
      if (!changes) return 0
      const [seg] = await tx`INSERT INTO segments (clip_id, start_ms, end_ms, layout, sort_order) VALUES (${clipId}, 0, ${length}, 'vertical', 0) RETURNING id`
      const [box] = await tx`INSERT INTO crop_boxes (segment_id, slot_index, source_video_id, source_offset_ms, muted) VALUES (${seg.id}, 0, NULL, 0, false) RETURNING id`
      await insertKeyframes(tx, box.id as string, changes, 0)
      await tx`UPDATE clips SET status = 'draft', output_url = NULL WHERE id = ${clipId}`
      return 1
    }
    const boxes = await tx<{ id: string; segment_id: string; slot_index: number; source_video_id: string | null; image_path: string | null }[]>`
      SELECT id, segment_id, slot_index, source_video_id, image_path FROM crop_boxes WHERE segment_id = ANY(${segs.map(s => s.id)}) ORDER BY slot_index
    `
    const kfs = await tx<{ box_id: string; t_ms: number; w: number; h: number }[]>`
      SELECT box_id, t_ms, w, h FROM box_keyframes WHERE box_id = ANY(${boxes.map(b => b.id)}) ORDER BY t_ms
    `
    // A layout should change where the camera cuts: a section boundary a little before or after a
    // cut would leave the old layout on the new picture, so it moves onto the cut (lib/snapSections.ts)
    const mainOnly = (segId: string) => boxes.filter(b => b.segment_id === segId).every(b => !b.image_path && !b.source_video_id)
    const snapped = snapSectionBoundaries(
      segs.map(s => ({
        id: s.id, start_ms: Number(s.start_ms), end_ms: Number(s.end_ms),
        movable: !s.locked && s.layout !== 'horizontal' && !s.layout.startsWith('frame_') && mainOnly(s.id),
      })),
      cuts.map(c => c - clipStart),
    )
    for (const m of snapped) {
      await tx`UPDATE segments SET start_ms = ${m.start_ms}, end_ms = ${m.end_ms} WHERE id = ${m.id}`
      const seg = segs.find(x => x.id === m.id)!
      seg.start_ms = m.start_ms
      seg.end_ms = m.end_ms
    }
    if (snapped.length) console.log(`[track_subjects] clip ${clipId}: ${snapped.length} section(s) moved onto the camera cuts`)
    let framed = 0
    for (const seg of segs) {
      const start = Number(seg.start_ms), end = Number(seg.end_ms)
      if (seg.locked || seg.layout === 'horizontal' || end - start <= 250) continue
      const frame = seg.layout.startsWith('frame_')
      const mainSlots = frame ? seg.frame?.main_slots ?? [0] : null
      const views = boxes.filter(b => b.segment_id === seg.id && !b.image_path && !b.source_video_id
        && (!mainSlots || (mainSlots.includes(b.slot_index) && !(CONTAIN_SLOTS[seg.layout] ?? []).includes(b.slot_index))))
      for (const [i, box] of views.entries()) {
        // The view keeps its size (its first keyframe); never framed: a full-height 9:16 in a Vertical section
        const first = kfs.find(k => k.box_id === box.id)
        const view = first ? { w: Number(first.w), h: Number(first.h) } : seg.layout === 'vertical' ? { w: reelW, h: 1 } : null
        if (!view) continue
        const changes = views.length === 1
          ? autoDirectChanges(people, start, end, toSource, view, cuts, toTimeline)
          : followChanges(people[i % people.length].path, start, end, toSource, view, 'smooth', cuts)
        if (!changes?.length) continue
        await tx`DELETE FROM box_keyframes WHERE box_id = ${box.id}`
        await insertKeyframes(tx, box.id, changes, start)
        framed++
      }
    }
    if (framed || snapped.length) await tx`UPDATE clips SET status = 'draft', output_url = NULL WHERE id = ${clipId}`
    return framed
  })
}

async function insertKeyframes(tx: TransactionSql, boxId: string, changes: ReturnType<typeof followChanges> & object, start: number) {
  const kfs = buildKeyframes(changes.map((c, k) => (k === 0 ? { ...c, t_ms: start } : c)))
    .map(k => ({ box_id: boxId, t_ms: Math.round(k.t_ms), x: round4(k.x), y: round4(k.y), w: round4(k.w), h: round4(k.h) }))
  for (let i = 0; i < kfs.length; i += 5000) await tx`INSERT INTO box_keyframes ${tx(kfs.slice(i, i + 5000))}`
}
