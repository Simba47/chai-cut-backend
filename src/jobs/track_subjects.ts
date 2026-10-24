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

// ── Who is who: the faces' identities (SFace, the detector's `emb`) ─────────────
// The colours of face and clothes can't tell people apart across camera angles: the same bearded man
// in two angles read 0.40 apart and him against a woman in white 0.42 (2026-10-24, a 7-person
// interview) — so the woman's other close-ups listed as other people and the man's track took 54% of
// the clip. Faces' identities, averaged over a track: the same person 0.45 alike or more, different
// people 0.31 at most. Faces too small to read (a wide shot) have none: there the colours still decide.
/** Two tracks' identities this alike (dot product of the means) or more: the same person */
const ID_SAME = 0.45
/** …under this: surely someone else (in between: not known, the colours decide) */
const ID_OTHER = 0.3
/** A track's identity is known from this many faces read */
const ID_MIN_FACES = 3

const dot = (a: number[], b: number[]) => a.reduce((t, v, i) => t + v * (b[i] ?? 0), 0)
const unit = (v: number[]) => { const n = Math.hypot(...v); return n > 0 ? v.map(x => x / n) : v }

/**
 * The people in a scan: face tracks (trackFaces' ids, per frame) put together by identity — average
 * likeness between groups of tracks, the most alike first, while at least ID_SAME. Two tracks seen in
 * the same frame are two people, always. Returns each track's person (tracks with no identity have none)
 * and each person's identity.
 */
export function identities(frames: FrameDetection[], ids: number[][]): { personOf: Map<number, number>; idOf: Map<number, number[]> } {
  const embs = new Map<number, number[][]>(), seen = new Map<number, Set<number>>()
  frames.forEach((f, fi) => ids[fi].forEach((id, k) => {
    const e = f.faces[k]?.emb
    if (e?.length) embs.set(id, [...(embs.get(id) ?? []), e])
    seen.set(id, (seen.get(id) ?? new Set()).add(fi))
  }))
  const known = [...embs].filter(([, v]) => v.length >= ID_MIN_FACES).map(([id, v]) => ({ id, e: unit(meanOf(v)) }))
  const sim = known.map(a => known.map(b => dot(a.e, b.e)))
  const apart = known.map(a => known.map(b => a === b ? false : [...seen.get(a.id)!].some(fi => seen.get(b.id)!.has(fi))))
  let groups = known.map((_, i) => [i])
  for (;;) {
    let best = ID_SAME, bi = -1, bj = -1
    for (let i = 0; i < groups.length; i++) for (let j = i + 1; j < groups.length; j++) {
      if (groups[i].some(a => groups[j].some(b => apart[a][b]))) continue
      let s = 0
      for (const a of groups[i]) for (const b of groups[j]) s += sim[a][b]
      s /= groups[i].length * groups[j].length
      if (s >= best) { best = s; bi = i; bj = j }
    }
    if (bi < 0) break
    groups[bi] = [...groups[bi], ...groups[bj]]
    groups = groups.filter((_, k) => k !== bj)
  }
  // (most seen first: person 0 is the one on screen most)
  const size = (g: number[]) => g.reduce((t, i) => t + seen.get(known[i].id)!.size, 0)
  groups.sort((a, b) => size(b) - size(a))
  const personOf = new Map<number, number>(), idOf = new Map<number, number[]>()
  groups.forEach((g, p) => {
    for (const i of g) personOf.set(known[i].id, p)
    idOf.set(p, unit(meanOf(g.flatMap(i => embs.get(known[i].id)!))))
  })
  return { personOf, idOf }
}

/** Who each face of a scan is (identities): face → person; faces too small to read aren't in it */
export function whoIs(frames: FrameDetection[], cuts: number[]): Map<CropBox, number> {
  const fr = [...frames].sort((a, b) => a.frame_index - b.frame_index).map(f => ({ ...f, faces: f.faces.filter(face => (face.score ?? 0) > 0) }))
  const ids = trackFaces(fr, cuts.map(c => c - FRAME_SHOWS_MS))
  const { personOf } = identities(fr, ids)
  const who = new Map<CropBox, number>()
  fr.forEach((f, fi) => ids[fi].forEach((id, k) => { const p = personOf.get(id); if (p !== undefined) who.set(f.faces[k], p) }))
  return who
}

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
  // Who is who, across camera angles (identities)
  const { personOf } = identities(fr, ids)
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
    // (the same face, or the same person in another angle)
    const other = [...marked].find(([, t]) => t === id || (personOf.has(id) && personOf.get(t) === personOf.get(id)))
    if (other) { errors.set(mk.id, 'This is the same person as another subject.'); continue }
    marked.set(mk.id, id)
  }

  // Everywhere else the subject's person is seen (the tracks with their identity) is theirs
  const subjects = [...marked.keys()]
  const owned = new Map<string, Set<number>>(subjects.map(s => [s, new Set([marked.get(s)!])]))
  const personOfSubject = new Map(subjects.map(s => [s, personOf.get(marked.get(s)!)]))
  for (const s of subjects) {
    const p = personOfSubject.get(s)
    if (p !== undefined) for (const [id, q] of personOf) if (q === p) owned.get(s)!.add(id)
  }
  const taken = new Set([...owned.values()].flatMap(set => [...set]))
  const framesOf = (s: string) => {
    const set = new Set<number>()
    for (const id of owned.get(s)!) for (const fi of tracks.get(id)!.keys()) set.add(fi)
    return set
  }
  // The other tracks, given to the subject whose colours they match (clearly), nearest first. A track
  // whose identity is known is someone else's than a subject with another identity: never by colours
  const offers: Array<{ id: number; s: string; d: number }> = []
  for (const [id, sig] of sigOf) {
    if (taken.has(id)) continue
    const ds = subjects
      .filter(s => !personOf.has(id) || personOfSubject.get(s) === undefined)
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
/**
 * A face is where the marked one was: its middle at most this far from it (fractions of the frame),
 * or SHOT_REACH_FACE of the marked face's width when that is more. Someone sitting still still leans
 * and turns: a seated man in a close-up moved 7% of the frame's width inside one camera shot
 * (measured 2026-10-17), which a fixed 6% read as another shot — his group's format stopped 2 s
 * before the cut. A bigger face moves more.
 */
const SHOT_REACH_X = 0.06
const SHOT_REACH_Y = 0.08
const SHOT_REACH_FACE = 0.9
const SHOT_SIZE_RATIO = 1.45
/**
 * The same people: the faces' signatures (colours of face and clothes; L1, each sums to 1), on
 * average over the shot's faces, against the marked ones — judged once for a whole stretch of a
 * camera shot (its median), never frame by frame. Measured: the same people 0.13–0.28 a frame (never
 * over 0.45), other people 0.7–0.9 — but two different men both in black read 0.47–0.67 (2026-10-17),
 * and a frame-by-frame line at 0.5 flipped in the middle of that shot, so a group's format started
 * 2 s after the cut on the wrong person. A stretch is the marked people when its median is under
 * SHOT_SAME_PEOPLE; a frame at SHOT_OTHER_PEOPLE or more is surely someone else (counted like a
 * face that isn't there: enough of those in a row splits the camera shot).
 */
const SHOT_SAME_PEOPLE = 0.38
const SHOT_OTHER_PEOPLE = 0.65
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
  /** The stretches before they are joined, each with how far its people's colours are from the marked
   *  ones (its median; null: not known) — so two groups claiming one stretch can be settled (settleGroups) */
  parts?: Array<{ span: Turn; apart: number | null }>
  /** The marked shot itself: the frame it was read from (ms from the start of the scan) and its faces */
  ref?: { t: number; faces: CropBox[] }
}

/**
 * Where the clip shows the shot marked at `markT` (ms from the start of the scan): [start, end]
 * stretches, and the people's box frame by frame. `cuts`: the camera cuts; `durationMs`: the scan.
 * `who` (whoIs): who each face is — where both a face and the marked one are known, that decides
 * whether they are the same people (colours can't: a man in black and a woman in white in close-ups
 * from the same seat, 2026-10-24); the colours decide only for faces too small to read.
 */
export function shotSpans(frames: FrameDetection[], cuts: number[], markT: number, durationMs: number, who?: Map<CropBox, number>): ShotResult {
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
    const reachX = Math.max(SHOT_REACH_X, SHOT_REACH_FACE * r.w), reachY = Math.max(SHOT_REACH_Y, SHOT_REACH_FACE * r.w)
    return Math.abs(x - rx) <= reachX && Math.abs(y - ry) <= reachY && ratio <= SHOT_SIZE_RATIO
      ? Math.abs(x - rx) + Math.abs(y - ry) : null
  }
  // Whose faces they are: each marked face's signature, averaged over the whole marked camera shot
  // (not just the second around the mark: then it doesn't matter where in the shot it was marked, and
  // a moment of odd light at the mark doesn't make the people read as someone else elsewhere)
  const homeFrames = fr.map((f, fi) => ({ f, t: timeOf(fi) })).filter(x => stretchOf(x.t) === home && x.f.faces.length === want)
  const refSigs = refFaces.map(r => {
    const sigs = homeFrames
      .flatMap(x => x.f.faces.filter(face => near(face, r) !== null && face.sig?.length).slice(0, 1).map(face => face.sig!))
    return sigs.length ? meanOf(sigs) : r.sig?.length ? r.sig : null
  })
  // …and who they are: the person seen most in each marked face's place through the marked camera shot
  const refPersons = refFaces.map(r => {
    if (!who) return undefined
    const votes = new Map<number, number>()
    for (const x of homeFrames) {
      const face = x.f.faces.find(g => near(g, r) !== null && who.has(g))
      if (!face) continue
      const p = who.get(face)!
      votes.set(p, (votes.get(p) ?? 0) + 1)
    }
    return [...votes].sort((p, q) => q[1] - p[1])[0]?.[0]
  })

  // Each frame: the marked shot (every face where it was), part of it (some faces not read), another
  // shot, or no faces read. `apart`: how far its faces' colours are from the marked people's (their
  // average), judged later over a whole stretch. A face more than at the mark is someone who isn't in
  // the marked shot — unless a face was seen in that place in the marked camera shot itself (user,
  // 2026-10-20): in a panel of seven the face reader sees 4 to 7 of them as heads turn, and a frame
  // with one more face counted as another shot, so a group marked on 5 faces wasn't found where the
  // same shot came back. Someone walking in (a place nobody was in) still is another shot
  // 'most': all the faces but one, each where it was (a group of 3 or more) — on a sofa of three the
  // face reader often misses one turned head for a whole camera shot (user, 2026-10-24: 29 frames of
  // the same three people, colours 0.10 apart, never all three read — the group wasn't found there)
  type State = 'match' | 'most' | 'part' | 'other' | 'none'
  // Every face seen in the marked camera shot (any frame of it): the places of its people
  const seenPlaces = fr.filter((_, fi) => stretchOf(timeOf(fi)) === home).flatMap(f => f.faces.filter(face => face.w >= SHOT_BACKGROUND * smallest))
  const states = fr.map((f): { s: State; box?: CropBox; apart?: number } => {
    const faces = f.faces.filter(face => face.w >= SHOT_BACKGROUND * smallest)
    // Faces, but all far smaller than the marked ones: a wider shot, not a moment of no faces (user,
    // 2026-10-24: a round-table wide shot of four, cut into a close-up, counted as "no faces" and went
    // with the close-up's scene — one frame on the cut showing the close-up was enough)
    if (!faces.length) return { s: f.faces.length ? 'other' : 'none' }
    const used = new Set<number>()
    let found = 0
    const apart: number[] = []
    let someoneElse = false
    refFaces.forEach((r, ri) => {
      let best = -1, bestD = Infinity
      faces.forEach((face, k) => {
        if (used.has(k)) return
        const d = near(face, r)
        if (d !== null && d < bestD) { bestD = d; best = k }
      })
      if (best < 0) return
      used.add(best); found++
      const p = who?.get(faces[best]), q = refPersons[ri]
      if (p !== undefined && q !== undefined) {
        // Known: the same person (as close as colours ever read), or surely someone else
        if (p !== q) someoneElse = true
        else apart.push(0)
        return
      }
      const sig = faces[best].sig, refSig = refSigs[ri]
      if (sig?.length && refSig) apart.push(l1(sig, refSig))
    })
    // Someone else in a marked person's place (by who they are): another shot
    if (someoneElse) return { s: 'other' }
    const far = apart.length ? apart.reduce((a, b) => a + b, 0) / apart.length : undefined
    // The same places, surely other people (another interview shot the same way)
    if (far !== undefined && far >= SHOT_OTHER_PEOPLE) return { s: 'other', apart: far }
    // Someone who isn't in the marked shot: a face more, in a place nobody was in the marked camera shot
    if (faces.some((face, k) => !used.has(k) && !seenPlaces.some(p => near(face, p) !== null))) return { s: 'other' }
    if (found < refFaces.length) return { s: found >= 2 && found >= refFaces.length - 1 ? 'most' : 'part', apart: far }
    // (the marked people's box: a face more isn't in it)
    const mine = [...used].map(k => faces[k])
    const xs = mine.flatMap(face => [face.x, face.x + face.w]), ys = mine.flatMap(face => [face.y, face.y + face.h])
    const x = Math.min(...xs), y = Math.min(...ys)
    return { s: 'match', box: { x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y }, apart: far }
  })
  /** Whether a stretch shows the marked people: its frames' colour distances, judged together (their median) */
  const apartOf = (piece: number[]) => {
    const d = piece.map(fi => states[fi].apart).filter((v): v is number => v !== undefined).sort((a, b) => a - b)
    return d.length ? d[Math.floor(d.length / 2)] : null
  }

  // The stretches. A camera shot (cut to cut) is one shot: when most of its frames are the marked
  // shot, all of it is — the face reader misses or misreads a face now and then, mostly as a shot
  // starts or ends (it used to leave a second of the shot unmarked there). Only something else for
  // SHOT_CHANGE_MS or longer (someone walks in, the camera turns to others) splits a camera shot.
  const parts: Array<{ span: Turn; apart: number | null }> = []
  const half = FRAME_INTERVAL_MS / 2
  for (let k = 0; k < bounds.length - 1; k++) {
    const lo = bounds[k], hi = bounds[k + 1]
    const idx = fr.map((_, fi) => fi).filter(fi => stretchOf(timeOf(fi)) === k)
    let from = lo
    let piece: number[] = []
    const flush = (to: number) => {
      // A frame right on a camera cut may already show the next shot (a frame shows its time ±0.1 s):
      // it doesn't decide the stretch when there are others (the clip's own start and end aren't cuts)
      const core = piece.filter(fi => (lo <= 0 || timeOf(fi) - lo >= SHORT_EDGE_MS) && (hi >= durationMs || hi - timeOf(fi) >= SHORT_EDGE_MS))
      const judged = core.length ? core : piece
      const match = judged.filter(fi => states[fi].s === 'match' || states[fi].s === 'most').length
      const other = judged.filter(fi => states[fi].s === 'other').length
      const apart = apartOf(piece)
      if (match && match >= other && to - from >= SHOT_MIN_MS && (apart === null || apart < SHOT_SAME_PEOPLE)) {
        parts.push({ span: [Math.round(from), Math.round(to)], apart })
      }
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
  const spans = parts.map(x => x.span)
  // The people's box, frame by frame, where the shot is
  const path: PathPoint[] = []
  fr.forEach((_, fi) => {
    const t = timeOf(fi), st = states[fi]
    if (st.s !== 'match' || !spans.some(([x, y]) => t >= x && t <= y)) return
    const bx = st.box!
    path.push([t, round4(bx.x), round4(bx.y), round4(bx.w), round4(bx.h), 0])
  })
  return { spans: joinSpans(spans, cuts, durationMs), path, parts, ref: { t: ref.t, faces: refFaces } }
}

/**
 * Stretches joined where they meet (the same shot either side of a cut that isn't one), or are a
 * moment apart inside one camera shot
 */
function joinSpans(spans: Turn[], cuts: number[], durationMs: number): Turn[] {
  const bounds = [0, ...cuts.filter(c => c > 0 && c < durationMs), durationMs]
  const stretchOf = (t: number) => { let k = 0; while (k < bounds.length - 2 && t >= bounds[k + 1]) k++; return k }
  const joined: Turn[] = []
  for (const s of [...spans].sort((a, b) => a[0] - b[0])) {
    const last = joined[joined.length - 1]
    if (last && (s[0] - last[1] <= 1 || (s[0] - last[1] <= SHOT_JOIN_MS && stretchOf(last[1] - 1) === stretchOf(s[0])))) last[1] = Math.max(last[1], s[1])
    else joined.push([s[0], s[1]])
  }
  return joined
}

/**
 * Groups found together (shotSpans for each) where two of them claim the same stretch: it goes to the
 * group whose people it is closest to. Two people who look alike in their own close-ups — both in
 * black, sitting in the same place — can each read under SHOT_SAME_PEOPLE against the other's mark
 * as the light changes through a video (0.30 to his own, 0.37–0.43 to the other man's, measured
 * 2026-10-17); the nearer one is always the right one. A stretch whose closeness isn't known stays
 * with both, as before.
 */
export function settleGroups(results: ShotResult[], cuts: number[], durationMs: number): ShotResult[] {
  const minus = ([a, b]: Turn, [c, d]: Turn): Turn[] => (d <= a || c >= b ? [[a, b]] : [...(c > a ? [[a, c] as Turn] : []), ...(d < b ? [[d, b] as Turn] : [])])
  return results.map((r, i) => {
    if (!r.parts?.length) return r
    const kept: Turn[] = []
    for (const p of r.parts) {
      let left: Turn[] = [p.span]
      if (p.apart !== null) {
        results.forEach((o, j) => {
          if (j === i) return
          for (const q of o.parts ?? []) {
            // (the same closeness: the group listed first keeps it)
            if (q.apart === null || q.apart > p.apart! || (q.apart === p.apart && j > i)) continue
            left = left.flatMap(x => minus(x, q.span))
          }
        })
      }
      kept.push(...left.filter(([a, b]) => b - a >= SHOT_MIN_MS))
    }
    const spans = joinSpans(kept, cuts, durationMs)
    return { ...r, spans, path: r.path.filter(([t]) => spans.some(([a, b]) => t >= a && t <= b)) }
  })
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
  /**
   * A moment that shows it (ms from the start of the scan), and the faces there, left to right.
   * `person`: who it is (identities) — the same number in every angle they are in; none when not known
   */
  at: number
  faces: Array<{ x: number; y: number; w: number; h: number; person?: number }>
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
  // Who each face is (identities; faces too small to read have no one)
  const personAt = whoIs(fr, cuts)
  /** Two faces surely the same person (true), surely not (false), or not known (null) */
  const samePerson = (f: CropBox, g: CropBox) => {
    const p = personAt.get(f), q = personAt.get(g)
    return p === undefined || q === undefined ? null : p === q
  }

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
    const shot = shotSpans(fr, cuts, withAll[Math.floor(withAll.length / 2)].t, durationMs, personAt)
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

  // Light motion isn't another angle: an angle that is only the same people moved a little goes into
  // the earlier one. The same people by identity where it is known (a man in black and a woman in white
  // in close-ups from the same seat read alike by colours), else by colours
  const sameAngle = (p: { faces: CropBox[] }, q: { faces: CropBox[] }) => {
    if (p.faces.length !== q.faces.length) return false
    const apart: number[] = []
    for (let i = 0; i < p.faces.length; i++) {
      const f = p.faces[i], g = q.faces[i]
      const dx = Math.abs(f.x + f.w / 2 - (g.x + g.w / 2)), dy = Math.abs(f.y + f.h / 2 - (g.y + g.h / 2))
      if (dx > ANGLE_MOTION_X || dy > ANGLE_MOTION_Y || Math.max(f.w / g.w, g.w / f.w) > ANGLE_MOTION_SIZE) return false
      const same = samePerson(f, g)
      if (same === false) return false
      if (same === null && f.sig?.length && g.sig?.length) apart.push(l1(f.sig, g.sig))
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
        if (samePerson(f, r) === false) return
        if (dx + dy < bestD) { bestD = dx + dy; best = k }
      })
      if (best < 0) return false
      used.add(best)
      if (samePerson(faces[best], r) === null && faces[best].sig?.length && r.sig?.length) apart.push(l1(faces[best].sig!, r.sig!))
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
  // Each face's person: its own, else whoever is seen most in its place through the angle (the
  // angle's frame may catch a face too briefly to be read); nobody is two faces of one angle
  const personsOf = (faces: CropBox[], id: number) => {
    const used = new Set<number>()
    const out = faces.map(face => personAt.get(face))
    out.forEach(p => { if (p !== undefined) used.add(p) })
    return out.map((own, i) => {
      if (own !== undefined) return own
      const face = faces[i], votes = new Map<number, number>()
      fr.forEach((f, fi) => {
        const t = timeOf(fi)
        if (!stretches.some(s => s.angle === id && t >= s.a && t < s.b)) return
        for (const g of f.faces) {
          const p = personAt.get(g)
          if (p === undefined || used.has(p)) continue
          const dx = Math.abs(g.x + g.w / 2 - (face.x + face.w / 2)), dy = Math.abs(g.y + g.h / 2 - (face.y + face.h / 2))
          const reach = SHOT_REACH_FACE * face.w
          if (dx <= Math.max(SHOT_REACH_X, reach) && dy <= Math.max(SHOT_REACH_Y, reach) && Math.max(g.w / face.w, face.w / g.w) <= SHOT_SIZE_RATIO) {
            votes.set(p, (votes.get(p) ?? 0) + 1)
          }
        }
      })
      const best = [...votes].sort((p, q) => q[1] - p[1])[0]
      if (!best || best[1] < ID_MIN_FACES) return undefined
      used.add(best[0])
      return best[0]
    })
  }
  const angles: Angle[] = found.flatMap((a, id) => {
    if (!kept.has(id)) return []
    const persons = personsOf(a.faces, id)
    return [{
      people: a.faces.length, at: Math.round(a.at),
      faces: a.faces.map((f, i) => ({ x: round4(f.x), y: round4(f.y), w: round4(f.w), h: round4(f.h), ...(persons[i] !== undefined ? { person: persons[i] } : {}) })),
      spans: spansOf(s => s.angle === id),
    }]
  })
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

export async function handleTrackSubjectsJob(job: Job, queueSignal?: AbortSignal) {
  const { clip_id, start_ms, end_ms, auto, angles } = job.payload as unknown as TrackSubjectsJobPayload
  if (auto) return subEditClip(job.id, clip_id, queueSignal)
  // Stop in the editor's Subjects tool (user, 2026-10-24): the scan and the following stop like Sub-edit's.
  // The editor puts the clip's rows back itself (services/subjects.ts stopSubjects): nothing is written here
  const watch = stopWatch(job.id, queueSignal)
  try {
    if (angles) return await scanAngles(clip_id, start_ms, end_ms, watch.signal)
    return await followMarked(clip_id, start_ms, end_ms, watch.signal)
  } catch (e) {
    if (watch.stopped()) {
      console.log(`[track_subjects] clip ${clip_id}: ${angles ? 'scan' : 'following'} stopped by the user`)
      throw new Error(STOPPED)
    }
    throw e
  } finally {
    watch.end()
  }
}

/**
 * Stopped by the user: the job row says so (status failed, error STOPPED) — checked every 2 s. Everything
 * running (the download, FFmpeg, the face reading) is stopped through `signal`.
 */
function stopWatch(jobId: string, queueSignal?: AbortSignal) {
  const stop = new AbortController()
  const signal = queueSignal ? AbortSignal.any([queueSignal, stop.signal]) : stop.signal
  const watch = setInterval(async () => {
    try {
      const [row] = await db<{ status: string; error: string | null }[]>`SELECT status, error FROM jobs WHERE id = ${jobId}`
      if (row && row.status === 'failed' && row.error === STOPPED) stop.abort(new Error(STOPPED))
    } catch { /* the next check */ }
  }, 2000)
  return { signal, stopped: () => stop.signal.aborted, end: () => clearInterval(watch) }
}
/** The signal was stopped by the user (not a timeout) */
const isStop = (signal?: AbortSignal) => !!signal?.aborted && (signal.reason as Error | undefined)?.message === STOPPED

/** Follow the subjects waiting (queued) through [start_ms, end_ms] of the source video */
async function followMarked(clip_id: string, start_ms: number, end_ms: number, signal: AbortSignal) {
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
    // (people: everyone sitting side by side is someone to follow — not merged into one, as for a crop)
    const [info, cuts, speech] = await Promise.all([
      detectFaces(tmp, videoPath, from, to, signal, true, false, true),
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
    // (found together: a stretch two groups claim goes to the nearer one)
    const who = shots.length ? whoIs(info.frames, cuts) : undefined
    const found = settleGroups(shots.map(s => shotSpans(info.frames, cuts, s.mark_ms - from, to - from, who)), cuts, to - from)
    for (const [k, s] of shots.entries()) await saveShot(s.id, found[k], from, to, sourceCuts)
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
    if (!isStop(signal)) await setSubjects(ids, { status: 'failed', error: `Tracking failed: ${msg}`, progress: 100 }).catch(() => {})
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
      // The faces exactly as the tracker reads them (lips, people side by side, who is who): a face
      // listed here is one that can be followed. The list used to read every face its own way, so a
      // flower vase listed as a face, the tracker never found it, and marking it failed (2026-10-24)
      detectFaces(tmp, videoPath, from, to, signal, true, false, true),
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
    if (!isStop(signal)) await db`UPDATE clip_angles SET status = 'failed', progress = 100, error = ${msg}, updated_at = now() WHERE clip_id = ${clipId}`.catch(() => {})
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
  const { personOf } = identities(fr, ids)
  const tracks = new Map<number, Array<{ fi: number; face: CropBox }>>()
  fr.forEach((f, fi) => ids[fi].forEach((id, k) => {
    const list = tracks.get(id) ?? []
    list.push({ fi, face: f.faces[k] })
    tracks.set(id, list)
  }))
  // Who they are first (identities, across camera angles); the tracks too small to read after,
  // joining whoever their colours match
  const people: Array<{ person?: number; sig: number[] | null; frames: number; best: { fi: number; face: CropBox } }> = []
  const order = [...tracks].sort((a, b) => Number(personOf.has(b[0])) - Number(personOf.has(a[0])) || b[1].length - a[1].length)
  for (const [id, list] of order) {
    const sigs = list.map(x => x.face.sig).filter((v): v is number[] => !!v && v.length > 0)
    const sig = sigs.length ? meanOf(sigs) : null
    const person = personOf.get(id)
    const same = person !== undefined
      ? people.find(p => p.person === person)
      : sig ? people.find(p => p.sig && l1(sig, p.sig) < SAME_PERSON) : undefined
    const biggest = list.reduce((a, b) => (b.face.w * b.face.h > a.face.w * a.face.h ? b : a))
    if (same) {
      same.frames += list.length
      // (someone known by identity is marked where it can be read: the tracker finds the rest by it)
      const readable = same.person === undefined || person !== undefined
      if (readable && biggest.face.w * biggest.face.h > same.best.face.w * same.best.face.h) same.best = biggest
      continue
    }
    people.push({ person, sig, frames: list.length, best: biggest })
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
  const watch = stopWatch(jobId, queueSignal)
  const signal = watch.signal
  let created: string[] = []
  try {
    await subEditSteps(clipId, signal, ids => { created = ids }, t0)
  } catch (e) {
    // Stopped: nothing half-done stays (the people it had just registered go; the clip's framing was never touched)
    if (watch.stopped()) {
      if (created.length) await db`DELETE FROM clip_subjects WHERE id = ANY(${created})`.catch(() => {})
      console.log(`[track_subjects] sub-edit clip ${clipId}: stopped by the user (${((Date.now() - t0) / 1000).toFixed(0)}s)`)
      throw new Error(STOPPED)
    }
    throw e
  } finally {
    watch.end()
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
      detectFaces(tmp, videoPath, from, to, signal, true, false, true),
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
    const who = groups.length ? whoIs(info.frames, cuts) : undefined
    const settled = settleGroups(groups.map(g => shotSpans(info.frames, cuts, Number(g.mark_ms) - from, to - from, who)), cuts, to - from)
    for (const [k, g] of groups.entries()) await saveShot(g.id, settled[k], from, to, cuts.map(c => Math.round(c + from)))
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
