import { mkdtemp, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { r2DownloadToFile } from '../r2.js'
import db from '../db.js'
import type { Job, TrackSubjectsJobPayload } from '../types.js'
import { detectFaces, detectCuts, trackFaces, planSpeakers, FRAME_INTERVAL_MS, type FrameDetection, type CropBox, type SpokenWord } from './ai_edit.js'

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
export interface SubjectResult { id: string; path: PathPoint[]; error?: string }

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
    if (!best) { errors.set(mk.id, 'No face where you clicked. Pause where their face can be seen and mark them on it.'); continue }
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

  // Who is talking, frame by frame (a track id; 'split': two people trading lines quickly)
  const talking = speech.length && durationMs > 0 ? planSpeakers(fr, ids, speech, durationMs) : fr.map(() => null)

  return marks.map(mk => {
    const err = errors.get(mk.id)
    if (err) return { id: mk.id, path: [], error: err }
    const mine = owned.get(mk.id)!
    const byFrame = new Map<number, CropBox>()
    for (const id of mine) for (const [fi, face] of tracks.get(id)!) byFrame.set(fi, face)
    const path = [...byFrame.entries()].sort((a, b) => a[0] - b[0])
      .map(([fi, f]): PathPoint => {
        const who = talking[fi]
        const talks = who === 'split' || (typeof who === 'number' && mine.has(who)) ? 1 : 0
        return [timeOf(fi), round4(f.x), round4(f.y), round4(f.w), round4(f.h), talks]
      })
    return { id: mk.id, path }
  })
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
  const { clip_id, start_ms, end_ms } = job.payload as unknown as TrackSubjectsJobPayload
  // The subjects waiting when this job starts (ones marked later get a job of their own)
  const subjects = await db<{ id: string; mark_ms: number; mark_x: number; mark_y: number }[]>`
    UPDATE clip_subjects SET status = 'running', progress = 5, error = NULL
    WHERE clip_id = ${clip_id} AND status = 'queued'
    RETURNING id, mark_ms, mark_x, mark_y
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

    // The editing copy (at most 720p) is plenty for faces and much quicker to fetch; its frames
    // keep the original's times. The original when there is none.
    const videoPath = join(tmp, 'source.mp4')
    try {
      if (!clip.proxy_path) throw new Error('no editing copy')
      await r2DownloadToFile(clip.proxy_path, videoPath, signal)
    } catch {
      await r2DownloadToFile(clip.storage_path, videoPath, signal)
    }
    await setSubjects(ids, { progress: 25 })

    // Faces with their lips (who is talking), the camera cuts, and the words said in this stretch
    const [info, cuts, speech] = await Promise.all([
      detectFaces(tmp, videoPath, from, to, signal, true),
      detectCuts(videoPath, from, to, signal),
      spokenWords(clip.video_id, from, to),
    ])
    signal?.throwIfAborted()
    await setSubjects(ids, { progress: 90 })

    const results = followSubjects(info.frames, cuts, subjects.map(s => ({ id: s.id, t: s.mark_ms - from, x: s.mark_x, y: s.mark_y })), speech, to - from)
    for (const r of results) {
      if (r.error) {
        await db`UPDATE clip_subjects SET status = 'failed', error = ${r.error}, progress = 100, path = NULL WHERE id = ${r.id}`
        continue
      }
      // Times in the source video
      const path = r.path.map(([t, x, y, w, h, talks]) => [Math.round(t + from), x, y, w, h, talks])
      await db`
        UPDATE clip_subjects SET status = 'done', error = NULL, progress = 100,
          path = ${db.json(path as never)}, scan_start_ms = ${from}, scan_end_ms = ${to}
        WHERE id = ${r.id}
      `
    }
    const seen = results.map(r => `${r.error ? 'failed' : `${r.path.length} frames, talking in ${r.path.filter(p => p[5]).length}`}`).join(', ')
    console.log(`[track_subjects] clip ${clip_id}: ${info.frames.length} frames, ${cuts.length} cut(s), subjects: ${seen} (${((Date.now() - t0) / 1000).toFixed(0)}s)`)
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    await setSubjects(ids, { status: 'failed', error: `Tracking failed: ${msg}`, progress: 100 }).catch(() => {})
    throw err
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
}
