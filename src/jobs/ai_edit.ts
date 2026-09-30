import { GoogleGenerativeAI } from '@google/generative-ai'
import { spawn, execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { mkdir, mkdtemp, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { r2DownloadToFile } from '../r2.js'
import db from '../db.js'
import type { Job, AiEditJobPayload } from '../types.js'
import { handleTranscribeJob } from './transcribe.js'
import { findClips, type FoundClip } from '../lib/clipFinder.js'
import { generateClipText, fontForText, type ClipText } from '../lib/clipText.js'
import { brollEnabled, pickBrollMoments, searchStock, downloadStock, withBroll } from '../lib/broll.js'
import { r2UploadFile } from '../r2.js'
import { fileURLToPath } from 'node:url'
import { dirname } from 'node:path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const execFileAsync = promisify(execFile)

interface CropBox { x: number; y: number; w: number; h: number; score?: number; lip?: number }
interface FrameDetection { frame_index: number; person_count: number; faces: CropBox[] }
interface FaceInfo { face_count: number; face_boxes: CropBox[]; frames: FrameDetection[] }
interface SlotKf { t_ms: number; x: number; y: number; w: number; h: number }
interface ClipSegment { start_ms: number; end_ms: number; layout: 'vertical' | 'split'; slotKfs: SlotKf[][]; broll?: string }
/** What the framing needs to know about a clip */
export interface ClipAnalysis {
  info: FaceInfo
  /** Camera cuts, ms from the clip's start */
  cuts: number[]
  /** Width of a 9:16 crop as a fraction of the source width (full height) */
  reelW: number
  /** The source is already vertical: show the whole frame */
  portrait: boolean
  /** Spoken words in the clip, ms from its start (speaker tracking) */
  speech: SpokenWord[]
}
export interface SpokenWord { start_ms: number; end_ms: number; speaker_id?: string | null }

// Follow whoever is talking when 2+ people are in the picture. SPEAKER_TRACKING=off goes back
// to framing by face count and size only.
const SPEAKER_TRACKING = process.env.SPEAKER_TRACKING !== 'off'

// Frames analysed per second, and the time between them. 4 fps catches quick head turns and
// lets the crop follow within a quarter second.
const DETECT_FPS = 4
const FRAME_INTERVAL_MS = 1000 / DETECT_FPS

// Minimum segment duration before we'll commit to a layout switch.
// Avoids rapid flickering when a person briefly walks on/off screen.
const MIN_SEGMENT_MS = 3000
// A shot between two camera cuts shorter than this never gets its own segment
const MIN_SHOT_MS = 1000
// A cut only starts a new segment when the crop would move at least this much (fraction of width)
const CUT_MOVE = 0.05

const EMPTY: FaceInfo = { face_count: 0, face_boxes: [], frames: [] }

// Speaker tracking: a talker's mouth moves this much more than anyone else's (1.2× right after
// the transcript's speaker changes), and at least MIN_LIP_MOVE (std of lip gap / face height
// within a second; closed-mouth listeners measure ~0.0005)
const SPEAKER_RATIO = 1.5
const SPEAKER_RATIO_HINTED = 1.2
const MIN_LIP_MOVE = 0.0015
// Each speaker (or split) choice is held at least this long
const SPEAKER_HOLD_MS = 2000
// With no speaker change in the transcript there, a new speaker must win this many seconds in a
// row before the crop moves (a cough or a silent laugh can win one or two)
const UNHINTED_SWITCH_S = 3
// Speaker changing more than once within this window → both people, split screen
const BACK_AND_FORTH_MS = 4000

export async function probeSize(videoPath: string): Promise<{ width: number; height: number } | null> {
  try {
    const { stdout } = await execFileAsync('ffprobe', [
      '-v', 'error', '-select_streams', 'v:0',
      '-show_entries', 'stream=width,height:stream_side_data=rotation', '-of', 'json', videoPath,
    ])
    const st = JSON.parse(stdout).streams?.[0]
    if (!st?.width || !st?.height) return null
    // Phone videos are often stored landscape with a 90° rotation flag
    const rot = Math.abs(Number(st.side_data_list?.find((d: { rotation?: number }) => d.rotation !== undefined)?.rotation ?? 0)) % 180
    return rot === 90 ? { width: st.height, height: st.width } : { width: st.width, height: st.height }
  } catch {
    return null
  }
}

async function extractFrames(videoPath: string, startMs: number, endMs: number, outDir: string, signal?: AbortSignal): Promise<void> {
  await execFileAsync('ffmpeg', [
    '-ss', String(startMs / 1000),
    '-t', String((endMs - startMs) / 1000),
    '-i', videoPath,
    '-an', '-vf', `fps=${DETECT_FPS},scale=640:-1`,
    '-q:v', '3',
    join(outDir, 'frame_%04d.jpg'),
  ], { signal })
}

/**
 * Camera cuts inside the clip (ms from its start), from FFmpeg's scene-change detector on a
 * small copy of the picture. Empty if detection fails: the crop then just isn't reset.
 */
export async function detectCuts(videoPath: string, startMs: number, endMs: number, signal?: AbortSignal): Promise<number[]> {
  try {
    const { stderr } = await execFileAsync('ffmpeg', [
      '-hide_banner', '-nostats',
      '-ss', String(startMs / 1000), '-t', String((endMs - startMs) / 1000), '-i', videoPath,
      '-an', '-vf', 'scale=320:-1,scdet=threshold=10', '-f', 'null', '-',
    ], { signal, maxBuffer: 16 * 1024 * 1024 })
    const cuts = [...stderr.matchAll(/lavfi\.scd\.time:\s*([\d.]+)/g)].map(m => Math.round(parseFloat(m[1]) * 1000))
    return [...new Set(cuts)].filter(t => t > 0 && t < endMs - startMs).sort((a, b) => a - b)
  } catch (e) {
    console.warn('[ai_edit] cut detection failed:', e instanceof Error ? e.message : e)
    return []
  }
}

async function detectFaces(tmp: string, videoPath: string, startMs: number, endMs: number, signal?: AbortSignal, lips = false): Promise<FaceInfo> {
  const framesDir = join(tmp, `frames-${startMs}`)
  await mkdir(framesDir, { recursive: true })
  try {
    await extractFrames(videoPath, startMs, endMs, framesDir, signal)
  } catch {
    return EMPTY
  }

  const result = await new Promise<FaceInfo>((resolve) => {
    const script = join(__dirname, '../../src/python/face_detect.py')
    const proc = spawn('python3', [script, '--frames-dir', framesDir, ...(lips ? ['--lips'] : [])], { stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8', MPLBACKEND: 'Agg' } })
    const kill = () => proc.kill('SIGKILL')
    signal?.addEventListener('abort', kill, { once: true })
    let stdout = ''
    let stderr = ''
    proc.stdout?.on('data', (d: Buffer) => { stdout += d.toString() })
    proc.stderr?.on('data', (d: Buffer) => { stderr += d.toString() })
    proc.on('close', code => {
      signal?.removeEventListener('abort', kill)
      // The summary line; MediaPipe's own start-up chatter is left out
      const summary = stderr.split('\n').filter(l => l.startsWith('[detect]') || l.startsWith('[face_detect]')).join(' | ')
      if (summary) console.log('[ai_edit] face_detect:', summary.slice(-400))
      if (code !== 0 || !stdout.trim()) { resolve(EMPTY); return }
      try { resolve(JSON.parse(stdout) as FaceInfo) } catch { resolve(EMPTY) }
    })
    proc.on('error', () => resolve(EMPTY))
  })
  await rm(framesDir, { recursive: true, force: true }).catch(() => {})
  return result
}

/**
 * Everything the framing needs for one clip: faces 4× a second and the camera cuts (run side
 * by side). A source that is already vertical skips detection and shows the whole frame.
 */
export async function analyseClip(
  tmp: string, videoPath: string, startMs: number, endMs: number,
  size: { width: number; height: number } | null, words: Array<SpokenWord & { word?: string }>, signal?: AbortSignal,
): Promise<ClipAnalysis> {
  const speech = words
    .filter(w => w.end_ms > startMs && w.start_ms < endMs)
    .map(w => ({ start_ms: w.start_ms - startMs, end_ms: w.end_ms - startMs, speaker_id: w.speaker_id ?? null }))
  const portrait = !!size && size.height >= size.width
  // 9:16 crop at full height: 81/256 ≈ 0.316 of a 16:9 source
  const reelW = size ? Math.min(1, (9 / 16) * (size.height / size.width)) : 81 / 256
  if (portrait) return { info: EMPTY, cuts: [], reelW, portrait, speech }
  const [info, cuts] = await Promise.all([
    detectFaces(tmp, videoPath, startMs, endMs, signal, SPEAKER_TRACKING),
    detectCuts(videoPath, startMs, endMs, signal),
  ])
  return { info, cuts, reelW, portrait, speech }
}

// ── Crop stabilisation helpers ────────────────────────────────────────────────
//
// Per-frame subject detection has natural jitter: the detected center shifts a
// few percent each frame even when nobody moved. Two passes fix this:
//
//   1. EMA (α=0.3 per second of video): blends toward the new position gradually.
//      A large sudden move takes ~3 seconds to fully land — no snap cuts.
//
//   2. Dead zone (3% of frame width): once smoothed, suppress any remaining
//      movement smaller than the threshold. The crop stays completely still
//      when the subject hasn't really moved; only genuine repositioning moves it.
//
// Together: stationary subject → locked crop. Moving subject → smooth follow.
// Both restart at every camera cut, so the crop jumps to the new shot instead of sliding.

// α per frame giving the same response per second as α=0.3 at 1 frame per second
const EMA_ALPHA = 1 - Math.pow(1 - 0.3, 1 / DETECT_FPS)

function _ema(values: number[], alpha = EMA_ALPHA): number[] {
  if (!values.length) return []
  const out = [values[0]]
  for (let i = 1; i < values.length; i++) {
    out.push(alpha * values[i] + (1 - alpha) * out[i - 1])
  }
  return out
}

function _deadZone(values: number[], threshold = 0.03): number[] {
  if (!values.length) return []
  const out = [values[0]]
  for (let i = 1; i < values.length; i++) {
    out.push(Math.abs(values[i] - out[i - 1]) > threshold ? values[i] : out[i - 1])
  }
  return out
}

// Smooth raw detected center positions before converting to keyframes, starting afresh at
// each index in `resets` (the first frame after a camera cut).
function stabilise(rawCx: number[], resets: number[] = []): number[] {
  const out: number[] = []
  const bounds = [0, ...resets.filter(r => r > 0 && r < rawCx.length), rawCx.length]
  for (let i = 0; i < bounds.length - 1; i++) {
    out.push(..._deadZone(_ema(rawCx.slice(bounds[i], bounds[i + 1]))))
  }
  return out
}

const median = (v: number[]) => {
  const s = [...v].sort((a, b) => a - b)
  return s.length ? s[Math.floor(s.length / 2)] : 0.5
}

// ── Speaker tracking ──────────────────────────────────────────────────────────

const iou = (a: CropBox, b: CropBox) => {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x))
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y))
  const inter = ix * iy
  const union = a.w * a.h + b.w * b.h - inter
  return union > 0 ? inter / union : 0
}

// A person missing for up to this long (a hand over the face, a missed detection) keeps their id
const TRACK_GAP_MS = 1000

/**
 * A stable id per person within the clip: each face takes the id of the person whose last box
 * it overlaps most (IoU > 0.3), or whose centre is closest (under half a box width), among the
 * people seen in the last TRACK_GAP_MS. Ids start fresh after every camera cut.
 */
export function trackFaces(frames: FrameDetection[], cuts: number[]): number[][] {
  const ids: number[][] = []
  let next = 0
  let recent = new Map<number, { box: CropBox; t: number }>()
  frames.forEach((f, fi) => {
    const t = f.frame_index * FRAME_INTERVAL_MS
    const prevT = fi > 0 ? frames[fi - 1].frame_index * FRAME_INTERVAL_MS : -1
    if (cuts.some(c => c > prevT && c <= t)) recent = new Map()
    for (const [id, r] of recent) if (t - r.t > TRACK_GAP_MS) recent.delete(id)
    const taken = new Set<number>()
    const row = f.faces.map(face => {
      let best = -1, bestScore = 0
      for (const [id, r] of recent) {
        if (taken.has(id)) continue
        const overlap = iou(face, r.box)
        const dist = Math.abs((face.x + face.w / 2) - (r.box.x + r.box.w / 2))
        const score = overlap > 0.3 ? 1 + overlap : dist < r.box.w / 2 ? 1 - dist / r.box.w : 0
        if (score > bestScore) { best = id; bestScore = score }
      }
      if (best < 0) best = next++
      taken.add(best)
      return best
    })
    f.faces.forEach((face, i) => recent.set(row[i], { box: face, t }))
    ids.push(row)
  })
  return ids
}

const pstdev = (v: number[]) => {
  const m = v.reduce((a, b) => a + b, 0) / v.length
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length)
}

/** Who to frame, per frame: a person's track id, 'split' for a quick back-and-forth, or null (no opinion) */
export type SpeakerChoice = number | 'split' | null

/**
 * Decides, second by second, who is talking: while someone speaks (a transcript word overlaps
 * the second), the person whose mouth moves clearly more than anyone else's. The transcript's
 * speaker changes lower that bar (they mark likely turns) but its ids are not trusted across
 * chunks. Choices are held for SPEAKER_HOLD_MS; the speaker changing more than once within
 * BACK_AND_FORTH_MS means both are talking, so both are shown.
 */
export function planSpeakers(frames: FrameDetection[], tracks: number[][], speech: SpokenWord[], clipDurationMs: number): SpeakerChoice[] {
  const seconds = Math.ceil(clipDurationMs / 1000)
  const bySecond: number[][] = Array.from({ length: seconds }, () => [])
  frames.forEach((f, fi) => {
    const s = Math.floor((f.frame_index * FRAME_INTERVAL_MS) / 1000)
    if (s < seconds) bySecond[s].push(fi)
  })
  // Speaker-id changes in the transcript (null ids ignored)
  const turns: number[] = []
  let lastId: string | null = null
  for (const w of speech) {
    if (!w.speaker_id) continue
    if (lastId !== null && w.speaker_id !== lastId) turns.push(w.start_ms)
    lastId = w.speaker_id
  }

  const hintedAt = (s: number) => turns.some(t => t >= s * 1000 - 500 && t < s * 1000 + 1500)

  // Raw pick per second
  const raw: Array<number | null> = bySecond.map((fis, s) => {
    const from = s * 1000, to = from + 1000
    if (!speech.some(w => w.start_ms < to && w.end_ms > from)) return null
    const lips = new Map<number, number[]>()
    for (const fi of fis) {
      frames[fi].faces.forEach((face, i) => {
        if (face.lip === undefined) return
        const id = tracks[fi][i]
        lips.set(id, [...(lips.get(id) ?? []), face.lip])
      })
    }
    const moves = [...lips.entries()].filter(([, v]) => v.length >= 2).map(([id, v]) => ({ id, move: pstdev(v) }))
      .sort((a, b) => b.move - a.move)
    if (moves.length < 2 || moves[0].move < MIN_LIP_MOVE) return null
    return moves[0].move >= (hintedAt(s) ? SPEAKER_RATIO_HINTED : SPEAKER_RATIO) * moves[1].move ? moves[0].id : null
  })

  // Back-and-forth: more than one change of (known) speaker within the window around a second,
  // with a speaker change in the transcript there too
  const half = Math.round(BACK_AND_FORTH_MS / 2000)
  const busy = raw.map((_, s) => {
    const known = raw.slice(Math.max(0, s - half), s + half).filter((x): x is number => x !== null)
    let changes = 0
    for (let i = 1; i < known.length; i++) if (known[i] !== known[i - 1]) changes++
    // Lip readings alone are noisy: the transcript must also show a turn in the window
    const from = (s - half) * 1000, to = (s + half) * 1000
    return changes > 1 && turns.some(t => t >= from && t < to)
  })

  // Hold each choice; a speaker who left the picture (camera cut: new track ids) frees the choice.
  // Where the transcript marks a turn the crop follows at once; elsewhere a new choice must win
  // UNHINTED_SWITCH_S seconds in a row, so a burst of mouth movement never moves the crop.
  const holdS = SPEAKER_HOLD_MS / 1000
  const plan: SpeakerChoice[] = []
  let current: SpeakerChoice = null
  let since = -Infinity
  let pending: SpeakerChoice = null
  let pendingFor = 0
  for (let s = 0; s < seconds; s++) {
    const present = new Set(bySecond[s].flatMap(fi => tracks[fi]))
    if (typeof current === 'number' && !present.has(current)) { current = null; since = -Infinity }
    const want: SpeakerChoice = busy[s] ? 'split' : raw[s]
    if (want !== null && want !== current && (current === null || s - since >= holdS)) {
      pendingFor = want === pending ? pendingFor + 1 : 1
      pending = want
      if (current === null || hintedAt(s) || pendingFor >= UNHINTED_SWITCH_S) {
        current = want
        since = s
        pending = null
        pendingFor = 0
      }
    } else if (want !== pending) {
      pending = null
      pendingFor = 0
    }
    plan.push(current)
  }
  // Per frame. A switch the transcript marks happens at the turn's first word, not at the
  // start of the second (so the crop doesn't move a beat early or late)
  const switchAt = new Map<number, number>()
  for (let s = 1; s < seconds; s++) {
    if (plan[s] === plan[s - 1]) continue
    const turn = turns.find(t => t >= s * 1000 - 500 && t < s * 1000 + 1500)
    if (turn !== undefined) switchAt.set(s, turn)
  }
  return frames.map(f => {
    const t = f.frame_index * FRAME_INTERVAL_MS
    const s = Math.min(seconds - 1, Math.floor(t / 1000))
    const at = switchAt.get(s), next = switchAt.get(s + 1)
    if (at !== undefined && t < at) return plan[s - 1] ?? null        // switch later in this second
    if (next !== undefined && t >= next) return plan[s + 1] ?? null   // switch earlier, in the second before
    return plan[s] ?? null
  })
}

// ── Layout brain ──────────────────────────────────────────────────────────────
//
// Analyses the per-frame detections (4 a second) and produces one or more ClipSegments,
// each with the right layout (vertical / split) and smooth motion keyframes.
//
// Rules — generic, apply to any video type:
//   - 1 person visible   → vertical, track that person
//   - 2+ people visible  → split, tight on the 2 most prominent (Option B)
//   - Within a shot, layout switches only happen when a run of frames holds the same layout
//     for at least MIN_SEGMENT_MS (avoids single-frame noise causing a cut)
//   - Camera cuts: smoothing restarts at every cut. A cut starts a new segment when the layout
//     changes there or the crop would move by CUT_MOVE or more, so the crop cuts with the
//     picture (the renderer pans between keyframes inside a segment). Shots shorter than
//     MIN_SHOT_MS join the shot before them.
//   - Already-vertical source → one full-frame segment
//   - Speaker tracking (2+ people in the picture): a clear speaker → vertical on them; a quick
//     back-and-forth → split. A change of speaker is a new segment, so the crop cuts to them.
//     Frames with one person are framed exactly as without speaker tracking.
//
// Keyframe t_ms values are clip-relative (0 = first frame of the clip).
// render.py subtracts seg.start_ms per segment to get FFmpeg-relative time.
export function buildSegments(analysis: ClipAnalysis, clipDurationMs: number): ClipSegment[] {
  const { info, cuts, reelW, portrait } = analysis
  const center = (1 - reelW) / 2

  if (portrait) {
    return [{ start_ms: 0, end_ms: clipDurationMs, layout: 'vertical', slotKfs: [[{ t_ms: 0, x: 0, y: 0, w: 1, h: 1 }]] }]
  }

  const vertKf = (t_ms: number, cx: number): SlotKf => ({
    t_ms, y: 0, w: reelW, h: 1,
    x: Math.max(0, Math.min(1 - reelW, cx - reelW / 2)),
  })
  const splitKf = (t_ms: number, cx: number): SlotKf => {
    const CW = 0.5
    return { t_ms, y: 0, w: CW, h: 1, x: Math.max(0, Math.min(1 - CW, cx - CW / 2)) }
  }

  // Fallback: no frames detected → one centered vertical segment
  if (info.frames.length === 0) {
    return [{
      start_ms: 0, end_ms: clipDurationMs, layout: 'vertical',
      slotKfs: [[{ t_ms: 0, x: center, y: 0, w: reelW, h: 1 }]],
    }]
  }

  type Layout = 'vertical' | 'split'
  // What a frame shows: its layout, or vertical on one tracked speaker
  type Mode = Layout | `speaker:${number}`
  const frameTime = (f: FrameDetection) => f.frame_index * FRAME_INTERVAL_MS
  const layoutOfMode = (m: Mode): Layout => (m === 'split' ? 'split' : 'vertical')

  const frames = [...info.frames].sort((a, b) => a.frame_index - b.frame_index)
  const tracks = trackFaces(frames, cuts)
  const trackOf = new Map(frames.map((f, i) => [f, tracks[i]]))
  const speakers = SPEAKER_TRACKING && frames.some(f => f.faces.some(face => face.lip !== undefined))
    ? planSpeakers(frames, tracks, analysis.speech, clipDurationMs)
    : frames.map(() => null)
  const modeOf = new Map<FrameDetection, Mode>(frames.map((f, i) => {
    const people = f.person_count ?? f.faces.length
    const pick = speakers[i]
    if (people >= 2 && pick === 'split') return [f, 'split']
    if (people >= 2 && typeof pick === 'number' && tracks[i].includes(pick)) return [f, `speaker:${pick}`]
    return [f, people >= 2 ? 'split' : 'vertical']
  }))

  // Where each speaker was last seen, for frames where their face was missed
  const lastSeen = new Map<number, number>()
  frames.forEach((f, i) => tracks[i].forEach((id, k) => lastSeen.set(id, f.faces[k].x + f.faces[k].w / 2)))
  const seenAt = new Map<FrameDetection, Map<number, number>>()
  {
    const running = new Map<number, number>()
    frames.forEach((f, i) => {
      tracks[i].forEach((id, k) => running.set(id, f.faces[k].x + f.faces[k].w / 2))
      seenAt.set(f, new Map(running))
    })
  }

  // Raw subject centre per slot for a frame and layout
  const rawCx = (f: FrameDetection, layout: Layout, slotIdx: number) => {
    if (layout === 'vertical') {
      const mode = modeOf.get(f)
      if (mode?.startsWith('speaker:')) {
        const id = Number(mode.slice(8))
        const i = trackOf.get(f)!.indexOf(id)
        if (i >= 0) return f.faces[i].x + f.faces[i].w / 2
        // The speaker's face was missed here: stay with the face nearest where they were
        const at = seenAt.get(f)!.get(id) ?? lastSeen.get(id)
        if (at !== undefined) {
          const near = [...f.faces].sort((a, b) => Math.abs(a.x + a.w / 2 - at) - Math.abs(b.x + b.w / 2 - at))[0]
          return near && Math.abs(near.x + near.w / 2 - at) < 0.15 ? near.x + near.w / 2 : at
        }
      }
      const best = [...f.faces].sort((a, b) => (b.w * b.h) - (a.w * a.h))[0]
      return best ? best.x + best.w / 2 : 0.5
    }
    const bySize = [...f.faces].sort((a, b) => (b.w * b.h) - (a.w * a.h))
    const top2 = bySize.slice(0, 2).sort((a, b) => (a.x + a.w / 2) - (b.x + b.w / 2))
    // With one face in a split frame, which half it belongs in is unknown: leave both to the
    // frames around it (never show the same person in both halves)
    if (top2.length < 2) return null
    return top2[slotIdx].x + top2[slotIdx].w / 2
  }
  /** Missing values take the nearest known one in the run (the default halves if none) */
  const fill = (vals: Array<number | null>, fallback: number) => {
    const out = [...vals]
    let last: number | null = null
    for (let i = 0; i < out.length; i++) { if (out[i] == null) out[i] = last; else last = out[i] }
    last = null
    for (let i = out.length - 1; i >= 0; i--) { if (vals[i] != null) last = vals[i]; else if (out[i] == null) out[i] = last }
    return out.map(v => v ?? fallback)
  }

  const minFrames = Math.ceil(MIN_SEGMENT_MS / FRAME_INTERVAL_MS)
  const minSpeakerFrames = Math.ceil(SPEAKER_HOLD_MS / FRAME_INTERVAL_MS)

  // Step 1: shots — frames between camera cuts (a shot under MIN_SHOT_MS joins the one before)
  type Shot = { start_ms: number; frames: FrameDetection[] }
  const shots: Shot[] = [{ start_ms: 0, frames: [] }]
  const cutQueue = [...cuts]
  for (const f of frames) {
    while (cutQueue.length && frameTime(f) >= cutQueue[0]) {
      shots.push({ start_ms: cutQueue.shift()!, frames: [] })
    }
    shots[shots.length - 1].frames.push(f)
  }
  const merged: Shot[] = []
  for (const shot of shots) {
    if (!shot.frames.length) continue
    const last = merged[merged.length - 1]
    const nextStart = shots[shots.indexOf(shot) + 1]?.start_ms ?? clipDurationMs
    if (last && nextStart - shot.start_ms < MIN_SHOT_MS) last.frames.push(...shot.frames)
    else merged.push({ start_ms: merged.length ? shot.start_ms : 0, frames: [...shot.frames] })
  }

  // Step 2: within each shot, runs of the same mode; runs under MIN_SEGMENT_MS (SPEAKER_HOLD_MS
  // for a speaker) merge into their bigger neighbour (in that shot only)
  type Run = { mode: Mode; start_ms: number; frames: FrameDetection[]; cutStart: boolean }
  const shortRun = (r: Run) => r.frames.length < (r.mode.startsWith('speaker:') ? minSpeakerFrames : minFrames)
  const runs: Run[] = []
  for (const shot of merged) {
    const shotRuns: Run[] = []
    for (const f of shot.frames) {
      const mode = modeOf.get(f)!
      const last = shotRuns[shotRuns.length - 1]
      if (last && last.mode === mode) last.frames.push(f)
      else shotRuns.push({ mode, start_ms: shotRuns.length ? frameTime(f) : shot.start_ms, frames: [f], cutStart: shotRuns.length === 0 })
    }
    // The shortest run goes first, and neighbours of the same mode are joined before judging
    // length: otherwise a long-ish run could be swallowed by a one-frame blip next to it
    // (a single frame with two faces turned three seconds of one person into split screen)
    const joinSame = () => {
      for (let i = shotRuns.length - 1; i > 0; i--) {
        if (shotRuns[i].mode === shotRuns[i - 1].mode) {
          shotRuns[i - 1].frames.push(...shotRuns[i].frames)
          shotRuns.splice(i, 1)
        }
      }
    }
    let changed = true
    while (changed && shotRuns.length > 1) {
      changed = false
      joinSame()
      if (shotRuns.length < 2) break
      const shortest = shotRuns
        .map((r, i) => ({ r, i }))
        .filter(x => shortRun(x.r))
        .sort((a, b) => a.r.frames.length - b.r.frames.length)[0]
      for (const i of shortest ? [shortest.i] : []) {
        const left = i > 0 ? shotRuns[i - 1].frames.length : -1
        const right = i < shotRuns.length - 1 ? shotRuns[i + 1].frames.length : -1
        // Absorbed frames take on the neighbour's mode
        for (const f of shotRuns[i].frames) modeOf.set(f, shotRuns[left >= right ? i - 1 : i + 1].mode)
        if (left >= right) {
          shotRuns[i - 1].frames.push(...shotRuns[i].frames)
        } else {
          shotRuns[i + 1].frames = [...shotRuns[i].frames, ...shotRuns[i + 1].frames]
          shotRuns[i + 1].start_ms = shotRuns[i].start_ms
          shotRuns[i + 1].cutStart = shotRuns[i].cutStart
        }
        shotRuns.splice(i, 1)
        changed = true
        break
      }
    }
    // Neighbours left with the same mode by the merging become one run
    for (const r of shotRuns) {
      const last = runs[runs.length - 1]
      if (last && !r.cutStart && last.mode === r.mode) last.frames.push(...r.frames)
      else runs.push(r)
    }
  }

  // Step 3: a run joins the run before it when the layout is the same and the crop would barely
  // move — then a cut needs no segment boundary, only a smoothing reset. Otherwise (another
  // speaker, another shot) it starts a new segment, so the crop cuts instead of panning.
  const slotCount = (layout: Layout) => (layout === 'split' ? 2 : 1)
  const edgeCx = (run: Run, slot: number, atEnd: boolean) => {
    const edge = atEnd ? run.frames.slice(-DETECT_FPS) : run.frames.slice(0, DETECT_FPS)
    return median(fill(edge.map(f => rawCx(f, layoutOfMode(run.mode), slot)), slot === 0 ? 0.3 : 0.7))
  }
  type Group = { layout: Layout; start_ms: number; frames: FrameDetection[]; resets: number[] }
  const groups: Group[] = []
  runs.forEach((run, ri) => {
    const last = groups[groups.length - 1]
    const lastRun = runs[ri - 1]
    const layout = layoutOfMode(run.mode)
    const cropClose = () => Array.from({ length: slotCount(layout) }, (_, s) => s)
      .every(s => Math.abs(edgeCx(lastRun, s, true) - edgeCx(run, s, false)) < CUT_MOVE)
    const join = !!last && last.layout === layout && ((!run.cutStart && lastRun.mode === run.mode) || cropClose())
    if (join) {
      if (run.cutStart) last.resets.push(last.frames.length)
      last.frames.push(...run.frames)
    } else {
      groups.push({ layout, start_ms: run.start_ms, frames: [...run.frames], resets: [] })
    }
  })

  // Step 4: one ClipSegment per group, with keyframes smoothed between cuts
  return groups.map((g, gi) => {
    const start_ms = gi === 0 ? 0 : g.start_ms
    const end_ms = groups[gi + 1]?.start_ms ?? clipDurationMs
    // The first keyframe sits on the segment start (the renderer extrapolates before it)
    const kfTime = (f: FrameDetection, i: number) => (i === 0 ? start_ms : Math.max(start_ms, frameTime(f)))
    const slotKfs = Array.from({ length: slotCount(g.layout) }, (_, slot) => {
      const cx = stabilise(fill(g.frames.map(f => rawCx(f, g.layout, slot)), g.layout === 'split' ? (slot === 0 ? 0.3 : 0.7) : 0.5), g.resets)
      return g.frames.map((f, i) => g.layout === 'vertical' ? vertKf(kfTime(f, i), cx[i]) : splitKf(kfTime(f, i), cx[i]))
    })
    return { start_ms, end_ms, layout: g.layout, slotKfs }
  })
}

type Word = { word: string; start_ms: number; end_ms: number; speaker_id?: string | null }

// Gemini picks the clips: the numbered-line, windowed, scored method shared with the app's
// "Best moments" (src/lib/clipFinder.ts)
async function selectHighlights(words: Word[], clipCount: number, durationMs: number, exclude: Array<[number, number]> = []): Promise<FoundClip[]> {
  const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!)
  const ask = async (system: string, user: string) => {
    const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash', systemInstruction: system })
    return (await model.generateContent(user)).response.text()
  }
  // Enough candidates per window that a short video (one window) can still fill the request
  const windowCount = Math.max(1, Math.ceil(durationMs / (9 * 60_000)))
  // More candidates when earlier batches already took some moments
  const perWindow = Math.min(12, Math.max(4, Math.ceil((clipCount * 1.5 + exclude.length) / windowCount)))
  return findClips({
    words, durationMs, mode: { kind: 'best' }, ask,
    limit: clipCount, perWindow, spread: true, exclude,
    log: msg => console.warn('[ai_edit] clip finder:', msg),
  })
}

// ── Captions ──────────────────────────────────────────────────────────────────
//
// AI clips get captions on, with the editor's defaults (DEFAULT_CAPTION_STYLE in the app).
// The font follows the script the clip is spoken in; ids are _FONT_NAMES in render.py.
// Scripts without a bundled font yet (Tamil, Kannada, …) are shown in Roman letters.

const SCRIPTS: Array<[string, RegExp]> = [
  ['telugu', /[\u0C00-\u0C7F]/], ['devanagari', /[\u0900-\u097F]/], ['latin', /[A-Za-z]/],
  ['other_indic', /[\u0980-\u0BFF\u0C80-\u0DFF]/], // Bengali, Gurmukhi, Gujarati, Odia, Tamil, Kannada, Malayalam, Sinhala
]

export function captionFontFor(words: Word[]): { font: string; language: string | null } {
  const counts: Record<string, number> = {}
  for (const w of words) {
    for (const ch of w.word) {
      for (const [name, re] of SCRIPTS) if (re.test(ch)) { counts[name] = (counts[name] ?? 0) + 1; break }
    }
  }
  const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0]
  if (top === 'telugu') return { font: 'noto-sans-telugu', language: null }
  if (top === 'devanagari') return { font: 'noto-sans-devanagari', language: null }
  if (top === 'other_indic') return { font: 'roboto', language: 'roman' }
  return { font: 'roboto', language: null }
}

// ── Transcript ────────────────────────────────────────────────────────────────

/**
 * Whether the video's transcript covers the whole video. A clip made before the full-video
 * captions finished leaves a transcript holding only that clip's words; picking clips from it
 * would miss everything else. "Whole" = words in at least 60% of the video's minutes.
 */
function coversVideo(words: Word[], durationMs: number) {
  if (words.length === 0) return false
  const minutes = Math.max(1, Math.ceil(durationMs / 60_000))
  const spoken = new Set(words.map(w => Math.floor(w.start_ms / 60_000)))
  return spoken.size / minutes >= 0.6
}

async function loadWords(videoId: string): Promise<Word[]> {
  const [row] = await db`SELECT id FROM transcripts WHERE video_id = ${videoId} ORDER BY created_at DESC LIMIT 1`
  if (!row) return []
  return db<Word[]>`SELECT word, start_ms, end_ms, speaker_id FROM transcript_words WHERE transcript_id = ${row.id} ORDER BY start_ms`
}

/** Waits (up to 10 min) for the full-video captions started at upload, instead of paying for a second transcription */
async function waitForFullTranscription(videoId: string, signal?: AbortSignal) {
  const deadline = Date.now() + 10 * 60_000
  while (Date.now() < deadline) {
    const [running] = await db`
      SELECT id FROM jobs
      WHERE type = 'transcribe' AND payload->>'video_id' = ${videoId}
        AND payload->>'transcribe_full' = 'true' AND status IN ('queued', 'processing')
      LIMIT 1
    `
    if (!running) return
    signal?.throwIfAborted()
    await new Promise(r => setTimeout(r, 5000))
  }
}

async function setProgress(aiEditJobId: string, progress: number) {
  await db`UPDATE ai_edit_jobs SET progress = ${Math.round(progress)} WHERE id = ${aiEditJobId}`
    .catch(e => console.warn('[ai_edit] could not save progress:', e))
}

export async function handleAiEditJob(job: Job, signal?: AbortSignal) {
  const payload = job.payload as unknown as AiEditJobPayload
  const { ai_edit_job_id, video_id, clip_count } = payload
  const addBroll = payload.add_broll === true && brollEnabled()
  if (payload.add_broll && !addBroll) console.log('[ai_edit] B-roll asked for but AUTO_BROLL is off or no stock API key (PEXELS_API_KEY / PIXABAY_API_KEY): skipping it')
  const t0 = Date.now()

  await db`UPDATE ai_edit_jobs SET status = 'running', progress = 5 WHERE id = ${ai_edit_job_id}`

  try {
    // 1. Get video info
    const [video] = await db`
      SELECT v.id, v.user_id, v.storage_path, v.duration_ms, v.title
      FROM videos v
      WHERE v.id = ${video_id}
    `
    if (!video?.storage_path) throw new Error('Video not found or missing storage path')

    // 2. Ensure a whole-video transcript — transcribe inline if there isn't one.
    //    Always from storage (R2), never yt-dlp, so this also runs on Railway.
    // A complete transcript is used as it is; only a missing or partial one waits for the
    // upload's full-video captions (if they are running) before transcribing here
    let words = await loadWords(video_id)
    const durationFor = (w: Word[]) => video.duration_ms ?? (w.length ? w[w.length - 1].end_ms : 0)
    if (!coversVideo(words, durationFor(words))) {
      await waitForFullTranscription(video_id, signal)
      words = await loadWords(video_id)
    }
    const knownDurationMs = durationFor(words)
    if (!coversVideo(words, knownDurationMs)) {
      console.log(`[ai_edit] Transcript for ${video_id} missing or partial (${words.length} words) — transcribing now`)
      await handleTranscribeJob({
        id: `ai-edit-tx-${ai_edit_job_id}`,
        type: 'transcribe',
        payload: { video_id, storage_path: video.storage_path, is_retranscribe: true } as Record<string, unknown>,
        status: 'processing',
        error: null,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      }, signal)
      words = await loadWords(video_id)
    }
    if (!words.length) throw new Error('This video has no speech to make clips from')
    const durationMs = video.duration_ms ?? words[words.length - 1].end_ms
    await setProgress(ai_edit_job_id, 20)

    // 3. Pick the clips across the whole video, never repeating one an earlier batch made
    const earlier = await db`
      SELECT start_ms, end_ms FROM clips WHERE video_id = ${video_id} AND ai_edit_job_id IS NOT NULL AND ai_edit_job_id <> ${ai_edit_job_id}
    `
    const exclude = earlier.map(c => [c.start_ms as number, c.end_ms as number] as [number, number])
    const highlights = await selectHighlights(words, clip_count, durationMs, exclude)
    if (highlights.length === 0) {
      throw new Error(exclude.length
        ? `AI could not find new moments beyond the ${exclude.length} clips already made from this video`
        : 'AI could not find any good clips in this video')
    }
    console.log(`[ai_edit] ${highlights.length}/${clip_count} clips picked for job ${ai_edit_job_id} (${((Date.now() - t0) / 1000).toFixed(0)}s)`)
    await setProgress(ai_edit_job_id, 30)

    // Hook, title, post caption and hashtags per clip (a clip whose call fails just has none)
    const texts: Array<ClipText | null> = await Promise.all(highlights.map(async (h, i) => {
      await new Promise(r => setTimeout(r, (i % 4) * 250)) // spread the calls a little
      try {
        return await generateClipText(words.filter(w => w.start_ms >= h.start_ms && w.start_ms < h.end_ms), video.title ?? null, process.env.GEMINI_API_KEY!)
      } catch (e) {
        console.warn(`[ai_edit] clip text failed for ${h.start_ms}-${h.end_ms}:`, e instanceof Error ? e.message : e)
        return null
      }
    }))
    await setProgress(ai_edit_job_id, 35)

    // 4. Download video once (for framing)
    const tmp = await mkdtemp(join(tmpdir(), 'ai-edit-'))
    try {
      const videoPath = join(tmp, 'source.mp4')
      console.log(`[ai_edit] Downloading video ${video.storage_path}`)
      await r2DownloadToFile(video.storage_path, videoPath, signal)
      const size = await probeSize(videoPath)
      await setProgress(ai_edit_job_id, 45)

      // 5. Frame each clip and save it as a draft. Nothing is exported: the user previews the
      //    clips in the app and exports the ones they want.
      for (let hi = 0; hi < highlights.length; hi++) {
        signal?.throwIfAborted()
        const highlight = highlights[hi]
        const clipDurationMs = highlight.end_ms - highlight.start_ms

        // Faces 4× a second and camera cuts across the entire clip
        const tDetect = Date.now()
        const analysis = await analyseClip(tmp, videoPath, highlight.start_ms, highlight.end_ms, size, words, signal)
        const detectS = ((Date.now() - tDetect) / 1000).toFixed(1)

        // Brain: dynamically switch layout within the clip
        let segments: ClipSegment[] = buildSegments(analysis, clipDurationMs)
        if (addBroll) segments = await addStockBroll(segments, words, highlight, video.user_id, tmp)
        const clipWords = words.filter(w => w.start_ms >= highlight.start_ms && w.start_ms < highlight.end_ms)
        const caption = captionFontFor(clipWords)

        const clip_id = await saveClip(video_id, ai_edit_job_id, highlight, segments, caption, texts[hi])

        const layoutSummary = segments.map(s =>
          `${s.layout}(${s.start_ms / 1000}s–${s.end_ms / 1000}s,${s.slotKfs[0]?.length ?? 0}kf)`
        ).join(' | ')
        console.log(`[ai_edit] Clip ${clip_id} (score ${highlight.score}, ${caption.font}${caption.language ? '/' + caption.language : ''}): ${segments.length} seg — ${layoutSummary}`)
        const counts = analysis.info.frames.map(f => f.person_count)
        console.log(analysis.portrait
          ? `[ai_edit] Detection summary: vertical source, full frame`
          : `[ai_edit] Detection summary: ${counts.length} frames in ${detectS}s, faces/frame min=${counts.length ? Math.min(...counts) : 0} `
            + `avg=${counts.length ? (counts.reduce((a, b) => a + b, 0) / counts.length).toFixed(2) : 0} max=${analysis.info.face_count}, `
            + `${analysis.cuts.length} camera cut(s)`)
        await setProgress(ai_edit_job_id, 45 + (55 * (hi + 1)) / highlights.length - 1)
      }
    } finally {
      await rm(tmp, { recursive: true, force: true })
    }

    await db`UPDATE ai_edit_jobs SET status = 'done', progress = 100 WHERE id = ${ai_edit_job_id}`
    console.log(`[ai_edit] Job ${ai_edit_job_id} done in ${((Date.now() - t0) / 1000).toFixed(0)}s (clips ready to preview)`)
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    console.error(`[ai_edit] Job ${ai_edit_job_id} failed:`, msg)
    await db`UPDATE ai_edit_jobs SET status = 'failed', error = ${msg} WHERE id = ${ai_edit_job_id}`
    throw err
  }
}

/**
 * Stock shots over the clip where a picture helps (src/lib/broll.ts). Any failure (no result,
 * Pexels down, a bad file) just means fewer or no shots; the clip is still made.
 */
async function addStockBroll(segments: ClipSegment[], words: Word[], highlight: FoundClip, userId: string, tmp: string): Promise<ClipSegment[]> {
  try {
    const moments = await pickBrollMoments(words, highlight.start_ms, highlight.end_ms, process.env.GEMINI_API_KEY!)
    const shots: Array<{ start_ms: number; end_ms: number; videoId: string }> = []
    for (const m of moments) {
      try {
        const stock = await searchStock(m.query)
        if (!stock) { console.log(`[ai_edit] B-roll: no stock video for "${m.query}"`); continue }
        const videoId = await stockAsset(userId, stock.ref, stock.url, m.query, tmp)
        if (videoId) shots.push({ start_ms: m.start_ms, end_ms: m.end_ms, videoId })
        console.log(`[ai_edit] B-roll "${m.query}" at ${(m.start_ms / 1000).toFixed(1)}s: ${stock.ref} (${stock.width}x${stock.height})`)
      } catch (e) {
        console.warn(`[ai_edit] B-roll "${m.query}" skipped:`, e instanceof Error ? e.message : e)
      }
    }
    return withBroll(segments, shots)
  } catch (e) {
    console.warn('[ai_edit] B-roll skipped:', e instanceof Error ? e.message : e)
    return segments
  }
}

/** The user's copy of a stock video (an 'asset' video, so the render job may use it), made once */
async function stockAsset(userId: string, ref: string, url: string, query: string, tmp: string): Promise<string | null> {
  const [existing] = await db`SELECT id FROM videos WHERE user_id = ${userId} AND stock_ref = ${ref} AND storage_path IS NOT NULL LIMIT 1`
  if (existing) return existing.id as string
  const path = await downloadStock(url, tmp, ref)
  const size = await probeSize(path)
  if (!size) throw new Error('not a readable video')
  const { stdout } = await execFileAsync('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path])
  const durationMs = Math.round(parseFloat(stdout) * 1000) || null
  const storagePath = `raw/${userId}/stock-${ref.replace(':', '-')}.mp4`
  await r2UploadFile(storagePath, path, 'video/mp4')
  const [row] = await db`
    INSERT INTO videos (user_id, source_type, storage_path, status, duration_ms, title, role, stock_ref)
    VALUES (${userId}, 'upload', ${storagePath}, 'ready', ${durationMs}, ${`${ref.startsWith('pixabay') ? 'Pixabay' : 'Pexels'}: ${query}`.slice(0, 120)}, 'asset', ${ref})
    RETURNING id
  `
  return row.id as string
}

/**
 * One clip with its formats, crop boxes, keyframes and caption style, in one transaction.
 * AI clips export with pauses and filler words removed (the user can switch it off).
 * Rows go in as bulk inserts: the database is several hundred ms away, and a round trip per
 * keyframe took minutes per clip.
 */
async function saveClip(
  videoId: string, aiEditJobId: string, highlight: FoundClip, segments: ClipSegment[],
  caption: { font: string; language: string | null },
  text: ClipText | null,
): Promise<string> {
  return db.begin(async tx => {
    const [clipRow] = await tx`
      INSERT INTO clips (video_id, start_ms, end_ms, status, title, ai_edit_job_id, ai_score, ai_reason, remove_fillers,
        hook_text, post_caption, hashtags)
      VALUES (${videoId}, ${highlight.start_ms}, ${highlight.end_ms}, 'draft', ${(text?.title || highlight.title).slice(0, 120) || 'Highlight'},
        ${aiEditJobId}, ${highlight.score}, ${highlight.reason || null}, true,
        ${text?.hook ?? null}, ${text?.post_caption ?? null}, ${text?.hashtags ?? null})
      RETURNING id
    `
    const clipId = clipRow.id as string

    // The hook, over the first 3 seconds near the top (x null = centred), like any text overlay
    if (text?.hook) {
      await tx`
        INSERT INTO text_overlays (clip_id, text, start_ms, end_ms, x, y, font, size, color)
        VALUES (${clipId}, ${text.hook}, 0, ${Math.min(3000, highlight.end_ms - highlight.start_ms)}, NULL, 0.15,
          ${fontForText(text.hook)}, 76, '#FFFFFF')
      `
    }

    const segRows = await tx`
      INSERT INTO segments ${tx(segments.map((seg, si) => ({
        clip_id: clipId, start_ms: seg.start_ms, end_ms: seg.end_ms, layout: seg.layout, sort_order: si, video_offset_ms: null,
      })))}
      RETURNING id, sort_order
    `
    const segIdByOrder = new Map(segRows.map(r => [r.sort_order as number, r.id as string]))

    // source_offset_ms is where in the clip this segment's video starts: render.py trims
    // the main video from it, so 0 made every later segment replay the clip's start
    // A stock shot plays its video from the start, muted (the speaker keeps talking under it)
    const boxInputs = segments.flatMap((seg, si) => seg.slotKfs.map((kfs, slotIdx) => ({
      row: seg.broll
        ? { segment_id: segIdByOrder.get(si)!, slot_index: slotIdx, source_video_id: seg.broll, source_offset_ms: 0, muted: true }
        : { segment_id: segIdByOrder.get(si)!, slot_index: slotIdx, source_video_id: null, source_offset_ms: seg.start_ms, muted: false },
      kfs,
    })))
    const boxRows = await tx`INSERT INTO crop_boxes ${tx(boxInputs.map(b => b.row))} RETURNING id, segment_id, slot_index`
    const boxId = new Map(boxRows.map(r => [`${r.segment_id}:${r.slot_index}`, r.id as string]))
    const kfRows = boxInputs.flatMap(b => b.kfs.map(kf => ({
      box_id: boxId.get(`${b.row.segment_id}:${b.row.slot_index}`)!, t_ms: kf.t_ms, x: kf.x, y: kf.y, w: kf.w, h: kf.h,
    })))
    // Postgres allows 65,535 parameters per statement: 6 per keyframe
    for (let i = 0; i < kfRows.length; i += 5000) await tx`INSERT INTO box_keyframes ${tx(kfRows.slice(i, i + 5000))}`

    await tx`
      INSERT INTO caption_styles (clip_id, font, size, color, position, animation, language)
      VALUES (${clipId}, ${caption.font}, 52, '#FFFFFF', 'bottom-center', 'karaoke', ${caption.language})
    `
    return clipId
  })
}
