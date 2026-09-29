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
import { fileURLToPath } from 'node:url'
import { dirname } from 'node:path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const execFileAsync = promisify(execFile)

interface CropBox { x: number; y: number; w: number; h: number; score?: number }
interface FrameDetection { frame_index: number; person_count: number; faces: CropBox[] }
interface FaceInfo { face_count: number; face_boxes: CropBox[]; frames: FrameDetection[] }
interface SlotKf { t_ms: number; x: number; y: number; w: number; h: number }
interface ClipSegment { start_ms: number; end_ms: number; layout: 'vertical' | 'split'; slotKfs: SlotKf[][] }
/** What the framing needs to know about a clip */
export interface ClipAnalysis {
  info: FaceInfo
  /** Camera cuts, ms from the clip's start */
  cuts: number[]
  /** Width of a 9:16 crop as a fraction of the source width (full height) */
  reelW: number
  /** The source is already vertical: show the whole frame */
  portrait: boolean
}

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

async function detectFaces(tmp: string, videoPath: string, startMs: number, endMs: number, signal?: AbortSignal): Promise<FaceInfo> {
  const framesDir = join(tmp, `frames-${startMs}`)
  await mkdir(framesDir, { recursive: true })
  try {
    await extractFrames(videoPath, startMs, endMs, framesDir, signal)
  } catch {
    return EMPTY
  }

  const result = await new Promise<FaceInfo>((resolve) => {
    const script = join(__dirname, '../../src/python/face_detect.py')
    const proc = spawn('python3', [script, '--frames-dir', framesDir], { stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8', MPLBACKEND: 'Agg' } })
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
  size: { width: number; height: number } | null, signal?: AbortSignal,
): Promise<ClipAnalysis> {
  const portrait = !!size && size.height >= size.width
  // 9:16 crop at full height: 81/256 ≈ 0.316 of a 16:9 source
  const reelW = size ? Math.min(1, (9 / 16) * (size.height / size.width)) : 81 / 256
  if (portrait) return { info: EMPTY, cuts: [], reelW, portrait }
  const [info, cuts] = await Promise.all([
    detectFaces(tmp, videoPath, startMs, endMs, signal),
    detectCuts(videoPath, startMs, endMs, signal),
  ])
  return { info, cuts, reelW, portrait }
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
  const frameTime = (f: FrameDetection) => f.frame_index * FRAME_INTERVAL_MS
  const layoutOf = (f: FrameDetection): Layout => ((f.person_count ?? f.faces.length) >= 2 ? 'split' : 'vertical')
  // Raw subject centre per slot for a frame and layout
  const rawCx = (f: FrameDetection, layout: Layout, slotIdx: number) => {
    if (layout === 'vertical') {
      const best = [...f.faces].sort((a, b) => (b.w * b.h) - (a.w * a.h))[0]
      return best ? best.x + best.w / 2 : 0.5
    }
    const bySize = [...f.faces].sort((a, b) => (b.w * b.h) - (a.w * a.h))
    const top2 = bySize.slice(0, 2).sort((a, b) => (a.x + a.w / 2) - (b.x + b.w / 2))
    const face = top2[slotIdx] ?? top2[0]
    return face ? face.x + face.w / 2 : slotIdx === 0 ? 0.3 : 0.7
  }

  const frames = [...info.frames].sort((a, b) => a.frame_index - b.frame_index)
  const minFrames = Math.ceil(MIN_SEGMENT_MS / FRAME_INTERVAL_MS)

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

  // Step 2: within each shot, runs of the same layout; runs under MIN_SEGMENT_MS merge into
  // their bigger neighbour (in that shot only)
  type Run = { layout: Layout; start_ms: number; frames: FrameDetection[]; cutStart: boolean }
  const runs: Run[] = []
  for (const shot of merged) {
    const shotRuns: Run[] = []
    for (const f of shot.frames) {
      const layout = layoutOf(f)
      const last = shotRuns[shotRuns.length - 1]
      if (last && last.layout === layout) last.frames.push(f)
      else shotRuns.push({ layout, start_ms: shotRuns.length ? frameTime(f) : shot.start_ms, frames: [f], cutStart: shotRuns.length === 0 })
    }
    let changed = true
    while (changed && shotRuns.length > 1) {
      changed = false
      for (let i = 0; i < shotRuns.length; i++) {
        if (shotRuns[i].frames.length >= minFrames) continue
        const left = i > 0 ? shotRuns[i - 1].frames.length : -1
        const right = i < shotRuns.length - 1 ? shotRuns[i + 1].frames.length : -1
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
    runs.push(...shotRuns)
  }

  // Step 3: a run that starts at a cut joins the run before it when the layout is the same and
  // the crop would barely move — the cut then needs no segment boundary, only a smoothing reset
  const slotCount = (layout: Layout) => (layout === 'split' ? 2 : 1)
  const edgeCx = (run: Run, slot: number, atEnd: boolean) => {
    const edge = atEnd ? run.frames.slice(-DETECT_FPS) : run.frames.slice(0, DETECT_FPS)
    return median(edge.map(f => rawCx(f, run.layout, slot)))
  }
  type Group = { layout: Layout; start_ms: number; frames: FrameDetection[]; resets: number[] }
  const groups: Group[] = []
  for (const run of runs) {
    const last = groups[groups.length - 1]
    const lastRun = runs[runs.indexOf(run) - 1]
    // Same layout with no cut between them (two runs left side by side by the merge above)
    // always join; across a cut only when the crop barely moves
    const join = !!last && last.layout === run.layout && (!run.cutStart
      || Array.from({ length: slotCount(run.layout) }, (_, s) => s)
        .every(s => Math.abs(edgeCx(lastRun, s, true) - edgeCx(run, s, false)) < CUT_MOVE))
    if (join) {
      if (run.cutStart) last.resets.push(last.frames.length)
      last.frames.push(...run.frames)
    } else {
      groups.push({ layout: run.layout, start_ms: run.start_ms, frames: [...run.frames], resets: [] })
    }
  }

  // Step 4: one ClipSegment per group, with keyframes smoothed between cuts
  return groups.map((g, gi) => {
    const start_ms = gi === 0 ? 0 : g.start_ms
    const end_ms = groups[gi + 1]?.start_ms ?? clipDurationMs
    // The first keyframe sits on the segment start (the renderer extrapolates before it)
    const kfTime = (f: FrameDetection, i: number) => (i === 0 ? start_ms : Math.max(start_ms, frameTime(f)))
    const slotKfs = Array.from({ length: slotCount(g.layout) }, (_, slot) => {
      const cx = stabilise(g.frames.map(f => rawCx(f, g.layout, slot)), g.resets)
      return g.frames.map((f, i) => g.layout === 'vertical' ? vertKf(kfTime(f, i), cx[i]) : splitKf(kfTime(f, i), cx[i]))
    })
    return { start_ms, end_ms, layout: g.layout, slotKfs }
  })
}

type Word = { word: string; start_ms: number; end_ms: number }

// Gemini picks the clips: the numbered-line, windowed, scored method shared with the app's
// "Best moments" (src/lib/clipFinder.ts)
async function selectHighlights(words: Word[], clipCount: number, durationMs: number): Promise<FoundClip[]> {
  const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!)
  const ask = async (system: string, user: string) => {
    const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash', systemInstruction: system })
    return (await model.generateContent(user)).response.text()
  }
  // Enough candidates per window that a short video (one window) can still fill the request
  const windowCount = Math.max(1, Math.ceil(durationMs / (9 * 60_000)))
  const perWindow = Math.min(10, Math.max(4, Math.ceil((clipCount * 1.5) / windowCount)))
  return findClips({
    words, durationMs, mode: { kind: 'best' }, ask,
    limit: clipCount, perWindow, spread: true,
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
  return db<Word[]>`SELECT word, start_ms, end_ms FROM transcript_words WHERE transcript_id = ${row.id} ORDER BY start_ms`
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
  const t0 = Date.now()

  await db`UPDATE ai_edit_jobs SET status = 'running', progress = 5 WHERE id = ${ai_edit_job_id}`

  try {
    // 1. Get video info (and the owner's plan, for the export watermark)
    const [video] = await db`
      SELECT v.id, v.storage_path, v.duration_ms, u.plan
      FROM videos v LEFT JOIN users u ON u.id = v.user_id
      WHERE v.id = ${video_id}
    `
    if (!video?.storage_path) throw new Error('Video not found or missing storage path')
    // Same rule as the app's plans (src/lib/plans.ts): only the free plan exports with a watermark
    const watermark = !['starter', 'creator', 'agency'].includes(video.plan ?? 'free')

    // 2. Ensure a whole-video transcript — transcribe inline if there isn't one.
    //    Always from storage (R2), never yt-dlp, so this also runs on Railway.
    await waitForFullTranscription(video_id, signal)
    let words = await loadWords(video_id)
    const knownDurationMs = video.duration_ms ?? (words.length ? words[words.length - 1].end_ms : 0)
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

    // 3. Pick the clips across the whole video
    const highlights = await selectHighlights(words, clip_count, durationMs)
    if (highlights.length === 0) throw new Error('AI could not find any good clips in this video')
    console.log(`[ai_edit] ${highlights.length}/${clip_count} clips picked for job ${ai_edit_job_id} (${((Date.now() - t0) / 1000).toFixed(0)}s)`)
    await setProgress(ai_edit_job_id, 35)

    // 4. Download video once (for framing)
    const tmp = await mkdtemp(join(tmpdir(), 'ai-edit-'))
    try {
      const videoPath = join(tmp, 'source.mp4')
      console.log(`[ai_edit] Downloading video ${video.storage_path}`)
      await r2DownloadToFile(video.storage_path, videoPath, signal)
      const size = await probeSize(videoPath)
      await setProgress(ai_edit_job_id, 45)

      // 5. Frame each clip, save it, and queue its render (the render queue exports it, so this
      //    job stays well inside the job timeout however many clips there are)
      for (let hi = 0; hi < highlights.length; hi++) {
        signal?.throwIfAborted()
        const highlight = highlights[hi]
        const clipDurationMs = highlight.end_ms - highlight.start_ms

        // Faces 4× a second and camera cuts across the entire clip
        const tDetect = Date.now()
        const analysis = await analyseClip(tmp, videoPath, highlight.start_ms, highlight.end_ms, size, signal)
        const detectS = ((Date.now() - tDetect) / 1000).toFixed(1)

        // Brain: dynamically switch layout within the clip
        const segments = buildSegments(analysis, clipDurationMs)
        const clipWords = words.filter(w => w.start_ms >= highlight.start_ms && w.start_ms < highlight.end_ms)
        const caption = captionFontFor(clipWords)

        const clip_id = await saveClip(video_id, ai_edit_job_id, highlight, segments, caption)
        const renderPayload = { clip_id, video_storage_path: video.storage_path, quality: '1080p', watermark }
        await db`INSERT INTO jobs (type, payload, status) VALUES ('render', ${db.json(renderPayload)}, 'queued')`

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
    console.log(`[ai_edit] Job ${ai_edit_job_id} done in ${((Date.now() - t0) / 1000).toFixed(0)}s (renders queued)`)
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    console.error(`[ai_edit] Job ${ai_edit_job_id} failed:`, msg)
    await db`UPDATE ai_edit_jobs SET status = 'failed', error = ${msg} WHERE id = ${ai_edit_job_id}`
    throw err
  }
}

/**
 * One clip with its formats, crop boxes, keyframes and caption style, in one transaction.
 * Rows go in as bulk inserts: the database is several hundred ms away, and a round trip per
 * keyframe took minutes per clip.
 */
async function saveClip(
  videoId: string, aiEditJobId: string, highlight: FoundClip, segments: ClipSegment[],
  caption: { font: string; language: string | null },
): Promise<string> {
  return db.begin(async tx => {
    const [clipRow] = await tx`
      INSERT INTO clips (video_id, start_ms, end_ms, status, title, ai_edit_job_id, ai_score, ai_reason)
      VALUES (${videoId}, ${highlight.start_ms}, ${highlight.end_ms}, 'rendering', ${highlight.title.slice(0, 120) || 'Highlight'},
        ${aiEditJobId}, ${highlight.score}, ${highlight.reason || null})
      RETURNING id
    `
    const clipId = clipRow.id as string

    const segRows = await tx`
      INSERT INTO segments ${tx(segments.map((seg, si) => ({
        clip_id: clipId, start_ms: seg.start_ms, end_ms: seg.end_ms, layout: seg.layout, sort_order: si, video_offset_ms: null,
      })))}
      RETURNING id, sort_order
    `
    const segIdByOrder = new Map(segRows.map(r => [r.sort_order as number, r.id as string]))

    // source_offset_ms is where in the clip this segment's video starts: render.py trims
    // the main video from it, so 0 made every later segment replay the clip's start
    const boxInputs = segments.flatMap((seg, si) => seg.slotKfs.map((kfs, slotIdx) => ({
      row: { segment_id: segIdByOrder.get(si)!, slot_index: slotIdx, source_video_id: null, source_offset_ms: seg.start_ms },
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
