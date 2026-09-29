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

// 9:16 crop width as a fraction of a 16:9 source (full height)
const REEL_W = 81 / 256  // ≈ 0.316

interface CropBox { x: number; y: number; w: number; h: number }
interface FrameDetection { frame_index: number; person_count: number; faces: CropBox[] }
interface FaceInfo { face_count: number; face_boxes: CropBox[]; frames: FrameDetection[] }
interface SlotKf { t_ms: number; x: number; y: number; w: number; h: number }
interface ClipSegment { start_ms: number; end_ms: number; layout: 'vertical' | 'split'; slotKfs: SlotKf[][] }

async function extractFrames(videoPath: string, startMs: number, endMs: number, outDir: string): Promise<void> {
  const startS = startMs / 1000
  const durationS = (endMs - startMs) / 1000
  await execFileAsync('ffmpeg', [
    '-ss', String(startS),
    '-t', String(Math.min(durationS, 80)),
    '-i', videoPath,
    '-vf', 'fps=1,scale=640:-1',
    '-q:v', '3',
    '-frames:v', '80',
    join(outDir, 'frame_%04d.jpg'),
  ])
}

async function detectFaces(tmp: string, videoPath: string, startMs: number, endMs: number): Promise<FaceInfo> {
  const framesDir = join(tmp, `frames-${startMs}`)
  await mkdir(framesDir, { recursive: true })
  try {
    await extractFrames(videoPath, startMs, endMs, framesDir)
  } catch {
    return { face_count: 0, face_boxes: [], frames: [] }
  }

  return new Promise((resolve) => {
    const script = join(__dirname, '../../src/python/face_detect.py')
    const proc = spawn('python3', [script, '--frames-dir', framesDir], { stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' } })
    let stdout = ''
    let stderr = ''
    proc.stdout?.on('data', (d: Buffer) => { stdout += d.toString() })
    proc.stderr?.on('data', (d: Buffer) => { stderr += d.toString() })
    proc.on('close', code => {
      if (stderr) console.warn('[ai_edit] face_detect stderr:', stderr.slice(-300))
      if (code !== 0 || !stdout.trim()) { resolve({ face_count: 0, face_boxes: [], frames: [] }); return }
      try { resolve(JSON.parse(stdout) as FaceInfo) } catch { resolve({ face_count: 0, face_boxes: [], frames: [] }) }
    })
    proc.on('error', () => resolve({ face_count: 0, face_boxes: [], frames: [] }))
  })
}

// Run one detection pass for the full clip — face_detect.py returns per-frame positions
// so we get one keyframe every 4 seconds for smooth subject tracking
async function detectClip(tmp: string, videoPath: string, startMs: number, endMs: number): Promise<FaceInfo> {
  return detectFaces(tmp, videoPath, startMs, endMs)
}

// FRAME_INTERVAL_MS matches the ffmpeg extraction rate: fps=1 → 1 frame per second
const FRAME_INTERVAL_MS = 1000

// Minimum segment duration before we'll commit to a layout switch.
// Avoids rapid flickering when a person briefly walks on/off screen.
const MIN_SEGMENT_MS = 3000

// ── Crop stabilisation helpers ────────────────────────────────────────────────
//
// Per-frame subject detection has natural jitter: the detected center shifts a
// few percent each frame even when nobody moved. Two passes fix this:
//
//   1. EMA (α=0.3): blends toward the new position gradually.
//      A large sudden move takes ~3 frames to fully land — no snap cuts.
//
//   2. Dead zone (3% of frame width): once smoothed, suppress any remaining
//      movement smaller than the threshold. The crop stays completely still
//      when the subject hasn't really moved; only genuine repositioning moves it.
//
// Together: stationary subject → locked crop. Moving subject → smooth follow.

function _ema(values: number[], alpha = 0.3): number[] {
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

// Smooth raw detected center positions before converting to keyframes.
function stabilise(rawCx: number[]): number[] {
  return _deadZone(_ema(rawCx))
}

// ── Layout brain ──────────────────────────────────────────────────────────────
//
// Analyses per-second frame detections and produces one or more ClipSegments,
// each with the right layout (vertical / split) and smooth motion keyframes.
//
// Rules — generic, apply to any video type:
//   - 1 person visible   → vertical, track that person
//   - 2+ people visible  → split, tight on the 2 most prominent (Option B)
//   - Layout switches only happen when a run of frames holds the same layout
//     for at least MIN_SEGMENT_MS (avoids single-frame noise causing a cut)
//
// Keyframe t_ms values are clip-relative (0 = first frame of the clip).
// render.py subtracts seg.start_ms per segment to get FFmpeg-relative time.
function buildSegments(info: FaceInfo, clipDurationMs: number): ClipSegment[] {
  const center = (1 - REEL_W) / 2

  const vertKf = (t_ms: number, cx: number): SlotKf => ({
    t_ms, y: 0, w: REEL_W, h: 1,
    x: Math.max(0, Math.min(1 - REEL_W, cx - REEL_W / 2)),
  })
  const splitKf = (t_ms: number, cx: number): SlotKf => {
    const CW = 0.5
    return { t_ms, y: 0, w: CW, h: 1, x: Math.max(0, Math.min(1 - CW, cx - CW / 2)) }
  }

  // Fallback: no frames detected → one centered vertical segment
  if (info.frames.length === 0) {
    return [{
      start_ms: 0, end_ms: clipDurationMs, layout: 'vertical',
      slotKfs: [[{ t_ms: 0, x: center, y: 0, w: REEL_W, h: 1 }]],
    }]
  }

  // Step 1: Per-frame layout decision using face_detect.py's person_count
  type FrameLayout = { frame: FrameDetection; layout: 'vertical' | 'split' }
  const frameLayouts: FrameLayout[] = info.frames.map(f => {
    const count = f.person_count ?? f.faces.length
    const layout = count >= 2 ? 'split' : 'vertical'
    const cxList = f.faces.map(face => (face.x + face.w / 2).toFixed(2)).join(',')
    console.log(`[brain] frame=${String(f.frame_index).padStart(2,'0')} persons=${count} → ${layout} cx=[${cxList}]`)
    return { frame: f, layout }
  })

  // Step 2: Group consecutive same-layout frames into runs
  type Run = { layout: 'vertical' | 'split'; frames: FrameDetection[] }
  const runs: Run[] = []
  for (const { frame, layout } of frameLayouts) {
    const last = runs[runs.length - 1]
    if (last && last.layout === layout) {
      last.frames.push(frame)
    } else {
      runs.push({ layout, frames: [frame] })
    }
  }

  // Step 3: Merge runs shorter than MIN_SEGMENT_MS into their neighbor
  const minFrames = Math.ceil(MIN_SEGMENT_MS / FRAME_INTERVAL_MS)
  let merged = true
  while (merged && runs.length > 1) {
    merged = false
    for (let i = 0; i < runs.length; i++) {
      if (runs[i].frames.length < minFrames) {
        const mergeLeft  = i > 0 ? runs[i - 1].frames.length : -1
        const mergeRight = i < runs.length - 1 ? runs[i + 1].frames.length : -1
        if (mergeLeft >= mergeRight) {
          runs[i - 1].frames = [...runs[i - 1].frames, ...runs[i].frames]
          runs.splice(i, 1)
        } else {
          runs[i + 1].frames = [...runs[i].frames, ...runs[i + 1].frames]
          runs.splice(i, 1)
        }
        merged = true
        break
      }
    }
  }

  // Step 4: Build ClipSegment per run, with stabilised keyframes
  return runs.map(run => {
    const sorted = [...run.frames].sort((a, b) => a.frame_index - b.frame_index)
    const start_ms = sorted[0].frame_index * FRAME_INTERVAL_MS
    const last = sorted[sorted.length - 1]
    const end_ms = Math.min((last.frame_index + 1) * FRAME_INTERVAL_MS, clipDurationMs)

    if (run.layout === 'vertical') {
      // Raw centers: largest detected subject per frame
      const rawCx = sorted.map(f => {
        const best = [...f.faces].sort((a, b) => (b.w * b.h) - (a.w * a.h))[0]
        return best ? best.x + best.w / 2 : 0.5
      })
      const cx = stabilise(rawCx)
      const kfs = sorted.map((f, i) => vertKf(f.frame_index * FRAME_INTERVAL_MS, cx[i]))
      return { start_ms, end_ms, layout: 'vertical' as const, slotKfs: [kfs.length ? kfs : [vertKf(start_ms, 0.5)]] }
    }

    // Split: stabilise each slot independently (Option B: tight on 2 people)
    const slotKfs: SlotKf[][] = [0, 1].map(slotIdx => {
      const rawCx = sorted.map(f => {
        const bySize = [...f.faces].sort((a, b) => (b.w * b.h) - (a.w * a.h))
        const top2   = bySize.slice(0, 2).sort((a, b) => (a.x + a.w / 2) - (b.x + b.w / 2))
        const face   = top2[slotIdx] ?? top2[0]
        return face ? face.x + face.w / 2 : slotIdx === 0 ? 0.3 : 0.7
      })
      const cx = stabilise(rawCx)
      const kfs = sorted.map((f, i) => splitKf(f.frame_index * FRAME_INTERVAL_MS, cx[i]))
      return kfs.length ? kfs : [splitKf(start_ms, slotIdx === 0 ? 0.3 : 0.7)]
    })
    return { start_ms, end_ms, layout: 'split' as const, slotKfs }
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
      await setProgress(ai_edit_job_id, 45)

      // 5. Frame each clip, save it, and queue its render (the render queue exports it, so this
      //    job stays well inside the job timeout however many clips there are)
      for (let hi = 0; hi < highlights.length; hi++) {
        signal?.throwIfAborted()
        const highlight = highlights[hi]
        const clipDurationMs = highlight.end_ms - highlight.start_ms

        // Detect per-second subject positions across the entire clip
        const faceInfo = await detectClip(tmp, videoPath, highlight.start_ms, highlight.end_ms)

        // Brain: dynamically switch layout per-second within the clip
        const segments = buildSegments(faceInfo, clipDurationMs)
        const clipWords = words.filter(w => w.start_ms >= highlight.start_ms && w.start_ms < highlight.end_ms)
        const caption = captionFontFor(clipWords)

        const clip_id = await saveClip(video_id, ai_edit_job_id, highlight, segments, caption)
        const renderPayload = { clip_id, video_storage_path: video.storage_path, quality: '1080p', watermark }
        await db`INSERT INTO jobs (type, payload, status) VALUES ('render', ${db.json(renderPayload)}, 'queued')`

        const layoutSummary = segments.map(s =>
          `${s.layout}(${s.start_ms / 1000}s–${s.end_ms / 1000}s,${s.slotKfs[0]?.length ?? 0}kf)`
        ).join(' | ')
        console.log(`[ai_edit] Clip ${clip_id} (score ${highlight.score}, ${caption.font}${caption.language ? '/' + caption.language : ''}): ${segments.length} seg — ${layoutSummary}`)
        console.log(`[ai_edit] Detection summary: ${faceInfo.frames.length} frames, max_persons=${faceInfo.face_count}`)
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
