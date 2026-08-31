import { GoogleGenerativeAI } from '@google/generative-ai'
import { spawn, execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { writeFile, mkdir, mkdtemp, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { GetObjectCommand } from '@aws-sdk/client-s3'
import { r2, R2_BUCKET } from '../r2.js'
import db from '../db.js'
import type { Job, AiEditJobPayload } from '../types.js'
import { handleTranscribeJob } from './transcribe.js'
import { renderClipWithLocalVideo } from './render.js'
import { fileURLToPath } from 'node:url'
import { dirname } from 'node:path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const execFileAsync = promisify(execFile)

// 9:16 crop width as a fraction of a 16:9 source (full height)
const REEL_W = 81 / 256  // ≈ 0.316

interface CropBox { x: number; y: number; w: number; h: number }
interface Highlight { start_ms: number; end_ms: number; title: string }
interface FrameDetection { frame_index: number; person_count: number; faces: CropBox[] }
interface FaceInfo { face_count: number; face_boxes: CropBox[]; frames: FrameDetection[] }
interface SlotKf { t_ms: number; x: number; y: number; w: number; h: number }
interface ClipSegment { start_ms: number; end_ms: number; layout: 'vertical' | 'split'; slotKfs: SlotKf[][] }

async function r2Download(key: string): Promise<Buffer> {
  const res = await r2.send(new GetObjectCommand({ Bucket: R2_BUCKET, Key: key }))
  const chunks: Buffer[] = []
  for await (const chunk of res.Body as AsyncIterable<Uint8Array>) chunks.push(Buffer.from(chunk))
  return Buffer.concat(chunks)
}

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
    const proc = spawn('python3', [script, '--frames-dir', framesDir], { stdio: ['ignore', 'pipe', 'pipe'] })
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

async function selectHighlights(
  words: Array<{ word: string; start_ms: number; end_ms: number }>,
  clipCount: number,
  durationMs: number,
): Promise<Highlight[]> {
  const targetMs = 60_000  // ~60s per clip

  // FIX 1: Sample evenly across the FULL video (not just the first 8000 chars)
  // 500 words × ~15 chars each ≈ 7500 chars, stays under token limit
  const MAX_SAMPLE = 500
  const step = Math.max(1, Math.floor(words.length / MAX_SAMPLE))
  const transcript = words
    .filter((_, i) => i % step === 0)
    .map(w => `${w.word}[${w.start_ms}]`)
    .join(' ')

  const prompt = `You are a professional video editor selecting the best highlight clips to post as short-form vertical content. Your job is to find the moments that will make viewers stop scrolling.

Transcript format: word[timestamp_ms] ... (sampled evenly across the full video)
${transcript}

Video duration: ${Math.round(durationMs / 1000)}s
Number of clips to select: ${clipCount}

WHAT MAKES A GREAT HIGHLIGHT (applies to any video type):
- A strong opinion, surprising statement, or counterintuitive insight
- A peak emotional moment: genuine laughter, shock, excitement, tension, vulnerability
- A self-contained story or anecdote with a clear beginning and payoff
- A quotable, memorable line followed by context that makes it land
- A moment of conflict, disagreement, or strong reaction
- A practical insight or advice that stands on its own
- Avoid: slow intros, sponsor reads, filler ("um", "so", "anyway"), pure logistics

HOW TO CHOOSE START AND END:
- Start just before the key moment begins (include the setup)
- End after the reaction or punchline lands — don't cut mid-thought
- Each clip must feel complete and make sense without watching the rest of the video

Return ONLY a JSON array with exactly ${clipCount} objects. No other text.
Each object: { "start_ms": <integer>, "end_ms": <integer>, "title": "<5 words max>" }

CRITICAL RULES:
- end_ms - start_ms MUST be between 55000 and 70000 (55 to 70 seconds). Never shorter, never longer.
- Clips must not overlap
- Spread clips across the full video — do not cluster them all at the start
- start_ms and end_ms must be actual timestamps from the transcript
- Return exactly ${clipCount} clips`

  const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!)
  const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' })
  const result = await model.generateContent(prompt)
  const text = result.response.text()

  const jsonMatch = text.match(/\[[\s\S]*\]/)
  if (!jsonMatch) throw new Error(`Gemini did not return a JSON array: ${text.slice(0, 300)}`)

  const parsed = JSON.parse(jsonMatch[0]) as Highlight[]
  if (!Array.isArray(parsed) || parsed.length === 0) {
    throw new Error('Gemini returned empty or invalid highlights array')
  }

  return parsed
    .filter(h => typeof h.start_ms === 'number' && typeof h.end_ms === 'number')
    .map(h => {
      const start_ms = Math.max(0, Math.min(durationMs, Math.round(h.start_ms)))
      let end_ms = Math.max(0, Math.min(durationMs, Math.round(h.end_ms)))
      const dur = end_ms - start_ms
      // Hard-enforce minimum length — Gemini sometimes ignores the prompt constraint
      if (dur < 55000) {
        end_ms = Math.min(durationMs, start_ms + 60000)
      }
      return { start_ms, end_ms, title: String(h.title ?? '').slice(0, 80) || 'Highlight' }
    })
    .filter(h => h.end_ms - h.start_ms >= 30000)  // discard anything still too short
    .slice(0, clipCount)
}

export async function handleAiEditJob(job: Job) {
  const payload = job.payload as unknown as AiEditJobPayload
  const { ai_edit_job_id, video_id, clip_count } = payload

  await db`UPDATE ai_edit_jobs SET status = 'running' WHERE id = ${ai_edit_job_id}`

  try {
    // 1. Get video info
    const [video] = await db`SELECT id, storage_path, duration_ms FROM videos WHERE id = ${video_id}`
    if (!video?.storage_path) throw new Error('Video not found or missing storage path')

    // 2. Ensure transcript exists — transcribe inline if not yet done
    const [existingTranscript] = await db`SELECT id FROM transcripts WHERE video_id = ${video_id} LIMIT 1`
    if (!existingTranscript) {
      console.log(`[ai_edit] No transcript for ${video_id} — transcribing now`)
      await handleTranscribeJob({
        id: `ai-edit-tx-${ai_edit_job_id}`,
        type: 'transcribe',
        payload: { video_id, storage_path: video.storage_path, is_retranscribe: true } as Record<string, unknown>,
        status: 'processing',
        error: null,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      })
    }

    // 3. Load transcript words
    const [transcriptRow] = await db`
      SELECT id FROM transcripts WHERE video_id = ${video_id} ORDER BY created_at DESC LIMIT 1
    `
    if (!transcriptRow) throw new Error('No transcript available')

    const words = await db<{ word: string; start_ms: number; end_ms: number }[]>`
      SELECT word, start_ms, end_ms FROM transcript_words
      WHERE transcript_id = ${transcriptRow.id}
      ORDER BY start_ms
    `
    if (!words.length) throw new Error('Transcript is empty')

    const durationMs = video.duration_ms ?? words[words.length - 1].end_ms

    // 4. Ask Gemini to select highlights across the full video
    const highlights = await selectHighlights(words, clip_count, durationMs)
    console.log(`[ai_edit] ${highlights.length} highlights selected for job ${ai_edit_job_id}`)

    // 5. Download video once
    const tmp = await mkdtemp(join(tmpdir(), 'ai-edit-'))
    try {
      const videoPath = join(tmp, 'source.mp4')
      console.log(`[ai_edit] Downloading video ${video.storage_path}`)
      await writeFile(videoPath, await r2Download(video.storage_path))

      // 6. Face detection + create DB records for all clips
      const createdClipIds: string[] = []
      for (const highlight of highlights) {
        const clipDurationMs = highlight.end_ms - highlight.start_ms

        // Detect per-second subject positions across the entire clip
        const faceInfo = await detectClip(tmp, videoPath, highlight.start_ms, highlight.end_ms)

        // Brain: dynamically switch layout per-second within the clip
        const segments = buildSegments(faceInfo, clipDurationMs)

        const [clipRow] = await db`
          INSERT INTO clips (video_id, start_ms, end_ms, status, title, ai_edit_job_id)
          VALUES (${video_id}, ${highlight.start_ms}, ${highlight.end_ms}, 'rendering', ${highlight.title}, ${ai_edit_job_id})
          RETURNING id
        `
        const clip_id = clipRow.id

        for (let si = 0; si < segments.length; si++) {
          const seg = segments[si]
          const [segRow] = await db`
            INSERT INTO segments (clip_id, start_ms, end_ms, layout, sort_order, video_offset_ms)
            VALUES (${clip_id}, ${seg.start_ms}, ${seg.end_ms}, ${seg.layout}, ${si}, NULL)
            RETURNING id
          `
          const seg_id = segRow.id

          for (let slotIdx = 0; slotIdx < seg.slotKfs.length; slotIdx++) {
            const [boxRow] = await db`
              INSERT INTO crop_boxes (segment_id, slot_index, source_video_id, source_offset_ms)
              VALUES (${seg_id}, ${slotIdx}, NULL, 0)
              RETURNING id
            `
            for (const kf of seg.slotKfs[slotIdx]) {
              await db`
                INSERT INTO box_keyframes (box_id, t_ms, x, y, w, h)
                VALUES (${boxRow.id}, ${kf.t_ms}, ${kf.x}, ${kf.y}, ${kf.w}, ${kf.h})
              `
            }
          }
        }

        const layoutSummary = segments.map(s =>
          `${s.layout}(${s.start_ms / 1000}s–${s.end_ms / 1000}s,${s.slotKfs[0]?.length ?? 0}kf)`
        ).join(' | ')
        console.log(`[ai_edit] Clip ${clip_id}: ${segments.length} seg — ${layoutSummary}`)
        console.log(`[ai_edit] Detection summary: ${faceInfo.frames.length} frames, max_persons=${faceInfo.face_count}`)
        createdClipIds.push(clip_id)
      }

      // 7. Render all clips inline — video already on disk, no re-download
      const RENDER_PARALLEL = 3
      for (let i = 0; i < createdClipIds.length; i += RENDER_PARALLEL) {
        const batch = createdClipIds.slice(i, i + RENDER_PARALLEL)
        await Promise.all(batch.map(id => renderClipWithLocalVideo(id, videoPath, video.storage_path)))
        console.log(`[ai_edit] Rendered batch ${Math.floor(i / RENDER_PARALLEL) + 1}/${Math.ceil(createdClipIds.length / RENDER_PARALLEL)}`)
      }
    } finally {
      await rm(tmp, { recursive: true, force: true })
    }

    await db`UPDATE ai_edit_jobs SET status = 'done' WHERE id = ${ai_edit_job_id}`
    console.log(`[ai_edit] Job ${ai_edit_job_id} done`)
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    console.error(`[ai_edit] Job ${ai_edit_job_id} failed:`, msg)
    await db`UPDATE ai_edit_jobs SET status = 'failed', error = ${msg} WHERE id = ${ai_edit_job_id}`
    throw err
  }
}
