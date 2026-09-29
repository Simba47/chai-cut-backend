import { spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import { writeFile, mkdtemp, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { GetObjectCommand } from '@aws-sdk/client-s3'
import { getSignedUrl } from '@aws-sdk/s3-request-presigner'
import { r2, R2_BUCKET, r2DownloadToFile, r2UploadFile } from '../r2.js'
import db from '../db.js'
import type { Job, RenderJobPayload } from '../types.js'
import { GoogleGenerativeAI } from '@google/generative-ai'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { computeCutRanges, keptRanges, makeTimeMap, removedMs, type Range } from '../lib/cuts.js'

const execFileAsync = promisify(execFile)
import { fileURLToPath } from 'node:url'
import { dirname } from 'node:path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

export async function handleRenderJob(job: Job, signal?: AbortSignal) {
  const payload = job.payload as unknown as RenderJobPayload
  const { clip_id, video_storage_path } = payload
  // The quality the export asked for (it used to be ignored: every render came out 1080p)
  const quality = payload.quality && QUALITY_DIMS[payload.quality] ? payload.quality : '1080p'

  const baseSpec = await buildRenderSpec(clip_id, video_storage_path, quality)
  // Other videos shown in the clip are fetched by ID: only ever the clip owner's own videos
  const [owner] = await db`SELECT v.user_id FROM clips c JOIN videos v ON v.id = c.video_id WHERE c.id = ${clip_id}`
  // Sign the source URL so FFmpeg can stream directly from R2 via HTTP range requests.
  // This avoids downloading the full source file (can be 1GB+) to Railway's disk
  // when we only need a 60-second window of it.
  const videoSignedUrl = await getSignedUrl(
    r2, new GetObjectCommand({ Bucket: R2_BUCKET, Key: video_storage_path }), { expiresIn: 7200 }
  )
  const tmp = await mkdtemp(join(tmpdir(), 'render-'))
  const outputPath = join(tmp, 'output.mp4')
  const specPath = join(tmp, 'spec.json')

  try {
    const t0 = Date.now()
    // Remove pauses and filler words: render from a copy of the clip with those parts cut out,
    // with every time in the spec moved to match. render.py and audio.py need no changes.
    let videoInput = videoSignedUrl
    let renderSpec = baseSpec
    if (baseSpec.remove_fillers) {
      const cut = await condenseClip(clip_id, baseSpec, videoSignedUrl, tmp, signal)
      if (cut) { videoInput = cut.path; renderSpec = cut.spec }
    }
    type SpecSeg = {
      crop_boxes?: Array<{ source_video_id?: string | null; image_path?: string | null }>
      frame?: { items?: Array<{ kind?: string; source_video_id?: string | null; image_path?: string | null }> } | null
    }
    const specSegs = (renderSpec.segments ?? []) as SpecSeg[]
    // Other videos: B-roll formats, and videos placed on frame lanes
    const otherVideoIds = new Set<string>()
    for (const seg of specSegs) {
      for (const box of seg.crop_boxes ?? []) if (box.source_video_id) otherVideoIds.add(box.source_video_id)
      for (const it of seg.frame?.items ?? []) if (it.kind === 'video' && it.source_video_id) otherVideoIds.add(it.source_video_id)
    }
    const secondaryVideos: Record<string, string> = {}
    for (const videoId of otherVideoIds) {
      const [vRow] = await db`SELECT storage_path FROM videos WHERE id = ${videoId} AND user_id = ${owner?.user_id ?? null}`
      if (!vRow?.storage_path) continue
      try {
        const t1 = Date.now()
        const localPath = join(tmp, `secondary_${videoId}.mp4`)
        await r2DownloadToFile(vRow.storage_path, localPath, signal)
        console.log(`[render] B-roll download: ${((Date.now()-t1)/1000).toFixed(1)}s`)
        secondaryVideos[videoId] = localPath
      } catch (e) { console.warn(`[render] Failed to download secondary video:`, e) }
    }
    console.log(`[render] Downloads done: ${((Date.now()-t0)/1000).toFixed(1)}s`)

    const overlayImages: Record<string, string> = {}
    for (const ov of (renderSpec.overlays ?? []) as Array<{ type?: string; storage_path?: string }>) {
      if (ov.type !== 'image' || !ov.storage_path) continue
      try {
        const ext = ov.storage_path.split('.').pop() ?? 'png'
        const localPath = join(tmp, `overlay_${createHash('sha1').update(ov.storage_path).digest('hex').slice(0, 16)}.${ext}`)
        await r2DownloadToFile(ov.storage_path, localPath, signal)
        overlayImages[ov.storage_path] = localPath
      } catch (e) { console.warn(`[render] Failed to download overlay:`, e) }
    }

    // Photos on frame lanes (and on frame slots saved before lanes existed)
    const framePhotoPaths = new Set<string>()
    for (const seg of specSegs) {
      for (const box of seg.crop_boxes ?? []) if (box.image_path) framePhotoPaths.add(box.image_path)
      for (const it of seg.frame?.items ?? []) if (it.kind === 'photo' && it.image_path) framePhotoPaths.add(it.image_path)
    }
    const frameImages: Record<string, string> = {}
    for (const path of framePhotoPaths) {
      try {
        const ext = path.split('.').pop() ?? 'png'
        const localPath = join(tmp, `frame_${createHash('sha1').update(path).digest('hex').slice(0, 16)}.${ext}`)
        await r2DownloadToFile(path, localPath, signal)
        frameImages[path] = localPath
      } catch (e) { console.warn(`[render] Failed to download frame photo:`, e) }
    }

    const overlayVideos: Record<string, string> = {}
    const seenOverlayVideoIds = new Set<string>()
    for (const ov of (renderSpec.overlays ?? []) as Array<{ type?: string; source_video_id?: string }>) {
      if (ov.type !== 'video' || !ov.source_video_id || seenOverlayVideoIds.has(ov.source_video_id)) continue
      seenOverlayVideoIds.add(ov.source_video_id)
      const [vRow] = await db`SELECT storage_path FROM videos WHERE id = ${ov.source_video_id} AND user_id = ${owner?.user_id ?? null}`
      if (!vRow?.storage_path) continue
      try {
        const localPath = join(tmp, `overlay_video_${ov.source_video_id}.mp4`)
        await r2DownloadToFile(vRow.storage_path, localPath, signal)
        overlayVideos[ov.source_video_id] = localPath
      } catch (e) { console.warn(`[render] Failed to download overlay video:`, e) }
    }

    await writeFile(specPath, JSON.stringify(renderSpec, null, 2))
    // __dirname is dist/jobs/ at runtime; Python files live in src/python/ (not copied by tsc)
    const pythonScript = join(__dirname, '../../src/python/render.py')
    const pythonArgs = [
      '--video', videoInput, '--spec', specPath, '--output', outputPath,
      '--secondary-videos', JSON.stringify(secondaryVideos),
      '--overlay-images', JSON.stringify(overlayImages),
      '--overlay-videos', JSON.stringify(overlayVideos),
      '--frame-images', JSON.stringify(frameImages),
    ]
    if (payload.watermark) pythonArgs.push('--watermark')
    const tPy = Date.now()
    await runPython(pythonScript, pythonArgs, signal)
    console.log(`[render] Python done: ${((Date.now()-tPy)/1000).toFixed(1)}s`)

    const outputStoragePath = `clips/${clip_id}/output.mp4`
    await r2UploadFile(outputStoragePath, outputPath, 'video/mp4')

    const output_url = await getSignedUrl(r2, new GetObjectCommand({ Bucket: R2_BUCKET, Key: outputStoragePath }), { expiresIn: 60 * 60 * 24 * 7 })
    await db`UPDATE clips SET status = 'done', output_url = ${output_url}, output_storage_path = ${outputStoragePath} WHERE id = ${clip_id}`

  } catch (err) {
    await db`UPDATE clips SET status = 'failed' WHERE id = ${clip_id}`
    throw err
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
}

const QUALITY_DIMS: Record<string, { w: number; h: number }> = {
  '480p': { w: 480, h: 854 }, '720p': { w: 720, h: 1280 },
  '1080p': { w: 1080, h: 1920 }, '2160p': { w: 2160, h: 3840 },
}

async function buildRenderSpec(clipId: string, videoStoragePath: string, quality = '2160p') {
  const [clip] = await db`SELECT * FROM clips WHERE id = ${clipId}`

  const segments = await db`
    SELECT s.*,
      COALESCE((
        SELECT json_agg(jsonb_build_object(
          'id', cb.id, 'segment_id', cb.segment_id, 'slot_index', cb.slot_index,
          'source_video_id', cb.source_video_id, 'source_offset_ms', cb.source_offset_ms,
          'image_path', cb.image_path, 'image_motion', cb.image_motion,
          'volume', cb.volume, 'muted', cb.muted,
          'box_keyframes', COALESCE((
            SELECT json_agg(bk.* ORDER BY bk.t_ms) FROM box_keyframes bk WHERE bk.box_id = cb.id
          ), '[]')
        ) ORDER BY cb.slot_index)
        FROM crop_boxes cb WHERE cb.segment_id = s.id
      ), '[]') AS crop_boxes
    FROM segments s WHERE s.clip_id = ${clipId} ORDER BY s.start_ms, s.sort_order
  `

  const [captionStyles, textOverlays, audioTracks, transitions, overlays] = await Promise.all([
    db`SELECT * FROM caption_styles WHERE clip_id = ${clipId}`,
    db`SELECT * FROM text_overlays WHERE clip_id = ${clipId}`,
    db`SELECT * FROM audio_tracks WHERE clip_id = ${clipId}`,
    db`SELECT * FROM transitions WHERE clip_id = ${clipId}`,
    db`SELECT * FROM overlays WHERE clip_id = ${clipId} ORDER BY z_index`,
  ])

  const [transcriptRow] = await db`
    SELECT id FROM transcripts WHERE video_id = ${clip.video_id} ORDER BY created_at DESC LIMIT 1
  `
  const words = transcriptRow
    ? await db`SELECT word, word_roman, start_ms, end_ms, speaker_id FROM transcript_words WHERE transcript_id = ${transcriptRow.id} ORDER BY start_ms`
    : []

  // Animated presets emphasise the words that matter; asked once per clip and kept
  const style = captionStyles[0]
  if (style && PRESETS.has(style.animation) && style.emphasis == null && words.length) {
    const clipWords = words.filter(w => w.start_ms >= clip.start_ms && w.start_ms < clip.end_ms)
    const emphasis = await pickEmphasis(clipWords.map(w => ({ word: w.word as string, start_ms: w.start_ms as number, end_ms: w.end_ms as number })))
    if (emphasis) {
      await db`UPDATE caption_styles SET emphasis = ${db.json(emphasis)} WHERE id = ${style.id}`
        .catch(e => console.warn('[render] could not save caption emphasis:', e))
      style.emphasis = emphasis
    }
  }

  const dims = QUALITY_DIMS[quality] ?? QUALITY_DIMS['1080p']
  return {
    clip_id: clipId,
    start_ms: clip.start_ms,
    end_ms: clip.end_ms,
    remove_fillers: clip.remove_fillers === true,
    segments, caption_styles: captionStyles, text_overlays: textOverlays,
    audio_tracks: audioTracks, transitions, overlays, words,
    output_width: dims.w, output_height: dims.h,
  }
}

// `signal` (the job timing out) kills Python together with the ffmpeg it started: on Linux Python
// runs in its own process group, and the whole group is killed (killing only Python would leave
// ffmpeg running on)
function runPython(script: string, args: string[], signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(signal.reason); return }
    const group = process.platform !== 'win32'
    const proc = spawn('python3', [script, ...args], { stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' }, detached: group })
    const kill = () => {
      try { if (group && proc.pid) process.kill(-proc.pid, 'SIGKILL'); else proc.kill('SIGKILL') } catch { /* already gone */ }
    }
    signal?.addEventListener('abort', kill, { once: true })
    proc.on('close', () => signal?.removeEventListener('abort', kill))
    let stderr = ''
    proc.stdout?.on('data', (d: Buffer) => process.stdout.write(d))
    proc.stderr?.on('data', (d: Buffer) => { process.stderr.write(d); stderr += d.toString() })
    proc.on('close', code => { if (code === 0) resolve(); else reject(new Error(`Python render failed (exit ${code}): ${stderr.slice(-500)}`)) })
    proc.on('error', reject)
  })
}

const PRESETS = new Set(['pop', 'highlight', 'bounce', 'word'])

/**
 * The 1–2 most important words in each caption line, as { "<word start_ms>": true } (the key
 * render.py and the editor look words up by). Gemini reads the clip's lines once; null when it
 * fails, so the clip renders without emphasis.
 */
export async function pickEmphasis(words: Array<{ word: string; start_ms: number; end_ms: number }>): Promise<Record<string, true> | null> {
  if (!words.length || !process.env.GEMINI_API_KEY) return null
  // Lines as the captions break them: a pause over 300 ms or 5 words
  const lines: number[][] = []
  words.forEach((w, i) => {
    const line = lines[lines.length - 1]
    if (!line || line.length >= 5 || w.start_ms - words[i - 1].end_ms > 300) lines.push([i])
    else line.push(i)
  })
  const text = lines.map((l, li) => `${li}: ` + l.map(i => `${i}=${words[i].word}`).join(' ')).join('\n')
  const prompt = `These are caption lines from a short video (any language, often Telugu, Hindi or Tamil mixed with English). Each word is written as index=word.
For each line pick the 1 or 2 words that carry the meaning or emotion (a key noun, number, strong verb or punchline word) — never filler words. Skip a line if nothing stands out.
Return ONLY JSON: {"words": [index, ...]}

${text}`
  try {
    const model = new GoogleGenerativeAI(process.env.GEMINI_API_KEY).getGenerativeModel({ model: 'gemini-2.5-flash' })
    const out = (await model.generateContent(prompt)).response.text()
    const match = out.match(/\{[\s\S]*\}/)
    const idx: unknown = match ? JSON.parse(match[0]).words : null
    if (!Array.isArray(idx)) return null
    const picked: Record<string, true> = {}
    for (const i of idx) if (Number.isInteger(i) && words[i]) picked[String(words[i].start_ms)] = true
    return picked
  } catch (e) {
    console.warn('[render] caption emphasis failed, rendering without it:', e instanceof Error ? e.message : e)
    return null
  }
}

// ── Remove pauses and filler words ──────────────────────────────────────────────

type Spec = Awaited<ReturnType<typeof buildRenderSpec>>
type Row = Record<string, any> // eslint-disable-line @typescript-eslint/no-explicit-any

async function probeSource(url: string, signal?: AbortSignal) {
  const { stdout } = await execFileAsync('ffprobe', [
    '-v', 'error', '-show_entries', 'stream=codec_type,r_frame_rate', '-of', 'json', url,
  ], { signal })
  const streams = (JSON.parse(stdout).streams ?? []) as Array<{ codec_type: string; r_frame_rate?: string }>
  const [n, d] = (streams.find(st => st.codec_type === 'video')?.r_frame_rate ?? '30/1').split('/').map(Number)
  return { fps: n && d ? n / d : 30, hasAudio: streams.some(st => st.codec_type === 'audio') }
}

/**
 * Cuts the clip's pauses and filler words (src/lib/cuts.ts, from the transcript), saves the
 * ranges on the clip, and writes a shortened copy of the clip: the kept parts joined, audio
 * faded over 15 ms at each join so there are no clicks, encoded like render.py encodes. Kept
 * parts start and end on video frames, so video, audio and captions stay in step to the end.
 * Returns null (render as normal) when nothing is cut.
 */
async function condenseClip(clipId: string, spec: Spec, videoUrl: string, tmp: string, signal?: AbortSignal) {
  const start = spec.start_ms as number, end = spec.end_ms as number
  const cuts = computeCutRanges(spec.words as Row[] as Parameters<typeof computeCutRanges>[0], start, end)
  await db`UPDATE clips SET cut_ranges = ${db.json(cuts)} WHERE id = ${clipId}`
  if (!cuts.length) return null

  const { fps, hasAudio } = await probeSource(videoUrl, signal)
  const frame = 1000 / fps
  const snap = (t: number) => start + Math.round((t - start) / frame) * frame
  const kept = keptRanges(cuts, start, end).map(([a, b]): Range => [snap(a), snap(b)]).filter(([a, b]) => b - a >= frame)
  const t0 = Date.now()

  const fc: string[] = []
  kept.forEach(([a, b], i) => {
    const s = ((a - start) / 1000).toFixed(6), e = ((b - start) / 1000).toFixed(6), d = (b - a) / 1000
    fc.push(`[0:v]trim=start=${s}:end=${e},setpts=PTS-STARTPTS[v${i}]`)
    if (hasAudio) {
      fc.push(`[0:a]atrim=start=${s}:end=${e},asetpts=PTS-STARTPTS,`
        + `afade=t=in:st=0:d=0.015,afade=t=out:st=${Math.max(0, d - 0.015).toFixed(6)}:d=0.015[a${i}]`)
    }
  })
  const pads = kept.map((_, i) => hasAudio ? `[v${i}][a${i}]` : `[v${i}]`).join('')
  fc.push(`${pads}concat=n=${kept.length}:v=1:a=${hasAudio ? 1 : 0}${hasAudio ? '[v][a]' : '[v]'}`)
  const out = join(tmp, 'condensed.mp4')
  await execFileAsync('ffmpeg', [
    '-y', '-ss', (start / 1000).toFixed(3), '-t', ((end - start) / 1000).toFixed(3), '-i', videoUrl,
    '-filter_complex', fc.join(';'),
    '-map', '[v]', ...(hasAudio ? ['-map', '[a]'] : []),
    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
    ...(hasAudio ? ['-c:a', 'aac', '-b:a', '192k'] : []), '-movflags', '+faststart', out,
  ], { signal, maxBuffer: 32 * 1024 * 1024 })
  console.log(`[render] Removed ${cuts.length} pause/filler cut(s), ${(removedMs(cuts) / 1000).toFixed(1)}s, in ${((Date.now() - t0) / 1000).toFixed(1)}s`)
  return { path: out, spec: remapSpec(spec, kept) }
}

/**
 * The spec on the shortened clip's timeline: it starts at 0, and every clip-relative time
 * (segments, crop keyframes, text, overlays, frame lane items) and word time moves to where
 * that moment now plays. Cut words are dropped; anything left with no time is dropped.
 */
function remapSpec(spec: Spec, kept: Range[]): Spec {
  const start = spec.start_ms as number
  const src = makeTimeMap(kept)                           // source time → new time
  const rel = (t: number) => Math.round(src(start + Number(t)))  // clip-relative → new time
  const inKept = (t: number) => kept.some(([a, b]) => t >= a && t < b)
  const total = kept.reduce((s, [a, b]) => s + (b - a), 0)

  const words = (spec.words as Row[])
    .filter(w => inKept((w.start_ms + w.end_ms) / 2))
    .map(w => ({ ...w, start_ms: Math.round(src(w.start_ms)), end_ms: Math.round(src(w.end_ms)) }))

  const segments: Row[] = (spec.segments as Row[]).map((seg): Row => ({
    ...seg,
    start_ms: rel(seg.start_ms), end_ms: rel(seg.end_ms),
    video_offset_ms: seg.video_offset_ms == null ? seg.video_offset_ms : rel(seg.video_offset_ms),
    frame: seg.frame?.items ? { ...seg.frame, items: (seg.frame.items as Row[]).map(it => ({ ...it, start_ms: rel(it.start_ms), end_ms: rel(it.end_ms) })) } : seg.frame,
    crop_boxes: (seg.crop_boxes as Row[]).map(b => ({
      ...b,
      // Main-video boxes point into the clip; other videos keep their own offsets
      source_offset_ms: b.source_offset_ms != null && !b.source_video_id && !b.image_path ? rel(b.source_offset_ms) : b.source_offset_ms,
      box_keyframes: (b.box_keyframes as Row[]).map(k => ({ ...k, t_ms: rel(k.t_ms) })),
    })),
  })).filter(seg => seg.end_ms > seg.start_ms)
  const segIds = new Set(segments.map(seg => seg.id))
  const timed = (rows: Row[]) => rows.map(r => ({ ...r, start_ms: rel(r.start_ms), end_ms: rel(r.end_ms) })).filter(r => r.end_ms > r.start_ms)

  return {
    ...spec,
    start_ms: 0,
    end_ms: Math.round(total),
    words,
    segments,
    text_overlays: timed(spec.text_overlays as Row[]),
    overlays: timed(spec.overlays as Row[]),
    transitions: (spec.transitions as Row[]).filter(t => segIds.has(t.after_segment_id)),
  } as unknown as Spec
}
