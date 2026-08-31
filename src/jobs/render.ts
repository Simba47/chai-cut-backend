import { spawn } from 'node:child_process'
import { createReadStream, createWriteStream } from 'node:fs'
import { writeFile, mkdtemp, rm } from 'node:fs/promises'
import { pipeline } from 'node:stream/promises'
import { Readable } from 'node:stream'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { GetObjectCommand, PutObjectCommand } from '@aws-sdk/client-s3'
import { getSignedUrl } from '@aws-sdk/s3-request-presigner'
import { r2, R2_BUCKET } from '../r2.js'
import db from '../db.js'
import type { Job, RenderJobPayload } from '../types.js'
import { fileURLToPath } from 'node:url'
import { dirname } from 'node:path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

// Stream R2 object directly to a local file — never buffers the whole file in memory.
async function r2DownloadToFile(key: string, filePath: string): Promise<void> {
  const res = await r2.send(new GetObjectCommand({ Bucket: R2_BUCKET, Key: key }))
  await pipeline(Readable.from(res.Body as AsyncIterable<Uint8Array>), createWriteStream(filePath))
}

export async function handleRenderJob(job: Job) {
  const payload = job.payload as unknown as RenderJobPayload
  const { clip_id, video_storage_path } = payload

  const renderSpec = await buildRenderSpec(clip_id, video_storage_path, '1080p')
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
    const secondaryVideos: Record<string, string> = {}
    const seenVideoIds = new Set<string>()
    for (const seg of (renderSpec.segments ?? []) as Array<{ crop_boxes?: Array<{ source_video_id?: string }> }>) {
      for (const box of seg.crop_boxes ?? []) {
        if (!box.source_video_id || seenVideoIds.has(box.source_video_id)) continue
        seenVideoIds.add(box.source_video_id)
        const [vRow] = await db`SELECT storage_path FROM videos WHERE id = ${box.source_video_id}`
        if (!vRow?.storage_path) continue
        try {
          const t1 = Date.now()
          const localPath = join(tmp, `secondary_${box.source_video_id}.mp4`)
          await r2DownloadToFile(vRow.storage_path, localPath)
          console.log(`[render] B-roll download: ${((Date.now()-t1)/1000).toFixed(1)}s`)
          secondaryVideos[box.source_video_id] = localPath
        } catch (e) { console.warn(`[render] Failed to download secondary video:`, e) }
      }
    }
    console.log(`[render] Downloads done: ${((Date.now()-t0)/1000).toFixed(1)}s`)

    const overlayImages: Record<string, string> = {}
    for (const ov of (renderSpec.overlays ?? []) as Array<{ type?: string; storage_path?: string }>) {
      if (ov.type !== 'image' || !ov.storage_path) continue
      try {
        const ext = ov.storage_path.split('.').pop() ?? 'png'
        const localPath = join(tmp, `overlay_${Buffer.from(ov.storage_path).toString('hex').slice(0, 16)}.${ext}`)
        await r2DownloadToFile(ov.storage_path, localPath)
        overlayImages[ov.storage_path] = localPath
      } catch (e) { console.warn(`[render] Failed to download overlay:`, e) }
    }

    const overlayVideos: Record<string, string> = {}
    const seenOverlayVideoIds = new Set<string>()
    for (const ov of (renderSpec.overlays ?? []) as Array<{ type?: string; source_video_id?: string }>) {
      if (ov.type !== 'video' || !ov.source_video_id || seenOverlayVideoIds.has(ov.source_video_id)) continue
      seenOverlayVideoIds.add(ov.source_video_id)
      const [vRow] = await db`SELECT storage_path FROM videos WHERE id = ${ov.source_video_id}`
      if (!vRow?.storage_path) continue
      try {
        const localPath = join(tmp, `overlay_video_${ov.source_video_id}.mp4`)
        await r2DownloadToFile(vRow.storage_path, localPath)
        overlayVideos[ov.source_video_id] = localPath
      } catch (e) { console.warn(`[render] Failed to download overlay video:`, e) }
    }

    await writeFile(specPath, JSON.stringify(renderSpec, null, 2))
    // __dirname is dist/jobs/ at runtime; Python files live in src/python/ (not copied by tsc)
    const pythonScript = join(__dirname, '../../src/python/render.py')
    const pythonArgs = [
      '--video', videoSignedUrl, '--spec', specPath, '--output', outputPath,
      '--secondary-videos', JSON.stringify(secondaryVideos),
      '--overlay-images', JSON.stringify(overlayImages),
      '--overlay-videos', JSON.stringify(overlayVideos),
    ]
    if (payload.watermark) pythonArgs.push('--watermark')
    const tPy = Date.now()
    await runPython(pythonScript, pythonArgs)
    console.log(`[render] Python done: ${((Date.now()-tPy)/1000).toFixed(1)}s`)

    const outputStoragePath = `clips/${clip_id}/output.mp4`
    await r2.send(new PutObjectCommand({ Bucket: R2_BUCKET, Key: outputStoragePath, Body: createReadStream(outputPath), ContentType: 'video/mp4' }))

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

  const dims = QUALITY_DIMS[quality] ?? QUALITY_DIMS['1080p']
  return {
    clip_id: clipId,
    start_ms: clip.start_ms,
    end_ms: clip.end_ms,
    segments, caption_styles: captionStyles, text_overlays: textOverlays,
    audio_tracks: audioTracks, transitions, overlays, words,
    output_width: dims.w, output_height: dims.h,
  }
}

export async function renderClipWithLocalVideo(
  clipId: string,
  videoLocalPath: string,
  videoStoragePath: string,
): Promise<void> {
  const renderSpec = await buildRenderSpec(clipId, videoStoragePath, '1080p')
  const tmp = await mkdtemp(join(tmpdir(), 'render-'))
  const outputPath = join(tmp, 'output.mp4')
  const specPath = join(tmp, 'spec.json')
  try {
    await writeFile(specPath, JSON.stringify(renderSpec, null, 2))
    const pythonScript = join(__dirname, '../../src/python/render.py')
    await runPython(pythonScript, [
      '--video', videoLocalPath, '--spec', specPath, '--output', outputPath,
      '--secondary-videos', '{}', '--overlay-images', '{}', '--overlay-videos', '{}',
    ])
    const outputStoragePath = `clips/${clipId}/output.mp4`
    await r2.send(new PutObjectCommand({ Bucket: R2_BUCKET, Key: outputStoragePath, Body: createReadStream(outputPath), ContentType: 'video/mp4' }))
    const output_url = await getSignedUrl(r2, new GetObjectCommand({ Bucket: R2_BUCKET, Key: outputStoragePath }), { expiresIn: 60 * 60 * 24 * 7 })
    await db`UPDATE clips SET status = 'done', output_url = ${output_url}, output_storage_path = ${outputStoragePath} WHERE id = ${clipId}`
    console.log(`[render] Clip ${clipId} done`)
  } catch (err) {
    await db`UPDATE clips SET status = 'failed' WHERE id = ${clipId}`
    throw err
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
}

function runPython(script: string, args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const proc = spawn('python3', [script, ...args], { stdio: ['ignore', 'pipe', 'pipe'] })
    let stderr = ''
    proc.stdout?.on('data', (d: Buffer) => process.stdout.write(d))
    proc.stderr?.on('data', (d: Buffer) => { process.stderr.write(d); stderr += d.toString() })
    proc.on('close', code => { if (code === 0) resolve(); else reject(new Error(`Python render failed (exit ${code}): ${stderr.slice(-500)}`)) })
    proc.on('error', reject)
  })
}
