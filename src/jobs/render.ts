import { spawn } from 'node:child_process'
import { writeFile, readFile, mkdtemp, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { GetObjectCommand, PutObjectCommand } from '@aws-sdk/client-s3'
import { getSignedUrl } from '@aws-sdk/s3-request-presigner'
import { r2, R2_BUCKET } from '../r2.js'
import { supabase } from '../supabase.js'
import type { Job, RenderJobPayload } from '@chai-cut/shared'
import { fileURLToPath } from 'node:url'
import { dirname } from 'node:path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

async function r2Download(key: string): Promise<Buffer> {
  const res = await r2.send(new GetObjectCommand({ Bucket: R2_BUCKET, Key: key }))
  const chunks: Buffer[] = []
  for await (const chunk of res.Body as AsyncIterable<Uint8Array>) {
    chunks.push(Buffer.from(chunk))
  }
  return Buffer.concat(chunks)
}

export async function handleRenderJob(job: Job) {
  const payload = job.payload as unknown as RenderJobPayload
  const { clip_id, video_storage_path, quality = '1080p' } = payload

  const renderSpec = await buildRenderSpec(clip_id, video_storage_path, quality)

  const tmp = await mkdtemp(join(tmpdir(), 'render-'))
  const videoLocalPath = join(tmp, 'source.mp4')
  const outputPath = join(tmp, 'output.mp4')
  const specPath = join(tmp, 'spec.json')

  try {
    // Download primary source video from R2
    const videoBuffer = await r2Download(video_storage_path)
    await writeFile(videoLocalPath, videoBuffer)

    // Download secondary source videos referenced by crop boxes
    const secondaryVideos: Record<string, string> = {}
    const seenVideoIds = new Set<string>()

    for (const seg of (renderSpec.segments ?? []) as Array<{ crop_boxes?: Array<{ source_video_id?: string }> }>) {
      for (const box of (seg.crop_boxes ?? [])) {
        if (!box.source_video_id || seenVideoIds.has(box.source_video_id)) continue
        seenVideoIds.add(box.source_video_id)

        const { data: vRow } = await supabase
          .from('videos')
          .select('storage_path')
          .eq('id', box.source_video_id)
          .single()

        if (!vRow?.storage_path) continue

        try {
          const vBuffer = await r2Download(vRow.storage_path)
          const localPath = join(tmp, `secondary_${box.source_video_id}.mp4`)
          await writeFile(localPath, vBuffer)
          secondaryVideos[box.source_video_id] = localPath
        } catch (e) {
          console.warn(`[render] Failed to download secondary video ${box.source_video_id}:`, e)
        }
      }
    }

    // Download overlay images from R2
    const overlayImages: Record<string, string> = {}
    for (const ov of (renderSpec.overlays ?? []) as Array<{ type?: string; storage_path?: string }>) {
      if (ov.type !== 'image' || !ov.storage_path) continue

      try {
        const ovBuffer = await r2Download(ov.storage_path)
        const ext = ov.storage_path.split('.').pop() ?? 'png'
        const localPath = join(tmp, `overlay_${Buffer.from(ov.storage_path).toString('hex').slice(0, 16)}.${ext}`)
        await writeFile(localPath, ovBuffer)
        overlayImages[ov.storage_path] = localPath
      } catch (e) {
        console.warn(`[render] Failed to download overlay ${ov.storage_path}:`, e)
      }
    }

    await writeFile(specPath, JSON.stringify(renderSpec, null, 2))

    const pythonScript = join(__dirname, '../python/render.py')
    await runPython(pythonScript, [
      '--video', videoLocalPath,
      '--spec', specPath,
      '--output', outputPath,
      '--secondary-videos', JSON.stringify(secondaryVideos),
      '--overlay-images', JSON.stringify(overlayImages),
    ])

    const outputBuffer = await readFile(outputPath)
    const outputStoragePath = `clips/${clip_id}/output.mp4`

    await r2.send(new PutObjectCommand({
      Bucket: R2_BUCKET,
      Key: outputStoragePath,
      Body: outputBuffer,
      ContentType: 'video/mp4',
    }))

    const output_url = await getSignedUrl(
      r2,
      new GetObjectCommand({ Bucket: R2_BUCKET, Key: outputStoragePath }),
      { expiresIn: 60 * 60 * 24 * 7 },
    )

    await supabase
      .from('clips')
      .update({ status: 'done', output_url, output_storage_path: outputStoragePath })
      .eq('id', clip_id)

  } catch (err) {
    await supabase.from('clips').update({ status: 'failed' }).eq('id', clip_id)
    throw err
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
}

const QUALITY_DIMS: Record<string, { w: number; h: number }> = {
  '480p':  { w: 480,  h: 854  },
  '720p':  { w: 720,  h: 1280 },
  '1080p': { w: 1080, h: 1920 },
  '2160p': { w: 2160, h: 3840 },
}

async function buildRenderSpec(clipId: string, videoStoragePath: string, quality = '1080p') {
  const [
    { data: clip },
    { data: segments },
    { data: captionStyles },
    { data: textOverlays },
    { data: audioTracks },
    { data: transitions },
    { data: overlays },
  ] = await Promise.all([
    supabase.from('clips').select('*').eq('id', clipId).single(),
    supabase.from('segments')
      .select('*, crop_boxes(*, box_keyframes(*))')
      .eq('clip_id', clipId)
      .order('sort_order'),
    supabase.from('caption_styles').select('*').eq('clip_id', clipId),
    supabase.from('text_overlays').select('*').eq('clip_id', clipId),
    supabase.from('audio_tracks').select('*').eq('clip_id', clipId),
    supabase.from('transitions').select('*').eq('clip_id', clipId),
    supabase.from('overlays').select('*').eq('clip_id', clipId).order('z_index'),
  ])

  const { data: transcripts } = await supabase
    .from('transcripts')
    .select('id')
    .eq('video_id', (clip as { video_id: string }).video_id)
    .order('created_at', { ascending: false })
    .limit(1)

  const transcriptId = transcripts?.[0]?.id
  const { data: words } = transcriptId
    ? await supabase
        .from('transcript_words')
        .select('word, start_ms, end_ms, speaker_id')
        .eq('transcript_id', transcriptId)
        .order('start_ms')
    : { data: [] }

  return {
    clip_id: clipId,
    start_ms: (clip as { start_ms: number }).start_ms,
    end_ms: (clip as { end_ms: number }).end_ms,
    segments,
    caption_styles: captionStyles,
    text_overlays: textOverlays,
    audio_tracks: audioTracks,
    transitions,
    overlays: overlays ?? [],
    words: words ?? [],
    output_width:  (QUALITY_DIMS[quality] ?? QUALITY_DIMS['1080p']).w,
    output_height: (QUALITY_DIMS[quality] ?? QUALITY_DIMS['1080p']).h,
  }
}

function runPython(script: string, args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const proc = spawn('python3', [script, ...args], {
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    let stderr = ''
    proc.stdout?.on('data', (d: Buffer) => process.stdout.write(d))
    proc.stderr?.on('data', (d: Buffer) => { process.stderr.write(d); stderr += d.toString() })
    proc.on('close', code => {
      if (code === 0) resolve()
      else reject(new Error(`Python render failed (exit ${code}): ${stderr.slice(-500)}`))
    })
    proc.on('error', reject)
  })
}
