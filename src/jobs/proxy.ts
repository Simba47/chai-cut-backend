import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { mkdtemp, rm, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { DeleteObjectCommand } from '@aws-sdk/client-s3'
import { r2, R2_BUCKET, r2DownloadToFile, r2UploadFile } from '../r2.js'
import db from '../db.js'
import type { Job, ProxyJobPayload } from '../types.js'

const execFileAsync = promisify(execFile)

// ── The editing copy ("proxy") of a video ──────────────────────────────────────
// What the editor plays, so dragging through a video is quick. Uploads and downloads usually have
// a full picture (keyframe) only every few seconds (6–7 s in videos we checked): every seek then
// decodes from the one before, at full size (~450 ms a seek). This copy has one every half second,
// no B-frames and is at most 720p on its short side (~60 ms a seek), about a quarter of the size.
// Its frames keep their times, so it lines up with the original exactly. The export always uses
// the original; the copy is only for watching while editing.

/** Its short side at most this many pixels */
const PROXY_SHORT_SIDE = 720

/** Where a video's editing copy is kept: next to it (deleted with it: see the frontend's deletes) */
export function proxyKeyFor(storagePath: string): string {
  return storagePath.replace(/\.[^.]+$/, '') + '_proxy.mp4'
}

/** The ffmpeg arguments for the editing copy */
export function proxyArgs(input: string, output: string): string[] {
  const s = PROXY_SHORT_SIDE
  return [
    '-hide_banner', '-loglevel', 'error', '-y', '-i', input,
    '-map', '0:v:0', '-map', '0:a:0?',
    // Landscape: at most 720 tall; portrait: at most 720 wide (never made bigger)
    '-vf', `scale='if(gt(iw,ih),-2,min(${s},iw))':'if(gt(iw,ih),min(${s},ih),-2)',format=yuv420p`,
    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28', '-bf', '0',
    // A keyframe every half second, whatever the frame rate; none extra at scene changes
    '-force_key_frames', 'expr:gte(t,n_forced*0.5)', '-sc_threshold', '0',
    // Every frame keeps its own time (phone videos have uneven frame rates)
    '-fps_mode', 'passthrough',
    '-c:a', 'aac', '-b:a', '96k', '-ac', '2',
    '-movflags', '+faststart',
    // Two cores at most: renders run on the same worker
    '-threads', '2',
    output,
  ]
}

/**
 * Queue the editing copy of a video (once: not while one is queued or being made). Best effort:
 * the editor plays the original until the copy is there.
 */
export async function queueProxy(videoId: string): Promise<void> {
  await db`
    INSERT INTO jobs (type, payload, status)
    SELECT 'proxy', ${db.json({ video_id: videoId } as never)}, 'queued'
    WHERE NOT EXISTS (
      SELECT 1 FROM jobs WHERE type = 'proxy' AND status IN ('queued', 'processing') AND payload->>'video_id' = ${videoId}
    )
  `
}

export async function handleProxyJob(job: Job, signal?: AbortSignal) {
  const raw = job.payload
  const { video_id } = (typeof raw === 'string' ? JSON.parse(raw) : raw) as ProxyJobPayload
  const [video] = await db`SELECT id, storage_path, status, proxy_path FROM videos WHERE id = ${video_id}`
  // Gone, not ready yet, or already has one: nothing to do
  if (!video || !video.storage_path || video.status !== 'ready' || video.proxy_path) {
    console.log(`[proxy] video ${video_id}: nothing to do`)
    return
  }
  const storagePath = video.storage_path as string
  const tmp = await mkdtemp(join(tmpdir(), 'chai-proxy-'))
  try {
    const src = join(tmp, 'source' + (storagePath.match(/\.[^./]+$/)?.[0] ?? '.mp4'))
    const out = join(tmp, 'proxy.mp4')
    await r2DownloadToFile(storagePath, src, signal)
    const started = Date.now()
    await execFileAsync('ffmpeg', proxyArgs(src, out), { signal, maxBuffer: 16 * 1024 * 1024 })
    const key = proxyKeyFor(storagePath)
    await r2UploadFile(key, out, 'video/mp4')
    // Only if it is still the same video (it may have been deleted meanwhile: then the copy goes too)
    const saved = await db`UPDATE videos SET proxy_path = ${key} WHERE id = ${video_id} AND storage_path = ${storagePath} RETURNING id`
    if (!saved.length) {
      await r2.send(new DeleteObjectCommand({ Bucket: R2_BUCKET, Key: key })).catch(() => {})
      console.log(`[proxy] video ${video_id} changed or was deleted while its copy was made: copy removed`)
      return
    }
    const [a, b] = await Promise.all([stat(src), stat(out)])
    console.log(`[proxy] video ${video_id}: editing copy made in ${Math.round((Date.now() - started) / 1000)} s, ${(b.size / 1e6).toFixed(1)} MB (original ${(a.size / 1e6).toFixed(1)} MB)`)
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }
}
