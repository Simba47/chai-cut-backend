import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { mkdtemp, rm, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const execFileAsync = promisify(execFile)

// ── A video's picture ("poster") for the dashboard ─────────────────────────────
// The dashboard showed each video by loading the video itself (its frame at 1 s): every card started on the
// whole file, slow with many videos, and the first second is often black, a fade or a channel's logo. One
// small JPEG instead (user, 2026-11-11), made once the video is ready: from a moment a little way in (a tenth
// of the way, 2–30 s), the clearest of the frames around it (ffmpeg's thumbnail filter skips black and blurred
// ones), 640 px wide. A video without one (older, or this failed) is shown as before.

/** Where a video's picture is kept: next to it (deleted with it: see the frontend's deletes) */
export function posterKeyFor(storagePath: string): string {
  return storagePath.replace(/\.[^.]+$/, '') + '_poster.jpg'
}

/** Where in the video the picture is taken from (seconds) */
export function posterAt(durationMs: number | null | undefined): number {
  if (!durationMs || durationMs <= 0) return 2
  const s = durationMs / 1000
  // (a very short video: from its middle)
  if (s < 4) return Math.max(0, s / 2)
  return Math.min(30, Math.max(2, s * 0.1))
}

/**
 * Make the video's picture and keep it (videos.poster_path). `localPath`: the video already on disk (quicker);
 * else it's read over a signed link (ffmpeg fetches only the part it needs). Never fails the caller: a problem
 * is only logged, and the dashboard shows the video as before.
 */
export async function makePoster(videoId: string, localPath?: string, signal?: AbortSignal): Promise<boolean> {
  let tmp: string | null = null
  try {
    // (storage and the database are loaded here, not at the top: the rules above are tested without them)
    const { r2SignedUrl, r2UploadFile } = await import('../r2.js')
    const { default: db } = await import('../db.js')
    const [video] = await db<{ storage_path: string | null; duration_ms: number | null; poster_path: string | null }[]>`
      SELECT storage_path, duration_ms, to_jsonb(videos)->>'poster_path' AS poster_path FROM videos WHERE id = ${videoId}`
    if (!video?.storage_path || video.poster_path) return false
    const storagePath = video.storage_path
    const input = localPath ?? await r2SignedUrl(storagePath, 15 * 60)
    tmp = await mkdtemp(join(tmpdir(), 'chai-poster-'))
    const out = join(tmp, 'poster.jpg')
    // Over a link: carry on through a dropped connection
    const http = /^https?:/.test(input) ? ['-reconnect', '1', '-reconnect_streamed', '1', '-reconnect_delay_max', '5'] : []
    const grab = (at: number) => execFileAsync('ffmpeg', [
      '-hide_banner', '-loglevel', 'error', '-ss', at.toFixed(2), ...http, '-i', input,
      '-an', '-sn', '-vf', 'thumbnail=48,scale=640:-2', '-frames:v', '1', '-q:v', '4', '-y', out,
    ], { signal, timeout: 90_000 })
    const made = async () => (await stat(out).catch(() => null))?.size ?? 0
    await grab(posterAt(video.duration_ms)).catch(() => undefined)
    // (past its end, or a stream that won't seek: from the start)
    if (!(await made())) await grab(0)
    if (!(await made())) throw new Error('ffmpeg wrote no picture')
    const key = posterKeyFor(storagePath)
    await r2UploadFile(key, out, 'image/jpeg')
    // (only if the video is still the same file: replaced meanwhile, its picture would be another's)
    await db`UPDATE videos SET poster_path = ${key} WHERE id = ${videoId} AND storage_path = ${storagePath}`
    console.log(`[poster] video ${videoId}: picture kept (${key})`)
    return true
  } catch (e) {
    if (signal?.aborted) return false
    console.warn(`[poster] video ${videoId}: no picture —`, e instanceof Error ? e.message : e)
    return false
  } finally {
    if (tmp) await rm(tmp, { recursive: true, force: true }).catch(() => {})
  }
}
