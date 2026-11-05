// ── A video's picture inside its black bars (lib/pictureBox.ts) ───────────────
// Found once per video, from frames spread over it (FFmpeg's cropdetect), and kept on the video
// (videos.picture_box: the whole frame when it has no bars, so it is never looked for again). The
// framing keeps inside it: Make my clips, Sub-edit, the editor (which reads the saved box).

import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import db from '../db.js'
import { r2SignedUrl } from '../r2.js'
import { pictureFromCrops, pictureOrWhole, WHOLE_FRAME, type PictureBox } from '../lib/pictureBox.js'

const execFileAsync = promisify(execFile)
/** Frames looked at: this many moments spread over the video, a few frames each */
const SAMPLES = 12
const FRAMES_EACH = 5
const AT_ONCE = 4

/**
 * The video's picture (fractions of the frame): saved, else found now (from `localPath` when the
 * video is already on disk, else over a link to it) and saved. When it can't be found the whole frame
 * is used and nothing is saved (the next job tries again) — the framing is then as it always was.
 */
export async function pictureOf(videoId: string, signal?: AbortSignal, localPath?: string): Promise<PictureBox> {
  const [v] = await db<{ picture_box: Partial<PictureBox> | null; storage_path: string | null; duration_ms: number | null }[]>`
    SELECT picture_box, storage_path, duration_ms FROM videos WHERE id = ${videoId}
  `
  if (!v) return WHOLE_FRAME
  if (v.picture_box) return pictureOrWhole(v.picture_box)
  try {
    const input = localPath ?? (v.storage_path ? await r2SignedUrl(v.storage_path, 3600) : null)
    if (!input) return WHOLE_FRAME
    const picture = await findPicture(input, v.duration_ms, signal)
    await db`UPDATE videos SET picture_box = ${db.json(picture as never)} WHERE id = ${videoId}`
    if (picture !== WHOLE_FRAME) console.log(`[picture] video ${videoId}: black bars — picture x ${picture.x.toFixed(3)} y ${picture.y.toFixed(3)} w ${picture.w.toFixed(3)} h ${picture.h.toFixed(3)}`)
    return picture
  } catch (e) {
    if (signal?.aborted) throw e
    console.warn(`[picture] video ${videoId}: bars not found, the whole frame is used:`, e instanceof Error ? e.message : e)
    return WHOLE_FRAME
  }
}

/** cropdetect on FRAMES_EACH frames at each of SAMPLES moments spread over the video */
async function findPicture(input: string, durationMs: number | null, signal?: AbortSignal): Promise<PictureBox> {
  const { stdout } = await execFileAsync('ffprobe', ['-v', 'error', '-select_streams', 'v:0',
    '-show_entries', 'stream=width,height:format=duration', '-of', 'json', input], { signal, timeout: 60_000 })
  const info = JSON.parse(stdout) as { streams?: Array<{ width?: number; height?: number }>; format?: { duration?: string } }
  const W = Number(info.streams?.[0]?.width), H = Number(info.streams?.[0]?.height)
  const seconds = durationMs ? durationMs / 1000 : Number(info.format?.duration)
  if (!(W > 0 && H > 0 && seconds > 0)) return WHOLE_FRAME
  const crops: Array<{ w: number; h: number; x: number; y: number }> = []
  let next = 0
  const run = async () => {
    while (next < SAMPLES) {
      const k = next++
      const t = ((k + 0.5) / SAMPLES) * seconds
      const { stderr } = await execFileAsync('ffmpeg', ['-hide_banner', '-ss', t.toFixed(2), '-i', input, '-frames:v', String(FRAMES_EACH),
        '-vf', 'cropdetect=limit=24:round=2:reset=0', '-f', 'null', '-'], { signal, timeout: 60_000 })
      const m = [...stderr.matchAll(/crop=(\d+):(\d+):(\d+):(\d+)/g)].pop()
      if (m) crops.push({ w: +m[1], h: +m[2], x: +m[3], y: +m[4] })
    }
  }
  await Promise.all(Array.from({ length: AT_ONCE }, run))
  return pictureFromCrops(crops, W, H)
}
