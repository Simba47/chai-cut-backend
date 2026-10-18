import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import db from '../db.js'
import { r2SignedUrl } from '../r2.js'
import type { Job } from '../types.js'

const execFileAsync = promisify(execFile)

// ── Silences of a clip (the editor's Cleanup tool) ─────────────────────────────
// Cleanup lists what can go from a clip; its silences come from the sound itself (not the captions):
// ffmpeg's silencedetect over the clip's stretch of the video, read straight from storage (only that
// stretch is fetched). Saved into clip_silences for the editor to read; each becomes a card there.

/** Quieter than this counts as silence (speech is around -20 to -10 dB, room noise -45 to -60) */
export const SILENCE_NOISE_DB = -35
/** Silences at least this long, seconds */
export const SILENCE_MIN_S = 0.6
/** A clip's scan reads at most this much (the editor's clips are at most 5 minutes) */
const MAX_SCAN_MS = 6 * 60_000

export interface DetectSilenceJobPayload { clip_id: string; start_ms: number; end_ms: number }

/**
 * ffmpeg's silencedetect lines → silent stretches, ms from the start of what was read. A silence
 * still going at the end (no silence_end) runs to `durationMs`.
 */
export function parseSilences(log: string, durationMs: number): Array<[number, number]> {
  const out: Array<[number, number]> = []
  let start: number | null = null
  for (const line of log.split(/\r?\n/)) {
    const s = /silence_start:\s*(-?[\d.]+)/.exec(line)
    if (s) { start = Math.max(0, parseFloat(s[1]) * 1000); continue }
    const e = /silence_end:\s*(-?[\d.]+)/.exec(line)
    if (e && start !== null) {
      const end = Math.min(durationMs, parseFloat(e[1]) * 1000)
      if (end > start) out.push([Math.round(start), Math.round(end)])
      start = null
    }
  }
  if (start !== null && durationMs > start) out.push([Math.round(start), Math.round(durationMs)])
  return out
}

export async function handleDetectSilenceJob(job: Job, signal?: AbortSignal) {
  const { clip_id, start_ms, end_ms } = job.payload as unknown as DetectSilenceJobPayload
  await db`UPDATE clip_silences SET status = 'running', updated_at = now() WHERE clip_id = ${clip_id}`
  try {
    const [v] = await db<{ storage_path: string | null; proxy_path: string | null }[]>`
      SELECT v.storage_path, v.proxy_path FROM clips c JOIN videos v ON v.id = c.video_id WHERE c.id = ${clip_id}
    `
    if (!v?.storage_path) throw new Error('The video of this clip could not be found')
    const from = Math.max(0, Math.round(start_ms))
    const dur = Math.max(1000, Math.min(MAX_SCAN_MS, Math.round(end_ms) - from))
    // (the editing copy when there is one: the same sound, a quarter of the size)
    const url = await r2SignedUrl(v.proxy_path || v.storage_path, 30 * 60)
    const { stderr } = await execFileAsync('ffmpeg', [
      '-hide_banner', '-nostats', '-ss', (from / 1000).toFixed(3), '-t', (dur / 1000).toFixed(3), '-i', url,
      '-vn', '-af', `silencedetect=noise=${SILENCE_NOISE_DB}dB:d=${SILENCE_MIN_S}`, '-f', 'null', '-',
    ], { signal, maxBuffer: 16 * 1024 * 1024 })
    const ranges = parseSilences(stderr, dur).map(([a, b]) => [from + a, from + b])
    await db`
      UPDATE clip_silences SET status = 'done', error = NULL, ranges = ${db.json(ranges as never)},
        scan_start_ms = ${from}, scan_end_ms = ${from + dur}, updated_at = now()
      WHERE clip_id = ${clip_id}
    `
    console.log(`[silence] clip ${clip_id}: ${ranges.length} silences in ${(dur / 1000).toFixed(1)} s`)
  } catch (e) {
    const why = e instanceof Error ? e.message : String(e)
    await db`UPDATE clip_silences SET status = 'failed', error = ${why.slice(0, 300)}, updated_at = now() WHERE clip_id = ${clip_id}`.catch(() => {})
    throw e
  }
}
