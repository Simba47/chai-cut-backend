import { execFile, spawn } from 'node:child_process'
import { promisify } from 'node:util'
import { writeFile, readFile, unlink, mkdtemp, readdir, rm } from 'node:fs/promises'
import { createWriteStream } from 'node:fs'
import { Readable, Transform } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import type { ReadableStream as WebReadableStream } from 'node:stream/web'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { DeleteObjectCommand, GetObjectCommand, PutObjectCommand } from '@aws-sdk/client-s3'
import { getSignedUrl } from '@aws-sdk/s3-request-presigner'
import { GoogleAuth } from 'google-auth-library'
import { r2, R2_BUCKET, r2DownloadToFile, r2SignedUrl, r2UploadFile } from '../r2.js'
import { aiProvider, askGemini, dropCode, isNetworkDrop, retryAfterMs, retryDelayMs } from '../lib/gemini.js'
import { sttRank, sttTurn, waitForStt, wavSeconds, type SttSide } from '../lib/googleBudget.js'
import { runawayReason, trimRunaway } from '../lib/transcriptGuard.js'
import { skippedGaps, rereadWindows, wordsInGaps, REREAD_MIN_SOUND_SEC, REREAD_ROUNDS, REREAD_ROUND_MIN_WORDS } from '../lib/skippedSpeech.js'
import { convertWordForWord } from '../lib/wordConvert.js'
import db from '../db.js'
import type { Job, TranscribeJobPayload } from '../types.js'
import { audioKeyFor, queueProxy } from './proxy.js'

async function r2Download(key: string): Promise<Buffer> {
  const res = await r2.send(new GetObjectCommand({ Bucket: R2_BUCKET, Key: key }))
  const chunks: Buffer[] = []
  for await (const chunk of res.Body as AsyncIterable<Uint8Array>) {
    chunks.push(Buffer.from(chunk))
  }
  return Buffer.concat(chunks)
}

async function r2Upload(key: string, body: Buffer, contentType: string): Promise<void> {
  await r2.send(new PutObjectCommand({ Bucket: R2_BUCKET, Key: key, Body: body, ContentType: contentType }))
}

const execFileAsync = promisify(execFile)

/** A failure the user can act on: its message is shown on the video's card */
class UserFacingError extends Error {}

async function setProgress(videoId: string, pct: number) {
  await db`UPDATE videos SET download_progress = ${pct} WHERE id = ${videoId}`
}

/** What a video with no sound track is told (captions, Make my clips, Best moments and Ask AI all need speech) */
export const NO_SOUND_MESSAGE = 'This video has no sound, so there is nothing to caption or to pick clips from. '
  + 'Videos saved from YouTube in 1080p often come without their sound: download it again with audio, then upload that one.'

/**
 * Stops with NO_SOUND_MESSAGE when the file has no audio track at all (FFmpeg would fail with
 * "Output file does not contain any stream" and pages of its own output). A file ffprobe can't
 * read is left to FFmpeg, which reports it.
 */
export async function requireSound(input: string, signal?: AbortSignal): Promise<void> {
  let tracks: string
  try {
    const { stdout } = await execFileAsync('ffprobe', [
      '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=index', '-of', 'csv=p=0', input,
    ], { signal, timeout: 120_000 })
    tracks = stdout.trim()
  } catch {
    return
  }
  if (!tracks) throw new UserFacingError(NO_SOUND_MESSAGE)
}

/** Length of a video file or URL in ms, or null if ffprobe can't tell */
async function probeDurationMs(input: string, signal?: AbortSignal): Promise<number | null> {
  try {
    const { stdout } = await execFileAsync('ffprobe', [
      '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input,
    ], { signal, timeout: 60_000 })
    const secs = parseFloat(stdout.trim())
    return isNaN(secs) ? null : Math.round(secs * 1000)
  } catch {
    return null
  }
}

/**
 * The browser reads a video's length when it's uploaded, but can't for formats it doesn't play
 * (MKV, iPhone HEVC…). Fill it in from storage then: ffprobe reads just the header over a signed URL.
 */
async function fillMissingDuration(videoId: string, storagePath: string, signal?: AbortSignal) {
  const [row] = await db`SELECT duration_ms FROM videos WHERE id = ${videoId}`
  if (!row || row.duration_ms) return
  const url = await getSignedUrl(r2, new GetObjectCommand({ Bucket: R2_BUCKET, Key: storagePath }), { expiresIn: 600 })
  const durationMs = await probeDurationMs(url, signal)
  if (durationMs) await db`UPDATE videos SET duration_ms = ${durationMs} WHERE id = ${videoId} AND duration_ms IS NULL`
}

/** How far a reading has got: `done` of `total` chunks (Make my clips shows it while it reads a video) */
export type ReadProgress = (done: number, total: number) => void | Promise<void>

/** What the clip board's Stop writes on a whole-video reading (the frontend's stopVideoTranscript) */
export const READING_STOPPED = 'Stopped by the user'

/**
 * A transcription job. A whole-video reading (the transcript the edit tools need) can be stopped
 * from the clip board: its job row says so, checked every 2 s, and everything running (FFmpeg, the
 * AI calls) is stopped. The parts already read are kept (the reading store), so a later Unlock
 * carries on from there.
 */
export async function handleTranscribeJob(job: Job, signal?: AbortSignal, onRead?: ReadProgress) {
  const raw = job.payload
  const p = (typeof raw === 'string' ? JSON.parse(raw) : raw) as TranscribeJobPayload
  // Its place in the shared Google line (lib/googleBudget.ts): a clip's captions (someone waits in
  // the editor) before a whole-video reading, the top plan first within each
  const turn = { rank: sttRank(!p.clip_id, p.priority === true), jobId: /^[0-9a-f-]{36}$/i.test(job.id) ? job.id : undefined }
  return sttTurn.run(turn, () => runTranscribeJob(job, p, signal, onRead))
}

async function runTranscribeJob(job: Job, p: TranscribeJobPayload, signal?: AbortSignal, onRead?: ReadProgress) {
  const stoppable = !!p.transcribe_full && !p.clip_id && !p.is_retranscribe && /^[0-9a-f-]{36}$/i.test(job.id)
  if (!stoppable) return transcribeJob(job, signal, onRead)
  const stop = new AbortController()
  const watch = setInterval(async () => {
    try {
      const [row] = await db<{ status: string; error: string | null }[]>`SELECT status, error FROM jobs WHERE id = ${job.id}`
      if (row && row.status === 'failed' && row.error === READING_STOPPED) stop.abort(new Error(READING_STOPPED))
    } catch { /* the next check */ }
  }, 2000)
  try {
    return await transcribeJob(job, signal ? AbortSignal.any([signal, stop.signal]) : stop.signal, onRead)
  } catch (e) {
    if (stop.signal.aborted) {
      console.log(`[transcribe] reading of video ${p.video_id} stopped by the user (the parts read are kept)`)
      throw new Error(READING_STOPPED)
    }
    throw e
  } finally {
    clearInterval(watch)
  }
}

async function transcribeJob(job: Job, signal?: AbortSignal, onRead?: ReadProgress) {
  const raw = job.payload
  const payload = (typeof raw === 'string' ? JSON.parse(raw) : raw) as TranscribeJobPayload
  const isLinkJob = !payload.storage_path
  const isRetranscribe = !!payload.is_retranscribe
  const isClipJob = !!payload.clip_id && payload.clip_start_ms !== undefined && payload.clip_end_ms !== undefined
  // Full-video captions right after upload/download, so clips open with captions ready.
  // The video is marked ready first — clipping never waits on this background transcription.
  const isFullJob = !!payload.transcribe_full && !isClipJob && !isRetranscribe
  // "Captions on" for a clip on a video already read in full: its words are all there (a part with
  // none is a pause or music), so it isn't read, and paid for, again. The transcript's time moves on,
  // so the editor (waiting for new words) shows them at once. A language picked, or an export with
  // fresh captions, still reads.
  if (isClipJob && payload.fill_gaps && !payload.render_after) {
    const [latest] = await db<{ id: string; whole_video: boolean | null }[]>`
      SELECT id, whole_video FROM transcripts WHERE video_id = ${payload.video_id} ORDER BY created_at DESC LIMIT 1`
    if (latest?.whole_video) {
      await db`UPDATE transcripts SET created_at = now() WHERE id = ${latest.id}`
      console.log(`[transcribe] clip ${payload.clip_id}: the video was read in full, nothing to read`)
      return
    }
  }
  let videoReady = false
  // A whole-video reading (AI edit) says how far it has got, in its job (read_pct): the clip board
  // shows it while Best moments and Ask AI wait for the words
  const report: ReadProgress | undefined = onRead ?? (isFullJob
    ? async (done, total) => {
      await db`UPDATE jobs SET payload = payload || ${db.json({ read_pct: Math.round((100 * done) / Math.max(1, total)) })} WHERE id = ${job.id}`
    }
    : undefined)

  // Plain upload jobs (already in storage, no clip): mark ready. Without transcribe_full
  // (older jobs) transcription is deferred to clip creation.
  if (!isLinkJob && !isClipJob && !isRetranscribe) {
    await db`UPDATE videos SET status = 'ready', download_progress = 100 WHERE id = ${payload.video_id}`
    videoReady = true
    await fillMissingDuration(payload.video_id, payload.storage_path, signal)
      .catch(e => console.warn('[transcribe] could not read video length:', e))
    // Its editing copy, made when the worker has nothing else to do (jobs/proxy.ts)
    await queueProxy(payload.video_id).catch(e => console.warn('[transcribe] could not queue the editing copy:', e))
    if (!isFullJob) {
      console.log(`[transcribe] upload video ${payload.video_id} ready — transcription deferred to clip creation`)
      return
    }
    console.log(`[transcribe] upload video ${payload.video_id} ready — captioning full video in background`)
  }

  if (isLinkJob && !isRetranscribe && !isClipJob) {
    await db`UPDATE videos SET status = 'transcribing', download_progress = 0 WHERE id = ${payload.video_id}`
  }

  const tmp = await mkdtemp(join(tmpdir(), 'chai-'))
  const startedAt = Date.now()
  const secs = () => `${((Date.now() - startedAt) / 1000).toFixed(1)} s`

  try {
    let videoPath = ''
    let storagePath = payload.storage_path

    const audioPath = join(tmp, 'audio.wav')

    // ── Fast path: the video's audio kept in storage (FLAC) ─────────────────────
    // Kept by the first whole-video reading, and by the editing copy (jobs/proxy.ts)
    const audioStoragePath = storagePath ? audioKeyFor(storagePath) : null
    let audioCached = false
    if ((isRetranscribe || isFullJob) && audioStoragePath) {
      try {
        const cachedBuf = await r2Download(audioStoragePath)
        const flacPath = join(tmp, 'cached.flac')
        await writeFile(flacPath, cachedBuf)
        // The cache holds the full video's audio — for clip jobs cut out just the clip's
        // range, since word times are offset by clip_start_ms below
        const clipRange = isClipJob
          ? ['-ss', String(payload.clip_start_ms! / 1000), '-t', String((payload.clip_end_ms! - payload.clip_start_ms!) / 1000)]
          : []
        await execFileAsync('ffmpeg', [...clipRange, '-i', flacPath, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', audioPath])
        audioCached = true
        console.log(`[transcribe] used the kept audio (${audioStoragePath})`)
      } catch { /* none kept: read it from the video */ }
    }

    if (!audioCached) {
      if (isLinkJob) {
        // Link job: Google Drive / Dropbox → plain download; other sites → yt-dlp (home worker only).
        // Either way the file is uploaded to storage.
        const result = payload.link_source
          ? await downloadDirectLink(payload.video_id, payload.link_source, tmp, payload.max_bytes, signal)
          : await downloadWithYtDlp(payload.video_id, tmp, signal)
        videoPath = result.localPath
        storagePath = result.storagePath
        await db`UPDATE videos SET storage_path = ${storagePath} WHERE id = ${payload.video_id}`

        // A link import is ready as soon as the file is stored — the user can start clipping
        // while captions are made in the background (or not at all on plans without them, in
        // which case the audio isn't needed and the job ends here)
        if (!isClipJob && !isRetranscribe) {
          const durationMs = await probeDurationMs(videoPath, signal)
          await db`UPDATE videos SET status = 'ready', storage_path = ${storagePath}, download_progress = 100, duration_ms = ${durationMs} WHERE id = ${payload.video_id}`
          videoReady = true
          await queueProxy(payload.video_id).catch(e => console.warn('[transcribe] could not queue the editing copy:', e))
          if (!isFullJob) {
            console.log(`[transcribe] link video ${payload.video_id} ready — no captions on this plan`)
            return
          }
          console.log(`[transcribe] link video ${payload.video_id} ready — captioning full video in background`)
        }
      }

      // Extract audio — for clip jobs, seek to clip range only (fast & cheap)
      const extract = async (input: string) => {
        await requireSound(input, signal)
        // Over a link: carry on through a dropped connection instead of ending the audio early
        const http = /^https?:/.test(input) ? ['-reconnect', '1', '-reconnect_streamed', '1', '-reconnect_delay_max', '5'] : []
        const range = isClipJob
          ? ['-ss', String(payload.clip_start_ms! / 1000), ...http, '-i', input, '-t', String((payload.clip_end_ms! - payload.clip_start_ms!) / 1000)]
          : [...http, '-i', input]
        await execFileAsync('ffmpeg', [...range, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', audioPath], { signal })
      }
      await setProgress(payload.video_id, 60)
      if (isLinkJob) {
        await extract(videoPath)
      } else {
        // In storage: the audio is read straight from it over a signed link, so a clip reads only
        // its own part of the video and a whole video isn't saved to disk first. If that fails, the
        // video is downloaded (streamed to disk — a whole video held in memory could run the
        // worker out of it) and read from there.
        const url = await r2SignedUrl(storagePath, 4 * 3600)
        try {
          await extract(url)
        } catch (e) {
          if (signal?.aborted || e instanceof UserFacingError) throw e
          console.warn('[transcribe] reading the audio from storage failed, downloading the video first:', (e instanceof Error ? e.message : String(e)).split('\n')[0])
          videoPath = join(tmp, 'video' + (storagePath.match(/\.[^./]+$/)?.[0] ?? '.mp4'))
          await r2DownloadToFile(storagePath, videoPath, signal)
          await extract(videoPath)
        }
      }
      if (!isClipJob) {
        // Cache full-video audio as FLAC for fast retranscription
        const cachePath = audioStoragePath ?? (storagePath ? audioKeyFor(storagePath) : null)
        if (cachePath) {
          const flacOut = join(tmp, 'audio_cache.flac')
          execFileAsync('ffmpeg', ['-i', audioPath, '-compression_level', '5', '-y', flacOut])
            .then(() => readFile(flacOut))
            .then(buf => r2Upload(cachePath, buf, 'audio/flac'))
            .catch(e => console.warn('[transcribe] audio cache upload failed:', e))
        }
      }
    }

    console.log(`[transcribe] audio ready after ${secs()}`)

    // ── Transcription ─────────────────────────────────────────────────────────
    // Progress only matters while the video is still 'transcribing' (not for background full-video captions)
    // Captioning part of a video again (a clip, or "fresh captions"): the video's language is
    // already known, so skip detecting it (for a clip that meant reading the clip twice). Only an
    // Indian language is reused; English runs on auto detection, which reads the audio once anyway.
    let languageCode = payload.language_code
    if (!languageCode && (isClipJob || isRetranscribe)) {
      const [known] = await db`
        SELECT language FROM transcripts WHERE video_id = ${payload.video_id} AND language IS NOT NULL
        ORDER BY created_at DESC LIMIT 1`
      const lang = known?.language as string | undefined
      if (lang && lang !== 'unknown' && !lang.startsWith('en')) languageCode = lang
    }
    // A whole video is read in 5-minute chunks, and the transcript is saved only when all of them
    // are in. Each finished chunk is kept in storage next to the video, so a run that failed or
    // was stopped part-way (Google refusing, the daily limit, "Stop") is continued by the next
    // one instead of being read, and paid for, from the start again.
    const readingKey = !isClipJob && storagePath ? storagePath.replace(/\.[^.]+$/, '_reading.json') : null
    const kept: ReadingStore | undefined = readingKey ? {
      load: async () => JSON.parse((await r2Download(readingKey)).toString('utf8')) as Reading,
      save: reading => r2Upload(readingKey, Buffer.from(JSON.stringify(reading)), 'application/json'),
    } : undefined
    const sarvamResult = await transcribeAudio(audioPath, (!isRetranscribe && !isClipJob && !videoReady) ? payload.video_id : undefined, languageCode, signal, kept, report)
    signal?.throwIfAborted()

    // The words come spelled (transcribeAudio spells each chunk as it is read: in Roman letters, and
    // English spoken inside a regional-language video in that language's script, so "Auto language"
    // captions are all in one script; word_roman keeps the English spelling). Only then is anything
    // saved, all at once: a transcript saved before its words read as a whole video with no speech
    // (Make my clips failed with "no speech" while it was being written, and for good when a restart
    // cut it off half-saved)
    const entries = sarvamResult.words
    const offsetMs = isClipJob ? payload.clip_start_ms! : 0

    // Clip jobs (first transcription or retranscribe): replace only this clip's time range in the
    // video's transcript, so other clips of the same video keep their captions. The editor reads
    // one transcript per video (the newest), so a clip job must never start a separate one —
    // that hid every other clip's captions. created_at is bumped so the editor's "since" poll
    // picks up the new words.
    await db.begin(async tx => {
      const [existing] = isClipJob
        ? await tx`
            SELECT t.id FROM transcripts t
            WHERE t.video_id = ${payload.video_id}
              AND (t.whole_video OR EXISTS (SELECT 1 FROM transcript_words w WHERE w.transcript_id = t.id))
            ORDER BY t.created_at DESC LIMIT 1`
        : []
      let transcriptId: string
      if (existing) {
        await tx`DELETE FROM transcript_words WHERE transcript_id = ${existing.id} AND start_ms >= ${payload.clip_start_ms!} AND start_ms < ${payload.clip_end_ms!}`
        await tx`UPDATE transcripts SET language = ${sarvamResult.language_code}, created_at = now() WHERE id = ${existing.id}`
        transcriptId = existing.id as string
      } else {
        if (isRetranscribe) {
          await tx`DELETE FROM transcripts WHERE video_id = ${payload.video_id}`
        }
        // whole_video: the whole video was read, so nothing needs to read it again (see ai_edit)
        // (gaps_filled_at: read with the skipped-speech re-read, nothing to fill later)
        const [inserted] = await tx`
          INSERT INTO transcripts (video_id, language, whole_video, gaps_filled_at) VALUES (${payload.video_id}, ${sarvamResult.language_code}, ${!isClipJob}, now()) RETURNING id
        `
        if (!inserted) throw new Error('Failed to insert transcript row')
        transcriptId = inserted.id as string
      }

      const words = entries.map(e => ({
        transcript_id: transcriptId,
        word: e.native ?? e.word,
        word_roman: e.roman ?? null,
        start_ms: Math.round(e.start * 1000) + offsetMs,
        end_ms:   Math.round(e.end   * 1000) + offsetMs,
        speaker_id:  e.speaker ?? null,
        confidence:  e.confidence ?? null,
      }))
      for (let i = 0; i < words.length; i += 500) {
        await tx`INSERT INTO transcript_words ${tx(words.slice(i, i + 500))}`
      }
    })

    // The transcript is saved: the kept chunks have done their job
    if (readingKey) await r2.send(new DeleteObjectCommand({ Bucket: R2_BUCKET, Key: readingKey })).catch(() => {})

    // (the video was marked ready, with its length, before it was read)
    console.log(`[transcribe] ${isClipJob ? `clip ${payload.clip_id}` : `video ${payload.video_id}`} done — ${entries.length} words in ${secs()}`)
  } catch (err) {
    // Only mark failed for fresh download jobs — retranscribe is called inline from ai_edit
    // on an already-ready video, and a failed background caption job leaves the video usable.
    if (!isClipJob && !isRetranscribe && !videoReady) {
      // A reason the user can act on is shown on the video's card (e.g. "set sharing to anyone
      // with the link"); anything else shows the generic "couldn't process" message
      const reason = err instanceof UserFacingError ? err.message : null
      await db`UPDATE videos SET status = 'failed', error = ${reason} WHERE id = ${payload.video_id}`
        .catch(() => db`UPDATE videos SET status = 'failed' WHERE id = ${payload.video_id}`.catch(() => {}))
    }
    throw err
  } finally {
    await rm(tmp, { recursive: true, force: true })
    // Re-render requested: queue the render now that captions are updated. Also on
    // failure, so the clip renders with its previous captions instead of staying stuck.
    if (payload.render_after) {
      const render = payload.render_after
      await db`INSERT INTO jobs (type, payload, status) VALUES ('render', ${db.json(render as never)}, 'queued')`
        .then(() => console.log(`[transcribe] queued render for clip ${render.clip_id}`))
        .catch(async e => {
          console.error(`[transcribe] failed to queue render for clip ${render.clip_id}:`, e)
          await db`UPDATE clips SET status = 'failed' WHERE id = ${render.clip_id}`.catch(() => {})
        })
    }
  }
}

// ── Google Drive / Dropbox share links ─────────────────────────────────────────
// A public share link downloads with plain HTTP from any server (no yt-dlp, no home IP needed).

const LINK_NAMES = { gdrive: 'Google Drive', dropbox: 'Dropbox' } as const
const VIDEO_EXT_BY_TYPE: Record<string, string> = {
  'video/mp4': '.mp4', 'video/quicktime': '.mov', 'video/webm': '.webm', 'video/x-matroska': '.mkv',
}
const VIDEO_TYPE_BY_EXT: Record<string, string> = { '.mp4': 'video/mp4', '.mov': 'video/quicktime', '.webm': 'video/webm', '.mkv': 'video/x-matroska' }

/** The URL that returns the file itself rather than the share page */
function directDownloadUrl(shareUrl: string, source: 'gdrive' | 'dropbox'): string {
  const u = new URL(shareUrl)
  if (source === 'dropbox') {
    u.searchParams.delete('raw')
    u.searchParams.set('dl', '1')
    return u.toString()
  }
  // drive.google.com/file/d/<id>/view · …/open?id=<id> · …/uc?id=<id>
  const id = u.pathname.match(/\/file\/d\/([\w-]+)/)?.[1] ?? u.searchParams.get('id')
  if (!id) throw new UserFacingError('That Google Drive link doesn\'t point to a file. Open the video in Drive and copy its link.')
  // confirm=t skips Drive's "can't scan this large file for viruses" page
  return `https://drive.usercontent.google.com/download?id=${encodeURIComponent(id)}&export=download&confirm=t`
}

function filenameFromDisposition(header: string | null): string | null {
  if (!header) return null
  const star = header.match(/filename\*\s*=\s*(?:UTF-8'')?([^;]+)/i)?.[1]
  if (star) { try { return decodeURIComponent(star.trim().replace(/^"|"$/g, '')) } catch { /* fall through */ } }
  return header.match(/filename\s*=\s*"?([^";]+)"?/i)?.[1]?.trim() ?? null
}

async function downloadDirectLink(
  videoId: string, source: 'gdrive' | 'dropbox', tmp: string, maxBytes: number | undefined, signal?: AbortSignal,
): Promise<{ localPath: string; storagePath: string }> {
  const [video] = await db`SELECT source_url, user_id FROM videos WHERE id = ${videoId}`
  if (!video?.source_url) throw new Error('No source URL for video ' + videoId)
  const name = LINK_NAMES[source]
  const tooBig = () => new UserFacingError(`This video is bigger than your plan's ${Math.round((maxBytes ?? 0) / 1024 ** 3)} GB limit.`)

  await setProgress(videoId, 5)
  let res = await fetch(directDownloadUrl(video.source_url, source), { redirect: 'follow', signal })
  // Large Drive files can come back as a "can't scan this file for viruses" page whose button
  // carries the real download link (a form with hidden fields): follow it once
  if (source === 'gdrive' && res.ok && (res.headers.get('content-type') ?? '').includes('text/html')) {
    const page = await res.text()
    const action = page.match(/<form[^>]+id="download-form"[^>]+action="([^"]+)"/i)?.[1]
    if (action) {
      const next = new URL(action.replace(/&amp;/g, '&'))
      for (const [, name, value] of page.matchAll(/<input[^>]+type="hidden"[^>]+name="([^"]+)"[^>]+value="([^"]*)"/gi)) {
        next.searchParams.set(name, value.replace(/&amp;/g, '&'))
      }
      res = await fetch(next, { redirect: 'follow', signal })
    } else {
      // No download button: a sign-in or "you need access" page, i.e. the file isn't shared
      throw new UserFacingError('Google Drive didn\'t share the file. Set sharing to "Anyone with the link" and try again.')
    }
  }
  if (!res.ok || !res.body) {
    throw new UserFacingError(res.status === 404
      ? `That file couldn't be found on ${name}. Check the link.`
      : `Couldn't download the file from ${name} (error ${res.status}). Check the link is shared with "Anyone with the link".`)
  }
  const type = (res.headers.get('content-type') ?? '').split(';')[0].trim().toLowerCase()
  // A web page instead of the file: the link isn't public (or it's a folder)
  if (type.startsWith('text/html')) {
    res.body.cancel().catch(() => {})
    throw new UserFacingError(`${name} didn't share the file. Set sharing to "Anyone with the link" and try again.`)
  }
  const total = Number(res.headers.get('content-length')) || 0
  if (maxBytes && total > maxBytes) { res.body.cancel().catch(() => {}); throw tooBig() }

  const filename = filenameFromDisposition(res.headers.get('content-disposition'))
  const extFromName = filename?.match(/\.[a-z0-9]+$/i)?.[0].toLowerCase()
  const ext = (extFromName && VIDEO_TYPE_BY_EXT[extFromName]) ? extFromName : VIDEO_EXT_BY_TYPE[type]
  if (!ext) {
    res.body.cancel().catch(() => {})
    throw new UserFacingError('That link isn\'t a video file we support (MP4, MOV, MKV or WebM).')
  }

  // Stream to disk, counting bytes: stop as soon as it's bigger than the plan allows
  // (Content-Length can be missing), and report progress as 5–45 %
  const localPath = join(tmp, `video${ext}`)
  let received = 0, lastPct = 5
  const counter = new Transform({
    transform(chunk: Buffer, _enc, done) {
      received += chunk.length
      if (maxBytes && received > maxBytes) { done(tooBig()); return }
      if (total) {
        const pct = Math.round(5 + (received / total) * 40)
        if (pct !== lastPct) { lastPct = pct; setProgress(videoId, pct).catch(() => {}) }
      }
      done(null, chunk)
    },
  })
  await pipeline(Readable.fromWeb(res.body as unknown as WebReadableStream), counter, createWriteStream(localPath), { signal })

  // Make sure it really is a video before storing it
  const { stdout } = await execFileAsync('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', localPath], { signal })
    .catch(() => ({ stdout: '' }))
  if (!stdout.includes('video')) throw new UserFacingError('That file doesn\'t contain a video we can read.')

  await setProgress(videoId, 47)
  const storagePath = `raw/${video.user_id}/${videoId}${ext}`
  await r2UploadFile(storagePath, localPath, VIDEO_TYPE_BY_EXT[ext])
  // The file's own name becomes the video's title (unless the user already named it)
  const title = filename?.replace(/\.[^.]+$/, '').trim().slice(0, 120)
  if (title) await db`UPDATE videos SET title = COALESCE(title, ${title}) WHERE id = ${videoId}`
  await setProgress(videoId, 58)
  return { localPath, storagePath }
}

async function downloadWithYtDlp(videoId: string, tmp: string, signal?: AbortSignal): Promise<{ localPath: string; storagePath: string }> {
  const [video] = await db`SELECT source_url, user_id FROM videos WHERE id = ${videoId}`
  if (!video?.source_url) throw new Error('No source URL for video ' + videoId)

  const outputTemplate = join(tmp, 'video.%(ext)s')

  // ── Stage 1: Download (progress 5 → 45%) ──────────────────────────────────
  console.log(`[transcribe] yt-dlp downloading ${video.source_url}`)
  await setProgress(videoId, 5)

  const cookiesPath = process.env.YOUTUBE_COOKIES_PATH
  const ytdlpArgs = [
    '-f', 'bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best',
    '--merge-output-format', 'mp4',
    '-o', outputTemplate,
    '--no-playlist',
    '--newline',
    ...(cookiesPath ? ['--cookies', cookiesPath] : ['--extractor-args', 'youtube:player_client=android']),
    video.source_url!,
  ]

  await new Promise<void>((resolve, reject) => {
    const proc = spawn('yt-dlp', ytdlpArgs, { signal })
    proc.on('error', reject)

    let lastUpdate = 0
    proc.stdout.on('data', (chunk: Buffer) => {
      const lines = chunk.toString().split('\n')
      for (const line of lines) {
        const m = line.match(/\[download\]\s+([\d.]+)%/)
        if (m) {
          const dlPct = parseFloat(m[1])
          // Map yt-dlp 0-100% → overall progress 5-45%
          const overall = Math.round(5 + dlPct * 0.4)
          if (overall !== lastUpdate) {
            lastUpdate = overall
            setProgress(videoId, overall).catch(() => {})
          }
          process.stdout.write(`\r[transcribe] downloading ${dlPct.toFixed(1)}%   `)
        }
      }
    })
    const stderrLines: string[] = []
    proc.stderr.on('data', (chunk: Buffer) => {
      const text = chunk.toString()
      stderrLines.push(text)
      process.stderr.write(text)
    })
    proc.on('close', code => {
      process.stdout.write('\n')
      if (code === 0) resolve()
      else reject(new Error(`yt-dlp exited with code ${code}: ${stderrLines.join('').slice(0, 500)}`))
    })
  })

  const files = await readdir(tmp)
  const videoFile = files.find(f => /\.(mp4|mkv|webm|mov)$/.test(f))
  if (!videoFile) throw new Error('yt-dlp did not produce a video file')

  const rawPath = join(tmp, videoFile)

  // ── Stage 2: Remux to MP4 (progress 45 → 55%) ───────────────────────────────
  // Stream-copy video (no quality loss, no duration cap) and encode audio to AAC.
  // R2 has no meaningful size limit, so the old 30-min / 480p budget is gone.
  const { stdout: durOut } = await execFileAsync('ffprobe', [
    '-v', 'error', '-show_entries', 'format=duration',
    '-of', 'default=noprint_wrappers=1:nokey=1', rawPath,
  ])
  const sourceDurationSec = parseFloat(durOut.trim()) || 60

  console.log(`[transcribe] remuxing full video (${(sourceDurationSec / 60).toFixed(1)} min) to MP4…`)
  await setProgress(videoId, 47)

  const localPath = join(tmp, 'video_final.mp4')

  // Track ffmpeg progress via -progress pipe
  await new Promise<void>((resolve, reject) => {
    const proc = spawn('ffmpeg', [
      '-i', rawPath,
      '-c:v', 'copy',          // copy video stream — original quality, fast
      '-c:a', 'aac',
      '-b:a', '192k',
      '-movflags', '+faststart',
      '-progress', 'pipe:1',
      '-y', localPath,
    ], { signal })
    proc.on('error', reject)

    let outTime = 0
    proc.stdout.on('data', (chunk: Buffer) => {
      const text = chunk.toString()
      const m = text.match(/out_time_ms=(\d+)/)
      if (m) {
        outTime = parseInt(m[1]) / 1_000_000  // seconds
        const pct = Math.min(1, outTime / sourceDurationSec)
        // Map ffmpeg 0-100% → overall 47-55%
        const overall = Math.round(47 + pct * 8)
        setProgress(videoId, overall).catch(() => {})
      }
    })
    proc.stderr.on('data', () => {})
    proc.on('close', code => {
      if (code === 0) resolve()
      else reject(new Error(`ffmpeg exited with code ${code}`))
    })
  })

  // ── Stage 3: Upload to R2 (progress 55 → 58%) ────────────────────────────
  await setProgress(videoId, 55)
  const storagePath = `raw/${video.user_id}/${videoId}.mp4`

  console.log(`[transcribe] uploading to R2: ${storagePath}`)
  await r2UploadFile(storagePath, localPath, 'video/mp4')

  await setProgress(videoId, 58)
  return { localPath, storagePath }
}

interface SarvamWord {
  word: string; start: number; end: number; speaker?: string; confidence?: number
  /** In English letters (Tenglish, Hinglish…) and in the language's own script; null = none. Missing = not spelled yet */
  roman?: string | null
  native?: string | null
}
interface SarvamResponse { language_code: string; transcript?: string; words: SarvamWord[] }

// ── fetch with timeout ──────────────────────────────────────────────────────
async function fetchWithTimeout(url: string, init: RequestInit, timeoutMs: number): Promise<Response> {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), timeoutMs)
  try {
    return await fetch(url, { ...init, signal: ctrl.signal })
  } finally {
    clearTimeout(timer)
  }
}

// ── Gemini 3.5 Transcribe: words + word-level timestamps in one call ─────────
// Audio is sent in 5-minute chunks: keeps each request under the Tier-1 limit of
// 10k input tokens/min (~25 tokens per second of audio) and limits timestamp drift.
// Each chunk is uploaded via the Files API, transcribed verbatim, then deleted.
const GEMINI_API_KEY    = process.env.GEMINI_API_KEY
const GEMINI_BASE       = 'https://generativelanguage.googleapis.com'
const GEMINI_STT_MODEL  = 'gemini-3.5-transcribe'
const GEMINI_CHUNK_SEC  = 300
/**
 * Chunks of one audio read at the same time (TRANSCRIBE_PARALLEL). Google charges per second of
 * audio, so reading them together costs the same — but our Vertex quota for the Transcribe model
 * lets through about one 5-minute chunk a minute (2026-10-22, 70-min video: one at a time 14.5 min;
 * 6 at a time refused 120+ times and failed). Raise it once the quota is raised: at 6, an hour of
 * video reads in ~3–4 min.
 */
const READ_PARALLEL     = Math.max(1, Number(process.env.TRANSCRIBE_PARALLEL) || 1)
/** Seconds of audio read to detect the spoken language (see transcribeAudio) */
const PROBE_SEC         = 60
const GEMINI_MAX_TRIES  = 8
/** Tries for a request Google wrongly refuses (see geminiTranscribeChunk) */
const GEMINI_REFUSAL_TRIES = 3

// Our language codes (Sarvam style) → BCP-47 codes Gemini accepts
function toGeminiLang(code: string): string {
  return code === 'od-IN' ? 'or-IN' : code
}

async function geminiUploadAudio(buf: Buffer): Promise<{ name: string; uri: string }> {
  const start = await fetchWithTimeout(`${GEMINI_BASE}/upload/v1beta/files`, {
    method: 'POST',
    headers: {
      'x-goog-api-key': GEMINI_API_KEY!,
      'X-Goog-Upload-Protocol': 'resumable',
      'X-Goog-Upload-Command': 'start',
      'X-Goog-Upload-Header-Content-Length': String(buf.length),
      'X-Goog-Upload-Header-Content-Type': 'audio/wav',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ file: { display_name: `chai-cut-${Date.now()}` } }),
  }, 60_000)
  const uploadUrl = start.headers.get('x-goog-upload-url')
  if (!uploadUrl) throw new Error(`Gemini upload start failed ${start.status}: ${await start.text()}`)

  const up = await fetchWithTimeout(uploadUrl, {
    method: 'POST',
    headers: { 'X-Goog-Upload-Offset': '0', 'X-Goog-Upload-Command': 'upload, finalize' },
    body: new Uint8Array(buf),
  }, 180_000)
  const { file } = await up.json() as { file?: { name: string; uri: string; state?: string } }
  if (!file?.uri) throw new Error(`Gemini upload failed (${up.status})`)

  // Audio files are usually ACTIVE immediately; poll briefly if still processing
  for (let i = 0; i < 30 && file.state && file.state !== 'ACTIVE'; i++) {
    await new Promise(r => setTimeout(r, 2000))
    const s = await fetchWithTimeout(`${GEMINI_BASE}/v1beta/${file.name}`, { headers: { 'x-goog-api-key': GEMINI_API_KEY! } }, 30_000)
    file.state = ((await s.json()) as { state?: string }).state
  }
  return { name: file.name, uri: file.uri }
}

// ── Vertex AI (Google Cloud) ─────────────────────────────────────────────────
// With GOOGLE_CLOUD_PROJECT set, transcription goes to Gemini 3.5 Transcribe on Vertex AI instead
// of the Gemini API key: Vertex has no daily request cap (the API key's Tier 1 allows 100 a day).
// Auth: a service account key in GOOGLE_CREDENTIALS_JSON (the key file's contents, as JSON or
// base64 — Railway has env vars, not files), else Application Default Credentials
// (`gcloud auth application-default login` locally, or a GOOGLE_APPLICATION_CREDENTIALS file).
// The model is only served from the 'global' location; audio goes inline (5-minute chunks, well
// under the 15-minute limit).
// With the key set too and STT_SHORT_SEC set, both are used (sttSides): Vertex and the key each
// have their own limit, so together they read more at once.
const VERTEX_PROJECT   = process.env.GOOGLE_CLOUD_PROJECT
const VERTEX_LOCATION  = process.env.GOOGLE_CLOUD_LOCATION || 'global'
const VERTEX_STT_MODEL = 'gemini-3.5-transcribe-preview'
let vertexAuth: GoogleAuth | null = null

/** The service account key from GOOGLE_CREDENTIALS_JSON (raw JSON or base64), if set */
export function vertexCredentials(): Record<string, unknown> | undefined {
  const raw = process.env.GOOGLE_CREDENTIALS_JSON?.trim()
  if (!raw) return undefined
  try {
    return JSON.parse(raw.startsWith('{') ? raw : Buffer.from(raw, 'base64').toString('utf8'))
  } catch {
    throw new Error('GOOGLE_CREDENTIALS_JSON is set but is not a valid service account key (paste the whole JSON file)')
  }
}

export function vertexAuthClient(): GoogleAuth {
  const credentials = vertexCredentials()
  vertexAuth ??= new GoogleAuth({ scopes: ['https://www.googleapis.com/auth/cloud-platform'], ...(credentials ? { credentials } : {}) })
  return vertexAuth
}

// Vertex answers 429 when its shared capacity is busy for a moment (there is no daily cap). Every
// reading in this worker goes through one gate: after a 429 it lets half as many requests out at
// once, and more again as they go through, so readings (several jobs on one worker, or chunks read
// together) slow down together instead of each using up its retries.
const STT_MAX_IN_FLIGHT = Math.max(1, Number(process.env.STT_MAX_IN_FLIGHT) || 8)
/** Tries for a request Vertex refuses as busy (429): about 5 minutes of waiting in all */
const VERTEX_BUSY_TRIES = 14
let sttLimit = STT_MAX_IN_FLIGHT
let sttOpen = 0
let sttLastCut = 0
const sttWaiting: (() => void)[] = []
async function sttGate<T>(send: () => Promise<T>): Promise<T> {
  while (sttOpen >= Math.max(1, Math.floor(sttLimit))) await new Promise<void>(r => sttWaiting.push(r))
  sttOpen++
  try {
    return await send()
  } finally {
    sttOpen--
    for (const wake of sttWaiting.splice(0)) wake()
  }
}
/** Busy: half as many at once (once per few seconds: refusals of requests already out count once) */
function sttBusy() {
  if (Date.now() - sttLastCut < 5000) return
  sttLimit = Math.max(1, sttLimit / 2)
  sttLastCut = Date.now()
  console.warn(`[vertex] busy: at most ${Math.floor(sttLimit)} transcription request(s) at once for now`)
}
/** Went through: a little more again, back up to STT_MAX_IN_FLIGHT */
function sttFine() {
  sttLimit = Math.min(STT_MAX_IN_FLIGHT, sttLimit + 0.5)
}

async function vertexTranscribeChunk(buf: Buffer, languageCode?: string, handOver?: HandOver): Promise<SarvamWord[]> {
  const auth = vertexAuthClient()
  const host = VERTEX_LOCATION === 'global' ? 'aiplatform.googleapis.com' : `${VERTEX_LOCATION}-aiplatform.googleapis.com`
  const url = `https://${host}/v1/projects/${VERTEX_PROJECT}/locations/${VERTEX_LOCATION}/publishers/google/models/${VERTEX_STT_MODEL}:generateContent`
  // Verbatim (the default mode), word timestamps, and a speaker per stretch of speech, so
  // caption lines never mix two speakers
  const audioTranscriptionConfig: Record<string, unknown> = { wordTimestamp: true, diarization: true }
  if (languageCode) audioTranscriptionConfig.languageCodes = [toGeminiLang(languageCode)]
  const body = JSON.stringify({
    contents: [{ role: 'user', parts: [{ inlineData: { mimeType: 'audio/wav', data: buf.toString('base64') } }] }],
    generationConfig: { audioTranscriptionConfig },
  })

  for (let attempt = 1; ; attempt++) {
    let res: Response
    try {
      // Its turn in the shared Google line, and its share of the budget (each try is sent anew)
      if (!await waitForStt(wavSeconds(buf), 'vertex', handOver?.lineMs)) throw new SttHandOver('the Vertex line is long')
      const token = await auth.getAccessToken()
      res = await sttGate(() => fetchWithTimeout(url, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}`, 'x-goog-user-project': VERTEX_PROJECT!, 'Content-Type': 'application/json' },
        body,
      }, 300_000))
    } catch (e) {
      // The connection dropped before an answer came (not a timeout or a stop): try again, like a 5xx
      if (!isNetworkDrop(e) || attempt >= GEMINI_MAX_TRIES) throw e
      const delayMs = retryDelayMs(attempt)
      console.warn(`[vertex] connection dropped (${dropCode(e)}) on attempt ${attempt}, retrying in ${(delayMs / 1000).toFixed(1)}s`)
      await new Promise(r => setTimeout(r, delayMs))
      continue
    }
    if (res.status === 429) sttBusy()
    else if (res.ok) sttFine()
    if (res.status === 429 && handOver) throw new SttHandOver('Vertex is busy')
    if (res.ok) {
      const raw = await res.json() as {
        candidates?: { content?: { parts?: { audioTranscription?: { speakerLabel?: string; words?: { word?: string; startOffset?: string; endOffset?: string }[] } }[] } }[]
      }
      const words: SarvamWord[] = []
      for (const part of raw.candidates?.[0]?.content?.parts ?? []) {
        const tx = part.audioTranscription
        for (const w of tx?.words ?? []) {
          if (!w.word?.trim()) continue
          words.push({ word: w.word.trim(), start: parseFloat(w.startOffset ?? '0'), end: parseFloat(w.endOffset ?? '0'), speaker: tx?.speakerLabel })
        }
      }
      return words
    }
    const text = await res.text()
    // 429 on Vertex means shared capacity is busy for a moment (there is no daily cap): retry
    const retryable = res.status === 429 || res.status >= 500
    const tries = res.status === 429 ? VERTEX_BUSY_TRIES : GEMINI_MAX_TRIES
    if (!retryable || attempt >= tries) throw new Error(`Vertex transcribe ${res.status}: ${text.slice(0, 300)}`)
    // Spread out at random, so the many jobs a busy moment refused don't all come back together
    const delayMs = retryDelayMs(attempt, retryAfterMs(res.headers.get('retry-after')))
    console.warn(`[vertex] ${res.status} on attempt ${attempt}, retrying in ${(delayMs / 1000).toFixed(1)}s`)
    await new Promise(r => setTimeout(r, delayMs))
  }
}

/** Seconds from Gemini's "Please retry in 22h28m7s" / "retry in 13.5s" (0 when not given) */
function retryHintSec(body: string): number {
  const m = body.match(/retry in ((?:[\d.]+[hms])+)/i)
  if (!m) return 0
  let sec = 0
  for (const [, n, unit] of m[1].matchAll(/([\d.]+)([hms])/gi)) sec += parseFloat(n) * (unit.toLowerCase() === 'h' ? 3600 : unit.toLowerCase() === 'm' ? 60 : 1)
  return sec
}

// ── Vertex and the key together ──────────────────────────────────────────────
// With both set and STT_SHORT_SEC (seconds) set: a job with that much audio or less (a clip's
// captions) goes to Vertex first — one request, no upload — and to the key if Vertex's line is
// longer than STT_SPILL_SEC or Vertex refuses it as busy (Vertex lets through only ~2–3 two-minute
// clips a minute). Longer ones (a whole video) go to the key first (Tier 2: ~10× Vertex's limit) and
// to Vertex if the key refuses them as busy or its daily limit is used up. Not set: Vertex when
// GOOGLE_CLOUD_PROJECT is set, else the key, as before.
const STT_SHORT_SEC = Math.max(0, Number(process.env.STT_SHORT_SEC) || 0)
const STT_SPILL_SEC = Math.max(1, Number(process.env.STT_SPILL_SEC) || 15)
/** Until when the key's daily limit is used up: everything goes to Vertex till then */
let keyRestUntil = 0

/** A request may go to the other side: if its line is longer than lineMs, or if this side refuses it as busy */
interface HandOver { lineMs?: number }
/** This side can't take the request now: the other side reads it */
class SttHandOver extends Error {}

/** The sides a request goes to, in order (the second only when the first hands it over) */
function sttSides(): SttSide[] {
  if (!VERTEX_PROJECT) return ['key']
  if (!GEMINI_API_KEY || !STT_SHORT_SEC || Date.now() < keyRestUntil) return ['vertex']
  // (no job around it, e.g. filling a transcript's gaps: a whole video)
  const audioSec = sttTurn.getStore()?.audioSec
  return audioSec != null && audioSec <= STT_SHORT_SEC ? ['vertex', 'key'] : ['key', 'vertex']
}

async function geminiTranscribeChunk(buf: Buffer, languageCode?: string): Promise<SarvamWord[]> {
  const [first, then] = sttSides()
  const read = (side: SttSide, handOver?: HandOver) =>
    side === 'vertex' ? vertexTranscribeChunk(buf, languageCode, handOver) : keyTranscribeChunk(buf, languageCode, handOver)
  if (!then) return read(first)
  try {
    return await read(first, first === 'vertex' ? { lineMs: STT_SPILL_SEC * 1000 } : {})
  } catch (e) {
    if (!(e instanceof SttHandOver)) throw e
    console.warn(`[stt] ${e.message}: this ${Math.round(wavSeconds(buf))} s goes to ${then === 'vertex' ? 'Vertex' : 'the Gemini key'}`)
    try {
      return await read(then)
    } catch (e2) {
      // Sent on to the key and its daily limit was used up: back to Vertex, to wait there
      if (!(e2 instanceof SttHandOver)) throw e2
      console.warn(`[stt] ${e2.message}: this ${Math.round(wavSeconds(buf))} s goes back to Vertex`)
      return read('vertex')
    }
  }
}

async function keyTranscribeChunk(buf: Buffer, languageCode?: string, handOver?: HandOver): Promise<SarvamWord[]> {
  // A dropped connection (upload or request, not a timeout) is tried again, like a 5xx
  const dropped = async (e: unknown, attempt: number) => {
    if (!isNetworkDrop(e) || attempt >= GEMINI_MAX_TRIES) throw e
    const delayMs = retryDelayMs(attempt)
    console.warn(`[gemini] connection dropped (${dropCode(e)}) on attempt ${attempt}, retrying in ${(delayMs / 1000).toFixed(1)}s`)
    await new Promise(r => setTimeout(r, delayMs))
  }
  for (let attempt = 1; ; attempt++) {
    // Its turn in the shared Google line, and its share of the budget (each try is sent anew)
    if (!await waitForStt(wavSeconds(buf), 'key', handOver?.lineMs)) throw new SttHandOver('the Gemini key line is long')
    let file: { name: string; uri: string }
    try { file = await geminiUploadAudio(buf) } catch (e) { await dropped(e, attempt); continue }
    try {
      // diarization: each word gets a speaker label, so caption lines never mix two speakers
      const transcription_config: Record<string, unknown> = { mode: { type: 'verbatim', timestamp_granularities: ['word'], diarization_mode: 'speaker' } }
      if (languageCode) transcription_config.language_codes = [toGeminiLang(languageCode)]

      let res: Response
      try {
        res = await fetchWithTimeout(`${GEMINI_BASE}/v1beta/interactions`, {
          method: 'POST',
          headers: { 'x-goog-api-key': GEMINI_API_KEY!, 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: GEMINI_STT_MODEL,
            input: [{ type: 'audio', uri: file.uri, mime_type: 'audio/wav' }],
            generation_config: { transcription_config },
          }),
        }, 300_000)
      } catch (e) { await dropped(e, attempt); continue }

      if (res.ok) {
        const raw = await res.json() as {
          steps?: { content?: { annotations?: { type: string; text: string; start_offset: string; end_offset: string; speaker?: string }[] }[] }[]
        }
        const words: SarvamWord[] = []
        for (const step of raw.steps ?? []) {
          for (const c of step.content ?? []) {
            for (const a of c.annotations ?? []) {
              if (a.type !== 'word_info' || !a.text?.trim()) continue
              words.push({ word: a.text.trim(), start: parseFloat(a.start_offset), end: parseFloat(a.end_offset), speaker: a.speaker })
            }
          }
        }
        return words
      }

      const body = await res.text()
      // Google's own fault, on and off for hours on 2026-10-07: a correct request refused with
      // 400 "Thinking is not enabled for this model", while the same request works moments later.
      // One refused chunk used to fail a whole video: try it twice more first.
      const passing400 = res.status === 400 && /thinking is not enabled/i.test(body) && attempt < GEMINI_REFUSAL_TRIES
      const retryable = res.status === 429 || res.status >= 500 || passing400
      if (!retryable || attempt >= GEMINI_MAX_TRIES) {
        // Shown to the user: say whose side it is on, not Google's raw JSON
        const said = body.match(/"message":\s*"([^"]+)"/)?.[1] ?? body.slice(0, 200)
        throw new Error(`Google's transcription service is not answering right now (${res.status}: ${said}). Please try again later.`)
      }
      // Rate limited (10k tokens/min on Tier 1) — wait for the window to reset. A daily limit
      // ("retry in 22h28m") will not reset while the job waits: stop now and say so.
      const hinted = retryHintSec(body)
      if (hinted > 120) {
        // Vertex (when set) reads everything until then
        keyRestUntil = Date.now() + hinted * 1000
        if (VERTEX_PROJECT) throw new SttHandOver(`the Gemini key's limit for today is used up (for ${Math.round(hinted / 60)} min)`)
        const hours = Math.max(1, Math.round(hinted / 3600))
        throw new Error(`Transcription limit for today is used up (Gemini API). It resets in about ${hours} hour${hours === 1 ? '' : 's'}.`)
      }
      if (res.status === 429 && handOver) throw new SttHandOver('the Gemini key is busy')
      const delaySec = Math.max(hinted, res.status === 429 ? 20 : 5 * attempt)
      console.warn(`[gemini] ${res.status} on attempt ${attempt}, retrying in ${delaySec}s`)
      await new Promise(r => setTimeout(r, delaySec * 1000))
    } finally {
      fetchWithTimeout(`${GEMINI_BASE}/v1beta/${file.name}`, { method: 'DELETE', headers: { 'x-goog-api-key': GEMINI_API_KEY! } }, 30_000)
        .catch(() => {})
    }
  }
}

/** A whole-video reading as far as it got: the words of every chunk already read */
export interface Reading {
  v: 1
  totalSec: number
  chunkSec: number
  /** The language every chunk is read in; null = left on automatic (English) */
  lang: string | null
  /** Words of each finished chunk, by chunk number (times are within the whole audio) */
  chunks: Record<string, SarvamWord[]>
}
/** Where a reading is kept between runs (handleTranscribeJob keeps it in storage) */
export interface ReadingStore {
  load(): Promise<Reading | null>
  save(reading: Reading): Promise<void>
}

export async function transcribeAudio(audioPath: string, videoId?: string, languageCode?: string, signal?: AbortSignal, kept?: ReadingStore, onRead?: ReadProgress): Promise<SarvamResponse> {
  if (!GEMINI_API_KEY && !VERTEX_PROJECT) throw new Error('Set GOOGLE_CLOUD_PROJECT (Vertex AI) or GEMINI_API_KEY for transcription')

  const { stdout } = await execFileAsync('ffprobe', [
    '-v', 'error', '-show_entries', 'format=duration',
    '-of', 'default=noprint_wrappers=1:nokey=1', audioPath,
  ])
  const totalSec = parseFloat(stdout.trim())
  // (how long the job's audio is decides where its requests go: sttSides)
  const turn = sttTurn.getStore()
  if (turn) turn.audioSec ??= totalSec
  const sides = sttSides().map(s => s === 'vertex' ? `Vertex AI ${VERTEX_STT_MODEL}` : `Gemini key ${GEMINI_STT_MODEL}`)
  console.log(`[transcribe] ${sides.join(', then ')} (verbatim, word timestamps)`)
  const numChunks = Math.max(1, Math.ceil(totalSec / GEMINI_CHUNK_SEC))
  console.log(`[transcribe] audio ${totalSec.toFixed(1)}s → ${numChunks} chunk(s), ${Math.min(READ_PARALLEL, numChunks)} at a time`)
  const readStart = Date.now()
  const since = () => `${((Date.now() - readStart) / 1000).toFixed(1)} s`
  if (videoId) await setProgress(videoId, 62)

  // Chunk i of the audio; `skipSec` starts it part-way (its first seconds were already read)
  const chunkWords = (i: number, lang: string | undefined, skipSec = 0) =>
    rangeWords(i * GEMINI_CHUNK_SEC + skipSec, GEMINI_CHUNK_SEC - skipSec, lang, `${i}`)

  async function rangeWords(startSec: number, durSec: number, lang: string | undefined, tag: string): Promise<SarvamWord[]> {
    // A job that ran past its timeout stops here, between chunks
    signal?.throwIfAborted()
    const chunkPath = audioPath.replace('.wav', `_gchunk${tag}.wav`)
    await execFileAsync('ffmpeg', [
      '-i', audioPath, '-ss', String(startSec), '-t', String(durSec),
      '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', chunkPath,
    ])
    const buf = await readFile(chunkPath)
    await unlink(chunkPath).catch(() => {})
    if (buf.length < 16_000) return []  // < 0.5s of audio — nothing to transcribe

    const label = tag.startsWith('probe') ? `language ${tag}` : `chunk ${parseInt(tag, 10) + 1}/${numChunks}`
    let words = await geminiTranscribeChunk(buf, lang)
    console.log(`[gemini] ${label} (${Math.round(durSec)}s): ${words.length} words${lang ? ` (${lang})` : ''}`)
    // A stuck reading (one phrase over and over, more words than anyone can say) would fill the
    // transcript with the loop: read the chunk once more, and if that is wrong too, cut the loops
    const seconds = Math.min(durSec, Math.max(0, totalSec - startSec))
    const wrong = runawayReason(words, seconds)
    if (wrong) {
      console.warn(`[transcribe] ${label}: the reading looks stuck (${wrong}) — reading it again`)
      const again = await geminiTranscribeChunk(buf, lang)
      const stillWrong = runawayReason(again, seconds)
      if (!stillWrong) {
        console.log(`[transcribe] ${label}: second reading is fine (${again.length} words)`)
        words = again
      } else {
        const trimmed = trimRunaway(again.length <= words.length ? again : words, seconds)
        console.warn(`[transcribe] ${label}: still stuck (${stillWrong}) — loops cut: ${Math.min(words.length, again.length)} → ${trimmed.length} words`)
        words = trimmed
      }
    }
    // Locked to an Indian language, Gemini drops sentences spoken purely in English — re-read
    // the stretches it left empty as English and merge those words in
    if (lang && !lang.startsWith('en')) {
      const chunkSec = Math.min(durSec, Math.max(0, totalSec - startSec))
      const extra = await englishGapPass(chunkPath, audioPath, startSec, chunkSec, words)
      if (extra.length) {
        console.log(`[gemini] chunk ${parseInt(tag, 10) + 1}/${numChunks}: +${extra.length} English words from untranscribed gaps`)
        words = [...words, ...extra].sort((a, b) => a.start - b.start)
      }
    }
    // Gemini also skips stretches of speech inside a piece, at any length (lib/skippedSpeech.ts): most
    // are English sentences in Telugu or Hindi talk, which the English pass above brings back; each gap
    // still empty (with sound) is read again on its own, in the same language. (Not the language samples)
    if (!tag.startsWith('probe')) {
      const chunkSec = Math.min(durSec, Math.max(0, totalSec - startSec))
      const back = await skippedSpeech(chunkPath, audioPath, startSec, chunkSec, words, lang, label, signal)
      if (back.length) words = [...words, ...back].sort((a, b) => a.start - b.start)
    }
    // Speaker labels are only consistent within one request, so scope them to the chunk
    return words.map(w => ({ ...w, start: w.start + startSec, end: w.end + startSec, speaker: w.speaker ? `c${tag}:${w.speaker}` : undefined }))
  }

  // Auto-detect drops a lot of Telugu speech, so detect the language from the script of what
  // is said, then lock it for every chunk. Long audio: detect from 1-minute samples (the start,
  // then the middle if the start has no speech or only English) instead of captioning the whole
  // first 5-minute chunk twice. Short audio (a clip): one auto pass is the sample.
  let lang = (languageCode && languageCode !== 'unknown') ? languageCode : undefined
  // What an earlier run already read of this same audio, in the same language: those chunks
  // (and the language it found) are used as they are
  let earlier = kept ? await kept.load().catch(() => null) : null
  if (earlier && !(earlier.v === 1 && earlier.chunkSec === GEMINI_CHUNK_SEC && Math.abs(earlier.totalSec - totalSec) <= 1
    && earlier.chunks && typeof earlier.chunks === 'object' && (!lang || earlier.lang === lang))) earlier = null
  let langKnown = !!lang
  if (earlier) {
    lang = earlier.lang ?? undefined
    langKnown = true
    console.log(`[transcribe] continuing an earlier reading: ${Object.keys(earlier.chunks).length} of ${numChunks} chunk(s) already read`)
  }
  const allWords: SarvamWord[] = []
  let probe: SarvamWord[] | null = null   // short audio: the auto pass of chunk 1
  // Long audio: the 1-minute samples read on auto. If the language stays auto (English), their
  // words are kept and those minutes aren't sent again.
  const samples: { from: number; to: number; words: SarvamWord[] }[] = []
  if (!langKnown) {
    if (totalSec <= PROBE_SEC * 1.5) {
      probe = await chunkWords(0, undefined)
      lang = detectSarvamLang(probe.map(w => w.word).join(' '))
      // Read as English: what it left out may be Telugu, Hindi… (languageInGaps)
      if (!lang) lang = await languageInGaps(audioPath, 0, Math.min(totalSec, GEMINI_CHUNK_SEC), probe, 'probe')
    } else {
      const sample = async (from: number, tag: string) => {
        const words = await rangeWords(from, PROBE_SEC, undefined, tag)
        samples.push({ from, to: from + PROBE_SEC, words })
        return detectSarvamLang(words.map(w => w.word).join(' '))
      }
      lang = await sample(0, 'probe-start')
      // The start can be music or an English intro: check the middle before settling on auto
      if (!lang) lang = await sample(Math.max(PROBE_SEC, Math.min(totalSec / 2, totalSec - PROBE_SEC)), 'probe-middle')
      // Both read as English: what they left out may be Telugu, Hindi… (languageInGaps; the
      // samples' words are back at the audio's times, the gaps are found from their own)
      for (const [k, sm] of samples.entries()) {
        if (lang) break
        const rel = sm.words.map(w => ({ ...w, start: w.start - sm.from, end: w.end - sm.from }))
        lang = await languageInGaps(audioPath, sm.from, Math.min(PROBE_SEC, totalSec - sm.from), rel, `probe-${k + 1}`)
      }
    }
    if (lang) console.log(`[transcribe] detected language: ${lang} — locking for all chunks (${since()})`)
  }
  const readChunk = async (i: number): Promise<SarvamWord[]> => {
    if (i === 0 && probe && !lang) return probe  // English / Latin script: keep auto result
    const from = i * GEMINI_CHUNK_SEC, to = Math.min(totalSec, from + GEMINI_CHUNK_SEC)
    const inChunk = lang ? [] : samples.filter(sm => sm.to > from && sm.from < to)
    if (inChunk.length) {
      // Staying on auto: read only the parts of this chunk the samples didn't cover
      const got: SarvamWord[] = []
      let cursor = from, part = 0
      for (const sm of inChunk.sort((x, y) => x.from - y.from)) {
        if (sm.from - cursor >= 1) got.push(...await rangeWords(cursor, sm.from - cursor, undefined, `${i}-${part++}`))
        got.push(...sm.words.filter(w => w.start >= from && w.start < to))
        cursor = Math.max(cursor, sm.to)
      }
      if (to - cursor >= 1) got.push(...await rangeWords(cursor, to - cursor, undefined, `${i}-${part++}`))
      return got
    }
    const words = await chunkWords(i, lang)
    // Chunk 1 was transcribed twice (auto + locked) — keep whichever caught more speech
    return i === 0 && probe && probe.length > words.length ? probe : words
  }
  // A chunk's words in English letters and in the language's own script, as soon as it is read
  // (while the other chunks are still being read). A chunk left on auto is English.
  const spellChunk = async (words: SarvamWord[]): Promise<SarvamWord[]> => {
    if (!words.length) return words
    const code = lang ?? 'en-IN'
    const [roman, native] = await Promise.all([transliterateToRoman(words, code), toNativeScript(words, code)])
    return words.map((w, k) => ({ ...w, roman: roman[k] ?? null, native: native[k] ?? null }))
  }
  const spelled = (words: SarvamWord[]) => words.every(w => w.roman !== undefined)
  // Kept once the language is known and after every chunk. A save that fails only means that
  // chunk is read again next time. Chunks finish in any order: saves go one at a time.
  const reading: Reading = earlier ?? { v: 1, totalSec, chunkSec: GEMINI_CHUNK_SEC, lang: lang ?? null, chunks: {} }
  let saving = Promise.resolve()
  const keep = () => {
    const store = kept
    if (!store || numChunks < 2) return saving
    saving = saving.then(() => store.save(reading))
      .catch(e => console.warn('[transcribe] could not keep the reading so far:', e instanceof Error ? e.message : e))
    return saving
  }
  if (!earlier) await keep()
  // (telling how far it has got never stops the reading)
  const tell = async (done: number) => { try { await onRead?.(done, numChunks) } catch { /* only a progress note */ } }
  // Chunks an earlier run read and spelled are used as they are; one it read but didn't spell
  // (from before spelling was done per chunk) is only spelled
  const todo = Array.from({ length: numChunks }, (_, i) => i).filter(i => !reading.chunks[i] || !spelled(reading.chunks[i]))
  let done = numChunks - todo.length
  if (done) await tell(done)
  // READ_PARALLEL chunks at a time. After the first failure no new chunk starts; the ones
  // under way finish and are kept, so the next run carries on from there.
  let next = 0
  let failure: unknown = null
  const readNext = async () => {
    while (next < todo.length && !failure) {
      const i = todo[next++]
      try {
        const had = reading.chunks[i]
        const got = had ?? await readChunk(i)
        reading.chunks[i] = got
        try {
          reading.chunks[i] = await spellChunk(got)
        } catch (e) {
          console.warn(`[transcribe] chunk ${i + 1}/${numChunks}: spelling failed, words kept as read:`, e instanceof Error ? e.message : e)
        }
        await keep()
        done++
        if (videoId) await setProgress(videoId, Math.round(63 + (done / numChunks) * 33))
        await tell(done)
      } catch (e) {
        failure ??= e
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(READ_PARALLEL, todo.length) }, readNext))
  await saving
  if (failure) throw failure
  console.log(`[transcribe] ${todo.length} chunk(s) read and spelled after ${since()}`)
  for (let i = 0; i < numChunks; i++) allWords.push(...(reading.chunks[i] ?? []))
  const language = lang ?? (allWords.length ? 'en-IN' : 'unknown')

  // Gemini occasionally returns a word slightly out of order — keep the timeline monotonic
  allWords.sort((a, b) => a.start - b.start)

  // Keep Gemini's real word end times so each caption ends when the voice ends — the
  // old pipeline stretched every word to the next word's start (up to 4 s), which kept
  // lines on screen through pauses and hid speaker turns. Only remove overlaps.
  for (let i = 0; i < allWords.length; i++) {
    const w = allWords[i]
    const next = allWords[i + 1]
    let end = Math.max(w.end, w.start + 0.08)
    if (next && end > next.start) end = Math.max(w.start, next.start)
    allWords[i] = { ...w, end: Math.min(end, totalSec) }
  }

  return { language_code: language, words: allWords }
}

// ── Skipped speech ─────────────────────────────────────────────────────────────
// (lib/skippedSpeech.ts plans it.) Gaps of 3 s+ with sound in them, read again — each stretch on its
// own, never stitched together (stitched, Gemini skipped 9 of 13 again) — a few at a time, in rounds
// while they still bring speech back. A stretch that fails, or reads as a stuck loop, adds nothing:
// the piece keeps what it had.
const REREAD_AT_ONCE = 3

async function skippedSpeech(
  chunkPathHint: string, audioPath: string, chunkStartSec: number, chunkSec: number,
  words: SarvamWord[], lang: string | undefined, label: string, signal?: AbortSignal,
): Promise<SarvamWord[]> {
  // Round after round on what is still empty (REREAD_ROUNDS), while a round still brings speech back
  const back: SarvamWord[] = []
  for (let round = 1; round <= REREAD_ROUNDS; round++) {
    const got = await skippedSpeechRound(chunkPathHint, audioPath, chunkStartSec, chunkSec, [...words, ...back], lang,
      round > 1 ? `${label} (round ${round})` : label, round, signal)
    back.push(...got)
    if (got.length < REREAD_ROUND_MIN_WORDS) break
  }
  return back
}

async function skippedSpeechRound(
  chunkPathHint: string, audioPath: string, chunkStartSec: number, chunkSec: number,
  words: SarvamWord[], lang: string | undefined, label: string, round: number, signal?: AbortSignal,
): Promise<SarvamWord[]> {
  const gaps = skippedGaps(words, chunkSec)
  if (!gaps.length) return []
  // Pauses aren't read again: only gaps with sound in them (on an error finding the silences, all of them)
  const voiced = await voicedParts(audioPath, chunkStartSec, chunkSec, gaps)
  const sound = (a: number, b: number) => voiced.reduce((t, [x, y]) => t + Math.max(0, Math.min(b, y) - Math.max(a, x)), 0)
  const windows = rereadWindows(gaps.filter(([a, b]) => sound(a, b) >= REREAD_MIN_SOUND_SEC), chunkSec)
  if (!windows.length) return []
  const found: SarvamWord[][] = windows.map(() => [])
  let next = 0
  const readNext = async () => {
    while (next < windows.length) {
      const n = next++
      const win = windows[n]
      signal?.throwIfAborted()
      const path = chunkPathHint.replace('.wav', `_re${round}-${n}.wav`)
      try {
        // (-ss before -i: a WAV seeks exactly, without reading the audio before it)
        await execFileAsync('ffmpeg', ['-ss', String(chunkStartSec + win.from), '-t', String(win.to - win.from), '-i', audioPath,
          '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', path])
        const buf = await readFile(path)
        if (buf.length < 16_000) continue
        const read = await geminiTranscribeChunk(buf, lang)
        if (runawayReason(read, win.to - win.from)) continue
        // (its own speaker labels: they only mean something within this one reading)
        found[n] = wordsInGaps(win, read).map(w => ({ ...w, speaker: w.speaker ? `re${round}.${n}:${w.speaker}` : undefined }))
      } catch (e) {
        if (signal?.aborted) throw e
        console.warn(`[gemini] ${label}: reading a skipped stretch again failed, kept as it was:`, e instanceof Error ? e.message : e)
      } finally {
        await unlink(path).catch(() => {})
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(REREAD_AT_ONCE, windows.length) }, readNext))
  const back = found.flat()
  const gapSec = windows.reduce((t, w) => t + w.gaps.reduce((u, [a, b]) => u + b - a, 0), 0)
  console.log(`[gemini] ${label}: ${gapSec.toFixed(0)} s skipped in ${windows.length} stretch(es) read again → +${back.length} words`)
  return back
}

/**
 * A video's transcript read before the skipped-speech re-read (gaps_filled_at NULL; the caller has found
 * it covers the whole video — by its whole_video flag or by its words, as Make my clips judges it): its gaps of
 * speech are read now, the same way (skippedSpeech, 5-minute piece by piece, from the audio kept in
 * storage or else the video), spelled like the rest, and added to it. Make my clips calls it the
 * first time it uses such a transcript (user, 2026-10-31: "when first used" — quota only for videos
 * that are used). Done once per transcript: the first caller marks it; if the reading fails or is
 * stopped the mark comes off and the transcript is as it was. Returns the words added (0: nothing to
 * fill, or another job has it).
 */
export async function fillTranscriptGaps(videoId: string, signal?: AbortSignal): Promise<number> {
  const [t] = await db<{ id: string; language: string | null }[]>`
    UPDATE transcripts SET gaps_filled_at = now()
    WHERE id = (SELECT id FROM transcripts WHERE video_id = ${videoId} ORDER BY created_at DESC LIMIT 1)
      AND gaps_filled_at IS NULL
    RETURNING id, language
  `
  if (!t) return 0
  const t0 = Date.now()
  const tmp = await mkdtemp(join(tmpdir(), 'fillgaps-'))
  try {
    const [v] = await db<{ storage_path: string | null }[]>`SELECT storage_path FROM videos WHERE id = ${videoId}`
    if (!v?.storage_path) throw new Error('the video has no file')
    // The audio: kept in storage (FLAC) since it was read, else from the video itself
    const audioPath = join(tmp, 'audio.wav')
    const toWav = (input: string) => execFileAsync('ffmpeg', ['-i', input, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', audioPath], { signal })
    try {
      const flac = join(tmp, 'kept.flac')
      await writeFile(flac, await r2Download(audioKeyFor(v.storage_path)))
      await toWav(flac)
    } catch (e) {
      if (signal?.aborted) throw e
      await toWav(await r2SignedUrl(v.storage_path, 4 * 3600))
    }
    const { stdout } = await execFileAsync('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', audioPath])
    const totalSec = parseFloat(stdout.trim())
    const rows = await db<{ start_ms: number; end_ms: number }[]>`SELECT start_ms, end_ms FROM transcript_words WHERE transcript_id = ${t.id} ORDER BY start_ms`
    // Read in its own language, as it was (English: automatic)
    const lang = t.language && t.language !== 'unknown' && !t.language.startsWith('en') ? t.language : undefined
    const pieces = Math.max(1, Math.ceil(totalSec / GEMINI_CHUNK_SEC))
    const found: SarvamWord[] = []
    for (let i = 0; i < pieces; i++) {
      const from = i * GEMINI_CHUNK_SEC, sec = Math.min(GEMINI_CHUNK_SEC, totalSec - from)
      // (a word running on from the piece before counts: no gap there)
      const inPiece = rows.filter(r => r.end_ms > from * 1000 && r.start_ms < (from + sec) * 1000)
        .map(r => ({ word: '', start: r.start_ms / 1000 - from, end: r.end_ms / 1000 - from }))
      const back = await skippedSpeech(join(tmp, `piece${i}.wav`), audioPath, from, sec, inPiece, lang, `filling ${videoId} piece ${i + 1}/${pieces}`, signal)
      found.push(...back.map(w => ({ ...w, start: w.start + from, end: w.end + from, speaker: w.speaker ? `f${i}:${w.speaker}` : undefined })))
    }
    if (found.length) {
      // Spelled like the rest of the transcript: in its own script, and in English letters
      const code = t.language && t.language !== 'unknown' ? t.language : 'en-IN'
      const [roman, native] = await Promise.all([transliterateToRoman(found, code), toNativeScript(found, code)])
      await db.begin(async tx => {
        const add = found.map((w, k) => ({
          transcript_id: t.id, word: native[k] ?? w.word, word_roman: roman[k] ?? null,
          start_ms: Math.round(w.start * 1000), end_ms: Math.round(w.end * 1000), speaker_id: w.speaker ?? null, confidence: null,
        }))
        for (let i = 0; i < add.length; i += 500) await tx`INSERT INTO transcript_words ${tx(add.slice(i, i + 500))}`
        // (the editor's "since" check sees new words)
        await tx`UPDATE transcripts SET created_at = now() WHERE id = ${t.id}`
      })
    }
    console.log(`[transcribe] video ${videoId}: skipped speech filled in — +${found.length} words in ${((Date.now() - t0) / 1000).toFixed(0)} s`)
    return found.length
  } catch (e) {
    await db`UPDATE transcripts SET gaps_filled_at = NULL WHERE id = ${t.id}`.catch(() => {})
    throw e
  } finally {
    await rm(tmp, { recursive: true, force: true }).catch(() => {})
  }
}

// ── English re-pass for gaps ──────────────────────────────────────────────────
// With language_codes locked to e.g. te-IN (auto-detect drops a lot of Telugu speech), a
// sentence spoken entirely in English comes back with no words at all. Find the stretches of
// the chunk with no words, stitch them into one clip (short silence between), transcribe that
// once as English, and map the words back to their real times.
const GAP_MIN_SEC = 1.8
const GAP_PAD_SEC = 0.15
const GAP_SEPARATOR_SEC = 0.6

/**
 * The other way round: on automatic (the start sounded English) Gemini drops a lot of Telugu, so
 * a clip that opens in English and goes on in Telugu came back with only its English words. The
 * stretches it left empty (with sound) are read on their own, on automatic; if what is said there
 * is in an Indian script, that is the language — the caller locks it, and the English gap pass
 * above then brings the English sentences back.
 */
async function languageInGaps(audioPath: string, startSec: number, durSec: number, words: SarvamWord[], tag: string): Promise<string | undefined> {
  const found = await gapPass(audioPath.replace('.wav', `_lgap${tag}.wav`), audioPath, startSec, durSec, words, 'auto')
  const lang = detectSarvamLang(found.map(w => w.word).join(' '))
  console.log(`[transcribe] ${tag}: what the English reading left out — ${found.length} words${lang ? `, in ${lang}` : ', none in an Indian script'}`)
  return lang
}

async function englishGapPass(chunkPathHint: string, audioPath: string, chunkStartSec: number, chunkSec: number, words: SarvamWord[]): Promise<SarvamWord[]> {
  return gapPass(chunkPathHint, audioPath, chunkStartSec, chunkSec, words, 'en-IN')
}

/** The stretches of a chunk with no words (and sound), read in `lang` ('auto' = automatic) and placed back at their times */
async function gapPass(chunkPathHint: string, audioPath: string, chunkStartSec: number, chunkSec: number, words: SarvamWord[], lang: string): Promise<SarvamWord[]> {
  const gaps: [number, number][] = []
  let cursor = 0
  for (const w of [...words].sort((a, b) => a.start - b.start)) {
    if (w.start - cursor >= GAP_MIN_SEC) gaps.push([cursor, w.start])
    cursor = Math.max(cursor, w.end)
  }
  if (chunkSec - cursor >= GAP_MIN_SEC) gaps.push([cursor, chunkSec])
  if (!gaps.length) return []
  // Most gaps are pauses: drop their silent parts so only stretches with sound are paid for
  const voiced = await voicedParts(audioPath, chunkStartSec, chunkSec, gaps)
  if (voiced.length < gaps.length || voiced.some((v, k) => v[1] - v[0] < gaps[k][1] - gaps[k][0])) {
    const before = gaps.reduce((t, [a, b]) => t + b - a, 0), after = voiced.reduce((t, [a, b]) => t + b - a, 0)
    console.log(`[gemini] ${lang === 'auto' ? 'Language' : 'English'} gap pass: ${after.toFixed(0)}s with sound of ${before.toFixed(0)}s of gaps`)
  }
  gaps.splice(0, gaps.length, ...voiced)
  if (!gaps.length) return []

  // Pieces of the full audio (absolute times) and where each lands in the stitched clip
  const pieces = gaps.map(([a, b]) => ({ from: Math.max(0, a - GAP_PAD_SEC), to: Math.min(chunkSec, b + GAP_PAD_SEC) }))
  let at = 0
  const placed = pieces.map(p => { const out = { ...p, at }; at += (p.to - p.from) + GAP_SEPARATOR_SEC; return out })
  const gapPath = chunkPathHint.replace('.wav', '_gaps.wav')
  const fc: string[] = []
  placed.forEach((p, k) => {
    fc.push(`[0:a]atrim=start=${(chunkStartSec + p.from).toFixed(3)}:end=${(chunkStartSec + p.to).toFixed(3)},asetpts=PTS-STARTPTS[g${k}]`)
    fc.push(`aevalsrc=0:d=${GAP_SEPARATOR_SEC}:s=16000[z${k}]`)
  })
  fc.push(`${placed.map((_, k) => `[g${k}][z${k}]`).join('')}concat=n=${placed.length * 2}:v=0:a=1[out]`)
  try {
    await execFileAsync('ffmpeg', ['-i', audioPath, '-filter_complex', fc.join(';'), '-map', '[out]',
      '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', gapPath])
    const buf = await readFile(gapPath)
    if (buf.length < 16_000) return []
    const found = await geminiTranscribeChunk(buf, lang === 'auto' ? undefined : lang)
    const out: SarvamWord[] = []
    for (const w of found) {
      const p = placed.find(x => w.start >= x.at - 0.05 && w.start < x.at + (x.to - x.from))
      if (!p) continue // landed in a separator
      const start = p.from + (w.start - p.at)
      const end = Math.min(p.to, p.from + (w.end - p.at))
      // Only keep words inside a real gap, so nothing overlaps what the main pass found
      if (!gaps.some(([a, b]) => start >= a - GAP_PAD_SEC && start < b)) continue
      out.push({ ...w, start, end: Math.max(end, start + 0.08), speaker: w.speaker ? `${lang === 'auto' ? 'g' : 'en'}:${w.speaker}` : undefined })
    }
    return out
  } catch (e) {
    console.warn(`[gemini] ${lang === 'auto' ? 'Language' : 'English'} gap pass failed, keeping the main pass only:`, e)
    return []
  } finally {
    await unlink(gapPath).catch(() => {})
  }
}

/**
 * The parts of each gap (chunk-relative seconds) that are not silence, using FFmpeg's
 * silencedetect on the chunk. Blips shorter than VOICED_MIN_SEC are dropped.
 * On any error the gaps are returned unchanged, so this can only save audio, never lose words.
 */
const SILENCE_DB = -40
const VOICED_MIN_SEC = 0.6
async function voicedParts(audioPath: string, chunkStartSec: number, chunkSec: number, gaps: [number, number][]): Promise<[number, number][]> {
  let silences: [number, number][]
  try {
    const { stderr } = await execFileAsync('ffmpeg', ['-hide_banner', '-nostats', '-ss', String(chunkStartSec), '-t', String(chunkSec), '-i', audioPath,
      '-af', `silencedetect=noise=${SILENCE_DB}dB:d=0.5`, '-f', 'null', '-'])
    silences = []
    let open: number | null = null
    for (const line of stderr.split('\n')) {
      const s0 = line.match(/silence_start: (-?[\d.]+)/)
      const s1 = line.match(/silence_end: (-?[\d.]+)/)
      if (s0) open = Math.max(0, parseFloat(s0[1]))
      if (s1 && open !== null) { silences.push([open, parseFloat(s1[1])]); open = null }
    }
    if (open !== null) silences.push([open, chunkSec])
  } catch {
    return gaps
  }
  const out: [number, number][] = []
  for (const [a, b] of gaps) {
    let pieces: [number, number][] = [[a, b]]
    for (const [sa, sb] of silences) {
      pieces = pieces.flatMap(([pa, pb]): [number, number][] => {
        if (sb <= pa || sa >= pb) return [[pa, pb]]
        return [[pa, Math.max(pa, sa)], [Math.min(pb, sb), pb]].filter(([x, y]) => y > x) as [number, number][]
      })
    }
    // A short English word ("okay", "yes") is ~0.5 s, so keep any sound from 0.6 s up
    out.push(...pieces.filter(([x, y]) => y - x >= VOICED_MIN_SEC))
  }
  return out
}

// ── Rule-based Telugu → Roman transliterator ──────────────────────────────────
// Maps Unicode codepoints directly — no API, deterministic Tenglish output.
const TE_CONSONANTS: Record<string, string> = {
  'క':'k','ఖ':'kh','గ':'g','ఘ':'gh','ఙ':'ng',
  'చ':'ch','ఛ':'chh','జ':'j','ఝ':'jh','ఞ':'ny',
  'ట':'t','ఠ':'th','డ':'d','ఢ':'dh','ణ':'n',
  'త':'t','థ':'th','ద':'d','ధ':'dh','న':'n',
  'ప':'p','ఫ':'ph','బ':'b','భ':'bh','మ':'m',
  'య':'y','ర':'r','ల':'l','వ':'v',
  'శ':'sh','ష':'sh','స':'s','హ':'h',
  'ళ':'l','ఱ':'r',  // ళ → 'l' (doubled naturally when clusters: ళ+్+ళ = 'll')
}
const TE_VOWELS: Record<string, string> = {
  'అ':'a','ఆ':'aa','ఇ':'i','ఈ':'ee',
  'ఉ':'u','ఊ':'oo','ఋ':'ru',
  'ఎ':'e','ఏ':'e','ఐ':'ai',
  'ఒ':'o','ఓ':'o','ఔ':'au',
}
// Vowel signs (matras) — excludes anusvara/visarga which are handled separately
const TE_VOWEL_SIGNS: Record<string, string> = {
  'ా':'aa','ి':'i','ీ':'ee','ు':'u','ూ':'oo','ృ':'ru',
  'ె':'e','ే':'e','ై':'ai','ొ':'o','ో':'o','ౌ':'au',
  '్':'',  // virama — suppress inherent vowel (handled inline)
  'ఁ':'',
}

function transliterateTeluguWord(word: string): string {
  const chars = [...word]
  let out = ''
  let i = 0

  const peek = (offset = 1) => chars[i + offset] ?? ''

  while (i < chars.length) {
    const c = chars[i]

    if (TE_CONSONANTS[c]) {
      const base = TE_CONSONANTS[c]
      const next = peek()
      if (next === '్') {
        // Virama: pure consonant cluster, no inherent vowel
        out += base
        i += 2
      } else if (next in TE_VOWEL_SIGNS) {
        // Explicit vowel matra
        out += base + TE_VOWEL_SIGNS[next]
        i += 2
        // Anusvara/visarga after the matra
        if (peek(0) === 'ం') { out += 'm'; i++ }
        else if (peek(0) === 'ః') { out += 'h'; i++ }
      } else {
        // Inherent vowel 'a'
        out += base + 'a'
        i++
        // Anusvara/visarga after inherent 'a'
        if (peek(0) === 'ం') { out += 'm'; i++ }
        else if (peek(0) === 'ః') { out += 'h'; i++ }
      }
    } else if (TE_VOWELS[c]) {
      out += TE_VOWELS[c]
      i++
      if (peek(0) === 'ం') { out += 'm'; i++ }
      else if (peek(0) === 'ః') { out += 'h'; i++ }
    } else if (c === 'ం') {
      out += 'm'; i++  // standalone anusvara
    } else if (c === 'ః') {
      out += 'h'; i++
    } else if (c in TE_VOWEL_SIGNS) {
      out += TE_VOWEL_SIGNS[c]; i++
    } else if (/[a-zA-Z0-9\s.,!?'-]/.test(c)) {
      out += c; i++
    } else {
      i++
    }
  }
  return out.toLowerCase()
}

// ── Transliteration: Indian script → Roman (Tenglish / Hinglish / etc.) ─────
// Maps short language codes (te, hi, ta…) to our xx-IN format
const WHISPER_TO_SARVAM: Record<string, string> = {
  'te': 'te-IN', 'hi': 'hi-IN', 'ta': 'ta-IN', 'kn': 'kn-IN',
  'ml': 'ml-IN', 'bn': 'bn-IN', 'gu': 'gu-IN', 'mr': 'mr-IN',
  'pa': 'pa-IN', 'or': 'od-IN',
}

const LANG_NAMES: Record<string, string> = {
  'te': 'Telugu', 'te-IN': 'Telugu',
  'hi': 'Hindi',  'hi-IN': 'Hindi',
  'ta': 'Tamil',  'ta-IN': 'Tamil',
  'kn': 'Kannada','kn-IN': 'Kannada',
  'ml': 'Malayalam','ml-IN': 'Malayalam',
  'bn': 'Bengali','bn-IN': 'Bengali',
  'gu': 'Gujarati','gu-IN': 'Gujarati',
  'mr': 'Marathi','mr-IN': 'Marathi',
  'pa': 'Punjabi','pa-IN': 'Punjabi',
  'or': 'Odia',  'od-IN': 'Odia',
}

function detectSarvamLang(text: string): string | undefined {
  if (/[ఀ-౿]/.test(text)) return 'te-IN'
  if (/[ऀ-ॿ]/.test(text)) return 'hi-IN'
  if (/[஀-௿]/.test(text)) return 'ta-IN'
  if (/[ಀ-೿]/.test(text)) return 'kn-IN'
  if (/[ഀ-ൿ]/.test(text)) return 'ml-IN'
  if (/[ঀ-৿]/.test(text)) return 'bn-IN'
  if (/[઀-૿]/.test(text)) return 'gu-IN'
  if (/[਀-੿]/.test(text)) return 'pa-IN'
  return undefined
}

/** Gemini's answer for the script conversions: one string per input word */
const STRING_LIST = { type: 'ARRAY', items: { type: 'STRING' } }

// Gemini batch transliteration → natural Tenglish / Hinglish (how people type it on
// WhatsApp/YouTube), English loanwords in normal English spelling. One output per
// input word, same order; words it couldn't convert are left undefined so the caller
// falls back to the rule-based Telugu transliterator for those words.
async function transliterateWithGemini(
  entries: SarvamWord[],
  languageCode: string,
): Promise<(string | undefined)[]> {
  const langName = LANG_NAMES[languageCode] ?? 'Indian'
  const style = langName === 'Telugu' ? 'Tenglish' : langName === 'Hindi' ? 'Hinglish' : `romanized ${langName}`
  const system =
`You convert ${langName} caption words into ${style}: the way ${langName} speakers write their language in English letters on WhatsApp or YouTube comments.
Rules:
- Exactly one output string per input word, same order. Never merge, split, drop or add words.
- Use common, natural spellings (e.g. నుంచి → "nunchi", నేను → "nenu", ఏమైంది → "emaindi", మనదే → "manade"), not academic transliteration and no diacritics.
- English words written in ${langName} script get their normal English spelling (ఇట్స్ → "it's", ట్రూ → "true", కాటన్ → "cotton", యాక్టర్స్ → "actors", యాడ్ → "ad").
- Numbers and words already in English letters stay as they are. Drop trailing punctuation.
- Lowercase, except names and the pronoun "I".`
  const answers = await convertWordForWord(
    entries.map(e => e.word.trim()),
    // gemini-2.5-flash writes the most natural Tenglish in tests
    batch => askGemini(JSON.stringify(batch), { system, schema: STRING_LIST, temperature: 0, thinkingBudget: 0, timeoutMs: 60_000 }),
    '[transliterate]',
  )
  const result = answers.map(v => (typeof v === 'string' && v.trim() ? stripDiacritics(v.trim()).replace(/[.,!?।]+$/g, '') : undefined))
  console.log(`[transliterate] ${languageCode} → ${style} via Gemini for ${entries.length} word(s)`)
  return result
}

/** Script ranges of the languages LANG_NAMES covers — a word containing any of these is already native */
const NATIVE_SCRIPT = /[\u0900-\u0D7F\u0B00-\u0B7F]/
const LATIN = /[A-Za-z]/

/**
 * Words in English letters inside a regional-language transcript (English phrases the speaker
 * used, or sentences recovered by the English gap pass), written in that language's script the
 * way its speakers spell English words: so → సో, think → థింక్. Returns one entry per input
 * word: the native spelling, or undefined to keep the word as it is. Trailing punctuation is
 * kept, since captions break lines at sentence ends.
 */
export async function toNativeScript(entries: SarvamWord[], languageCode: string): Promise<(string | undefined)[]> {
  const out: (string | undefined)[] = new Array(entries.length).fill(undefined)
  const langName = LANG_NAMES[languageCode]
  if (!langName || !aiProvider()) return out
  const todo = entries
    .map((e, i) => ({ i, word: e.word.trim() }))
    .filter(x => LATIN.test(x.word) && !NATIVE_SCRIPT.test(x.word))
  if (!todo.length) return out

  // Punctuation stays outside the conversion and goes back on afterwards
  const split = todo.map(x => {
    const m = x.word.match(/^(.*?)([.,!?।…:;"')\]]*)$/)!
    return { core: m[1], tail: m[2] }
  })
  const system =
`You write English words in ${langName} script, the way ${langName} captions and news write English words spoken in a ${langName} video (phonetically, as ${langName} speakers pronounce them).
Rules:
- Exactly one output string per input word, same order. Never merge, split, drop or add words.
- Spell by sound in ${langName} script${langName === 'Telugu' ? ' (so → "సో", think → "థింక్", actually → "యాక్చువల్లీ", I → "ఐ", it\'s → "ఇట్స్", brand → "బ్రాండ్", YouTube → "యూట్యూబ్")' : ''}.
- Numbers stay as digits. Keep capital-letter brand names readable by sound. No English letters in the output unless the input is only digits.`
  const answers = await convertWordForWord(
    split.map(x => x.core),
    batch => askGemini(JSON.stringify(batch), { system, schema: STRING_LIST, temperature: 0, thinkingBudget: 0, timeoutMs: 60_000 }),
    '[native-script]',
  )
  todo.forEach((x, j) => {
    const v = answers[j]
    // Only take real conversions; anything still in English letters keeps the original
    if (typeof v === 'string' && v.trim() && NATIVE_SCRIPT.test(v) && !LATIN.test(v)) out[x.i] = v.trim() + split[j].tail
  })
  console.log(`[native-script] ${todo.length} English word(s) → ${langName} script`)
  return out
}

// Strip IAST diacritics → plain ASCII so captions read as normal English letters.
function stripDiacritics(s: string): string {
  return s.normalize('NFD')
    .replace(/[̀-ͯ]/g, '')  // remove all combining marks
    .replace(/ḍ/g, 'd').replace(/ṭ/g, 't').replace(/ṇ/g, 'n')
    .replace(/ṣ/g, 's').replace(/ś/g, 'sh').replace(/ñ/g, 'n')
    .replace(/ṅ/g, 'ng').replace(/ḷ/g, 'l').replace(/ṃ/g, 'm').replace(/ḥ/g, 'h')
    .replace(/Ḍ/g, 'D').replace(/Ṭ/g, 'T')
}

async function transliterateToRoman(
  entries: SarvamWord[],
  languageCode: string,
): Promise<(string | undefined)[]> {
  // Resolve to a canonical lang code for lookup/detection
  const resolvedLang =
    (languageCode.includes('-') ? languageCode : WHISPER_TO_SARVAM[languageCode]) ??
    detectSarvamLang(entries.map(e => e.word).join(' '))

  if (!resolvedLang) return entries.map(() => undefined)
  const lang = resolvedLang

  // English is already Roman — return words directly, no API needed
  if (lang.startsWith('en')) {
    return entries.map(e => {
      const word = e.word.trim().toLowerCase().replace(/[.,!?।]/g, '')
      return word || undefined
    })
  }

  // Built-in rules for Telugu. Other languages have no fallback: a word Gemini couldn't
  // convert gets no Roman spelling.
  const fallback = async (): Promise<(string | undefined)[]> => {
    if (lang === 'te-IN' || lang === 'te') {
      return entries.map(e => {
        const word = e.word.trim()
        if (!word || !/[ఀ-౿]/.test(word)) return /[a-zA-Z]/.test(word) ? word.toLowerCase() : undefined
        const roman = transliterateTeluguWord(word).replace(/[.,!?।]/g, '').trim()
        return roman || undefined
      })
    }
    return entries.map(() => undefined)
  }

  // Gemini gives natural Tenglish/Hinglish (e.g. "nunchi", not "numchi"); words it
  // couldn't convert are filled from the fallback
  if (aiProvider()) {
    const viaGemini = await transliterateWithGemini(entries, lang)
    if (viaGemini.every(Boolean)) return viaGemini
    const fb = await fallback()
    return viaGemini.map((r, i) => r ?? fb[i])
  }
  return fallback()
}
