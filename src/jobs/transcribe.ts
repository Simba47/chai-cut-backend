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
import { r2, R2_BUCKET, r2DownloadToFile, r2UploadFile } from '../r2.js'
import db from '../db.js'
import type { Job, TranscribeJobPayload } from '../types.js'

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

export async function handleTranscribeJob(job: Job, signal?: AbortSignal) {
  const raw = job.payload
  const payload = (typeof raw === 'string' ? JSON.parse(raw) : raw) as TranscribeJobPayload
  const isLinkJob = !payload.storage_path
  const isRetranscribe = !!payload.is_retranscribe
  const isClipJob = !!payload.clip_id && payload.clip_start_ms !== undefined && payload.clip_end_ms !== undefined
  // Full-video captions right after upload/download, so clips open with captions ready.
  // The video is marked ready first — clipping never waits on this background transcription.
  const isFullJob = !!payload.transcribe_full && !isClipJob && !isRetranscribe
  let videoReady = false

  // Plain upload jobs (already in storage, no clip): mark ready. Without transcribe_full
  // (older jobs) transcription is deferred to clip creation.
  if (!isLinkJob && !isClipJob && !isRetranscribe) {
    await db`UPDATE videos SET status = 'ready', download_progress = 100 WHERE id = ${payload.video_id}`
    videoReady = true
    await fillMissingDuration(payload.video_id, payload.storage_path, signal)
      .catch(e => console.warn('[transcribe] could not read video length:', e))
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

  try {
    let videoPath = ''
    let storagePath = payload.storage_path

    const audioPath = join(tmp, 'audio.wav')

    // ── Fast path for retranscription: download cached FLAC audio ────────────
    const audioStoragePath = storagePath ? storagePath.replace(/\.[^.]+$/, '_audio.flac') : null
    let audioCached = false
    if (isRetranscribe && audioStoragePath) {
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
        console.log(`[transcribe] retranscribe: used cached audio (${audioStoragePath})`)
      } catch { /* fall through to full video download */ }
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
          if (!isFullJob) {
            console.log(`[transcribe] link video ${payload.video_id} ready — no captions on this plan`)
            return
          }
          console.log(`[transcribe] link video ${payload.video_id} ready — captioning full video in background`)
        }
      } else {
        // Clip or retranscribe job: download video from storage (streamed to disk — a whole
        // video held in memory could run the worker out of it)
        await setProgress(payload.video_id, 30)
        videoPath = join(tmp, 'video.mp4')
        await r2DownloadToFile(storagePath, videoPath, signal)
        await setProgress(payload.video_id, 55)
      }

      // Extract audio — for clip jobs, seek to clip range only (fast & cheap)
      await setProgress(payload.video_id, 60)
      await requireSound(videoPath, signal)
      if (isClipJob) {
        const startSec = payload.clip_start_ms! / 1000
        const durSec   = (payload.clip_end_ms! - payload.clip_start_ms!) / 1000
        await execFileAsync('ffmpeg', [
          '-ss', String(startSec), '-i', videoPath,
          '-t', String(durSec),
          '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', audioPath,
        ], { signal })
      } else {
        await execFileAsync('ffmpeg', [
          '-i', videoPath, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', audioPath,
        ], { signal })
        // Cache full-video audio as FLAC for fast retranscription
        const cachePath = audioStoragePath ?? (storagePath ? storagePath.replace(/\.[^.]+$/, '_audio.flac') : null)
        if (cachePath) {
          const flacOut = join(tmp, 'audio_cache.flac')
          execFileAsync('ffmpeg', ['-i', audioPath, '-compression_level', '5', '-y', flacOut])
            .then(() => readFile(flacOut))
            .then(buf => r2Upload(cachePath, buf, 'audio/flac'))
            .catch(e => console.warn('[transcribe] audio cache upload failed:', e))
        }
      }
    }

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
    const sarvamResult = await transcribeAudio(audioPath, (!isRetranscribe && !isClipJob && !videoReady) ? payload.video_id : undefined, languageCode, signal, kept)
    signal?.throwIfAborted()

    // Clip jobs (first transcription or retranscribe): replace only this clip's time range in the
    // video's transcript, so other clips of the same video keep their captions. The editor reads
    // one transcript per video (the newest), so a clip job must never start a separate one —
    // that hid every other clip's captions. created_at is bumped so the editor's "since" poll
    // picks up the new words.
    const [existing] = isClipJob
      ? await db`
          SELECT t.id FROM transcripts t
          WHERE t.video_id = ${payload.video_id}
            AND (t.whole_video OR EXISTS (SELECT 1 FROM transcript_words w WHERE w.transcript_id = t.id))
          ORDER BY t.created_at DESC LIMIT 1`
      : []
    let transcript: { id: string }
    if (existing) {
      await db`DELETE FROM transcript_words WHERE transcript_id = ${existing.id} AND start_ms >= ${payload.clip_start_ms!} AND start_ms < ${payload.clip_end_ms!}`
      await db`UPDATE transcripts SET language = ${sarvamResult.language_code}, created_at = now() WHERE id = ${existing.id}`
      transcript = { id: existing.id as string }
    } else {
      if (isRetranscribe) {
        await db`DELETE FROM transcripts WHERE video_id = ${payload.video_id}`
      }
      // whole_video: the whole video was read, so nothing needs to read it again (see ai_edit)
      const [inserted] = await db`
        INSERT INTO transcripts (video_id, language, whole_video) VALUES (${payload.video_id}, ${sarvamResult.language_code}, ${!isClipJob}) RETURNING id
      `
      if (!inserted) throw new Error('Failed to insert transcript row')
      transcript = { id: inserted.id as string }
    }

    const entries = sarvamResult.words
    if (entries.length > 0) {
      const offsetMs = isClipJob ? payload.clip_start_ms! : 0
      const romanized = await transliterateToRoman(entries, sarvamResult.language_code)
      // English spoken inside a regional-language video: store it in that language's script too,
      // so "Auto language" captions are all in one script (word_roman keeps the English spelling)
      const native = await toNativeScript(entries, sarvamResult.language_code)

      const words = entries.map((e, i) => ({
        transcript_id: transcript.id,
        word: native[i] ?? e.word,
        word_roman: romanized[i] ?? null,
        start_ms: Math.round(e.start * 1000) + offsetMs,
        end_ms:   Math.round(e.end   * 1000) + offsetMs,
        speaker_id:  e.speaker ?? null,
        confidence:  e.confidence ?? null,
      }))
      for (let i = 0; i < words.length; i += 500) {
        await db`INSERT INTO transcript_words ${db(words.slice(i, i + 500))}`
      }
    }

    // The transcript is saved: the kept chunks have done their job
    if (readingKey) await r2.send(new DeleteObjectCommand({ Bucket: R2_BUCKET, Key: readingKey })).catch(() => {})

    if (!isRetranscribe && !isClipJob) {
      let durationMs: number | null = null
      try {
        const { stdout } = await execFileAsync('ffprobe', [
          '-v', 'error', '-show_entries', 'format=duration',
          '-of', 'default=noprint_wrappers=1:nokey=1', videoPath,
        ])
        const secs = parseFloat(stdout.trim())
        if (!isNaN(secs)) durationMs = Math.round(secs * 1000)
      } catch { /* optional */ }
      await db`UPDATE videos SET status = 'ready', storage_path = ${storagePath}, download_progress = 100, duration_ms = ${durationMs} WHERE id = ${payload.video_id}`
    }

    console.log(`[transcribe] ${isClipJob ? `clip ${payload.clip_id}` : `video ${payload.video_id}`} done — ${entries.length} words`)
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

interface SarvamWord { word: string; start: number; end: number; speaker?: string; confidence?: number }
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

async function vertexTranscribeChunk(buf: Buffer, languageCode?: string): Promise<SarvamWord[]> {
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
    const token = await auth.getAccessToken()
    const res = await fetchWithTimeout(url, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}`, 'x-goog-user-project': VERTEX_PROJECT!, 'Content-Type': 'application/json' },
      body,
    }, 300_000)
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
    if (!retryable || attempt >= GEMINI_MAX_TRIES) throw new Error(`Vertex transcribe ${res.status}: ${text.slice(0, 300)}`)
    const delaySec = Math.min(30, 5 * attempt)
    console.warn(`[vertex] ${res.status} on attempt ${attempt}, retrying in ${delaySec}s`)
    await new Promise(r => setTimeout(r, delaySec * 1000))
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

async function geminiTranscribeChunk(buf: Buffer, languageCode?: string): Promise<SarvamWord[]> {
  if (VERTEX_PROJECT) return vertexTranscribeChunk(buf, languageCode)
  for (let attempt = 1; ; attempt++) {
    const file = await geminiUploadAudio(buf)
    try {
      // diarization: each word gets a speaker label, so caption lines never mix two speakers
      const transcription_config: Record<string, unknown> = { mode: { type: 'verbatim', timestamp_granularities: ['word'], diarization_mode: 'speaker' } }
      if (languageCode) transcription_config.language_codes = [toGeminiLang(languageCode)]

      const res = await fetchWithTimeout(`${GEMINI_BASE}/v1beta/interactions`, {
        method: 'POST',
        headers: { 'x-goog-api-key': GEMINI_API_KEY!, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: GEMINI_STT_MODEL,
          input: [{ type: 'audio', uri: file.uri, mime_type: 'audio/wav' }],
          generation_config: { transcription_config },
        }),
      }, 300_000)

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
        const hours = Math.max(1, Math.round(hinted / 3600))
        throw new Error(`Transcription limit for today is used up (Gemini API). It resets in about ${hours} hour${hours === 1 ? '' : 's'}.`)
      }
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

export async function transcribeAudio(audioPath: string, videoId?: string, languageCode?: string, signal?: AbortSignal, kept?: ReadingStore): Promise<SarvamResponse> {
  if (!GEMINI_API_KEY && !VERTEX_PROJECT) throw new Error('Set GOOGLE_CLOUD_PROJECT (Vertex AI) or GEMINI_API_KEY for transcription')
  console.log(`[transcribe] ${VERTEX_PROJECT ? `Vertex AI ${VERTEX_STT_MODEL}` : GEMINI_STT_MODEL} (verbatim, word timestamps)`)

  const { stdout } = await execFileAsync('ffprobe', [
    '-v', 'error', '-show_entries', 'format=duration',
    '-of', 'default=noprint_wrappers=1:nokey=1', audioPath,
  ])
  const totalSec = parseFloat(stdout.trim())
  const numChunks = Math.max(1, Math.ceil(totalSec / GEMINI_CHUNK_SEC))
  console.log(`[transcribe] audio ${totalSec.toFixed(1)}s → ${numChunks} chunk(s)`)
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

    let words = await geminiTranscribeChunk(buf, lang)
    console.log(`[gemini] ${tag.startsWith('probe') ? `language ${tag}` : `chunk ${parseInt(tag, 10) + 1}/${numChunks}`} (${Math.round(durSec)}s): ${words.length} words${lang ? ` (${lang})` : ''}`)
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
    } else {
      const sample = async (from: number, tag: string) => {
        const words = await rangeWords(from, PROBE_SEC, undefined, tag)
        samples.push({ from, to: from + PROBE_SEC, words })
        return detectSarvamLang(words.map(w => w.word).join(' '))
      }
      lang = await sample(0, 'probe-start')
      // The start can be music or an English intro: check the middle before settling on auto
      if (!lang) lang = await sample(Math.max(PROBE_SEC, Math.min(totalSec / 2, totalSec - PROBE_SEC)), 'probe-middle')
    }
    if (lang) console.log(`[transcribe] detected language: ${lang} — locking for all chunks`)
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
  // Kept once the language is known and after every chunk. A save that fails only means that
  // chunk is read again next time.
  const reading: Reading = earlier ?? { v: 1, totalSec, chunkSec: GEMINI_CHUNK_SEC, lang: lang ?? null, chunks: {} }
  const keep = async () => {
    if (!kept || numChunks < 2) return
    await kept.save(reading).catch(e => console.warn('[transcribe] could not keep the reading so far:', e instanceof Error ? e.message : e))
  }
  if (!earlier) await keep()
  for (let i = 0; i < numChunks; i++) {
    const had = reading.chunks[i]
    if (had) { allWords.push(...had); continue }
    if (videoId) await setProgress(videoId, Math.round(63 + (i / numChunks) * 33))
    const got = await readChunk(i)
    allWords.push(...got)
    reading.chunks[i] = got
    await keep()
  }
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

// ── English re-pass for gaps ──────────────────────────────────────────────────
// With language_codes locked to e.g. te-IN (auto-detect drops a lot of Telugu speech), a
// sentence spoken entirely in English comes back with no words at all. Find the stretches of
// the chunk with no words, stitch them into one clip (short silence between), transcribe that
// once as English, and map the words back to their real times.
const GAP_MIN_SEC = 1.8
const GAP_PAD_SEC = 0.15
const GAP_SEPARATOR_SEC = 0.6

async function englishGapPass(chunkPathHint: string, audioPath: string, chunkStartSec: number, chunkSec: number, words: SarvamWord[]): Promise<SarvamWord[]> {
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
    console.log(`[gemini] English gap pass: ${after.toFixed(0)}s with sound of ${before.toFixed(0)}s of gaps`)
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
    const found = await geminiTranscribeChunk(buf, 'en-IN')
    const out: SarvamWord[] = []
    for (const w of found) {
      const p = placed.find(x => w.start >= x.at - 0.05 && w.start < x.at + (x.to - x.from))
      if (!p) continue // landed in a separator
      const start = p.from + (w.start - p.at)
      const end = Math.min(p.to, p.from + (w.end - p.at))
      // Only keep words inside a real gap, so nothing overlaps what the main pass found
      if (!gaps.some(([a, b]) => start >= a - GAP_PAD_SEC && start < b)) continue
      out.push({ ...w, start, end: Math.max(end, start + 0.08), speaker: w.speaker ? `en:${w.speaker}` : undefined })
    }
    return out
  } catch (e) {
    console.warn('[gemini] English gap pass failed, keeping the main pass only:', e)
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

// Gemini batch transliteration → natural Tenglish / Hinglish (how people type it on
// WhatsApp/YouTube), English loanwords in normal English spelling. One output per
// input word, same order; any batch that doesn't come back 1:1 is left undefined so
// the caller falls back to the rule-based Telugu transliterator for those words.
async function transliterateWithGemini(
  entries: SarvamWord[],
  languageCode: string,
): Promise<(string | undefined)[]> {
  const langName = LANG_NAMES[languageCode] ?? 'Indian'
  const style = langName === 'Telugu' ? 'Tenglish' : langName === 'Hindi' ? 'Hinglish' : `romanized ${langName}`
  const BATCH = 80
  const result: (string | undefined)[] = new Array(entries.length).fill(undefined)

  for (let i = 0; i < entries.length; i += BATCH) {
    const batch = entries.slice(i, i + BATCH)
    const words = batch.map(e => e.word.trim())
    // A batch occasionally comes back merged/split despite temperature 0 and explicit
    // instructions — one retry before falling back recovers most of these for free.
    for (let attempt = 1; attempt <= 2; attempt++) {
      try {
        // gemini-2.5-flash writes the most natural Tenglish in tests; Google has started retiring
        // 2.5 models for new accounts, so fall back to the always-current Flash alias on 404
        let res: Response | undefined
        for (const model of ['gemini-2.5-flash', 'gemini-flash-latest']) {
          res = await fetchWithTimeout(
          `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
            method: 'POST',
            headers: { 'x-goog-api-key': GEMINI_API_KEY!, 'Content-Type': 'application/json' },
            body: JSON.stringify({
              systemInstruction: { parts: [{ text:
`You convert ${langName} caption words into ${style}: the way ${langName} speakers write their language in English letters on WhatsApp or YouTube comments.
Rules:
- Exactly one output string per input word, same order. Never merge, split, drop or add words.
- Use common, natural spellings (e.g. నుంచి → "nunchi", నేను → "nenu", ఏమైంది → "emaindi", మనదే → "manade"), not academic transliteration and no diacritics.
- English words written in ${langName} script get their normal English spelling (ఇట్స్ → "it's", ట్రూ → "true", కాటన్ → "cotton", యాక్టర్స్ → "actors", యాడ్ → "ad").
- Numbers and words already in English letters stay as they are. Drop trailing punctuation.
- Lowercase, except names and the pronoun "I".` }] },
              contents: [{ role: 'user', parts: [{ text: JSON.stringify(words) }] }],
              generationConfig: {
                temperature: 0,
                responseMimeType: 'application/json',
                responseSchema: { type: 'ARRAY', items: { type: 'STRING' } },
                thinkingConfig: { thinkingBudget: 0 },
              },
            }),
          }, 60_000)
          if (res.status !== 404) break
        }
        if (!res!.ok) throw new Error(`Gemini ${res!.status}: ${(await res!.text()).slice(0, 200)}`)
        const data = await res!.json() as { candidates?: { content?: { parts?: { text?: string }[] } }[] }
        const arr = JSON.parse(data.candidates?.[0]?.content?.parts?.[0]?.text ?? '[]') as unknown[]
        if (!Array.isArray(arr) || arr.length !== batch.length) {
          throw new Error(`expected ${batch.length} words, got ${Array.isArray(arr) ? arr.length : 'non-array'}`)
        }
        for (let j = 0; j < batch.length; j++) {
          const v = arr[j]
          result[i + j] = typeof v === 'string' && v.trim() ? stripDiacritics(v.trim()).replace(/[.,!?।]+$/g, '') : undefined
        }
        break
      } catch (err) {
        if (attempt === 2) console.warn(`[transliterate] Gemini batch ${i}-${i + batch.length} failed, using fallback:`, err)
      }
    }
  }

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
  if (!langName || !GEMINI_API_KEY) return out
  const todo = entries
    .map((e, i) => ({ i, word: e.word.trim() }))
    .filter(x => LATIN.test(x.word) && !NATIVE_SCRIPT.test(x.word))
  if (!todo.length) return out

  const BATCH = 80
  for (let b = 0; b < todo.length; b += BATCH) {
    const batch = todo.slice(b, b + BATCH)
    // Punctuation stays outside the conversion and goes back on afterwards
    const split = batch.map(x => {
      const m = x.word.match(/^(.*?)([.,!?।…:;"')\]]*)$/)!
      return { core: m[1], tail: m[2] }
    })
    try {
      let res: Response | undefined
      for (const model of ['gemini-2.5-flash', 'gemini-flash-latest']) {
        res = await fetchWithTimeout(
          `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
            method: 'POST',
            headers: { 'x-goog-api-key': GEMINI_API_KEY!, 'Content-Type': 'application/json' },
            body: JSON.stringify({
              systemInstruction: { parts: [{ text:
`You write English words in ${langName} script, the way ${langName} captions and news write English words spoken in a ${langName} video (phonetically, as ${langName} speakers pronounce them).
Rules:
- Exactly one output string per input word, same order. Never merge, split, drop or add words.
- Spell by sound in ${langName} script${langName === 'Telugu' ? ' (so → "సో", think → "థింక్", actually → "యాక్చువల్లీ", I → "ఐ", it\'s → "ఇట్స్", brand → "బ్రాండ్", YouTube → "యూట్యూబ్")' : ''}.
- Numbers stay as digits. Keep capital-letter brand names readable by sound. No English letters in the output unless the input is only digits.` }] },
              contents: [{ role: 'user', parts: [{ text: JSON.stringify(split.map(x => x.core)) }] }],
              generationConfig: {
                temperature: 0,
                responseMimeType: 'application/json',
                responseSchema: { type: 'ARRAY', items: { type: 'STRING' } },
                thinkingConfig: { thinkingBudget: 0 },
              },
            }),
          }, 60_000)
        if (res.status !== 404) break
      }
      if (!res!.ok) throw new Error(`Gemini ${res!.status}: ${(await res!.text()).slice(0, 200)}`)
      const data = await res!.json() as { candidates?: { content?: { parts?: { text?: string }[] } }[] }
      const arr = JSON.parse(data.candidates?.[0]?.content?.parts?.[0]?.text ?? '[]') as unknown[]
      if (!Array.isArray(arr) || arr.length !== batch.length) {
        throw new Error(`expected ${batch.length} words, got ${Array.isArray(arr) ? arr.length : 'non-array'}`)
      }
      batch.forEach((x, j) => {
        const v = arr[j]
        // Only take real conversions; anything still in English letters keeps the original
        if (typeof v === 'string' && v.trim() && NATIVE_SCRIPT.test(v) && !LATIN.test(v)) out[x.i] = v.trim() + split[j].tail
      })
    } catch (err) {
      console.warn(`[native-script] batch ${b}-${b + batch.length} failed, keeping English letters:`, err)
    }
  }
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
  if (GEMINI_API_KEY) {
    const viaGemini = await transliterateWithGemini(entries, lang)
    if (viaGemini.every(Boolean)) return viaGemini
    const fb = await fallback()
    return viaGemini.map((r, i) => r ?? fb[i])
  }
  return fallback()
}
