import { execFile, spawn } from 'node:child_process'
import { promisify } from 'node:util'
import { writeFile, readFile, unlink, mkdtemp, readdir, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { GetObjectCommand, PutObjectCommand } from '@aws-sdk/client-s3'
import { r2, R2_BUCKET } from '../r2.js'
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

async function setProgress(videoId: string, pct: number) {
  await db`UPDATE videos SET download_progress = ${pct} WHERE id = ${videoId}`
}

export async function handleTranscribeJob(job: Job) {
  const raw = job.payload
  const payload = (typeof raw === 'string' ? JSON.parse(raw) : raw) as TranscribeJobPayload
  const isLinkJob = !payload.storage_path
  const isRetranscribe = !!payload.is_retranscribe
  const isClipJob = !!payload.clip_id && payload.clip_start_ms !== undefined && payload.clip_end_ms !== undefined

  // Plain upload jobs (already in storage, no clip): just mark ready.
  // Transcription is deferred to clip creation now.
  if (!isLinkJob && !isClipJob && !isRetranscribe) {
    await db`UPDATE videos SET status = 'ready', download_progress = 100 WHERE id = ${payload.video_id}`
    console.log(`[transcribe] upload video ${payload.video_id} ready — transcription deferred to clip creation`)
    return
  }

  if (!isRetranscribe && !isClipJob) {
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
        // Link job: yt-dlp → transcode → upload to storage
        const result = await downloadWithYtDlp(payload.video_id, tmp)
        videoPath = result.localPath
        storagePath = result.storagePath
        await db`UPDATE videos SET storage_path = ${storagePath} WHERE id = ${payload.video_id}`
      } else {
        // Clip or retranscribe job: download video from storage
        await setProgress(payload.video_id, 30)
        videoPath = join(tmp, 'video.mp4')
        await writeFile(videoPath, await r2Download(storagePath))
        await setProgress(payload.video_id, 55)
      }

      // Extract audio — for clip jobs, seek to clip range only (fast & cheap)
      await setProgress(payload.video_id, 60)
      if (isClipJob) {
        const startSec = payload.clip_start_ms! / 1000
        const durSec   = (payload.clip_end_ms! - payload.clip_start_ms!) / 1000
        await execFileAsync('ffmpeg', [
          '-ss', String(startSec), '-i', videoPath,
          '-t', String(durSec),
          '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', audioPath,
        ])
      } else {
        await execFileAsync('ffmpeg', [
          '-i', videoPath, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', audioPath,
        ])
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

    // ── Link-only jobs (no clip_id): just set video ready, skip transcription ──
    if (isLinkJob && !isClipJob && !isRetranscribe) {
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
      console.log(`[transcribe] link video ${payload.video_id} ready — transcription deferred to clip creation`)
      return
    }

    // ── Transcription ─────────────────────────────────────────────────────────
    const sarvamResult = await transcribeAudio(audioPath, (!isRetranscribe && !isClipJob) ? payload.video_id : undefined, payload.language_code)

    // Clip retranscribe: replace only this clip's time range in the latest transcript, so
    // other clips of the same video keep their captions. created_at is bumped so the
    // editor's "since" poll picks up the new words.
    const [existing] = isRetranscribe && isClipJob
      ? await db`SELECT id FROM transcripts WHERE video_id = ${payload.video_id} ORDER BY created_at DESC LIMIT 1`
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
      const [inserted] = await db`
        INSERT INTO transcripts (video_id, language) VALUES (${payload.video_id}, ${sarvamResult.language_code}) RETURNING id
      `
      if (!inserted) throw new Error('Failed to insert transcript row')
      transcript = { id: inserted.id as string }
    }

    const entries = sarvamResult.words
    if (entries.length > 0) {
      const offsetMs = isClipJob ? payload.clip_start_ms! : 0
      const romanized = await transliterateToRoman(entries, sarvamResult.language_code, process.env.SARVAM_API_KEY)

      const words = entries.map((e, i) => ({
        transcript_id: transcript.id,
        word: e.word,
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
    // on an already-ready video, so corrupting its status there would be wrong.
    if (!isClipJob && !isRetranscribe) {
      await db`UPDATE videos SET status = 'failed' WHERE id = ${payload.video_id}`.catch(() => {})
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

async function downloadWithYtDlp(videoId: string, tmp: string): Promise<{ localPath: string; storagePath: string }> {
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
    const proc = spawn('yt-dlp', ytdlpArgs)

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
    ])

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
  const fileBuffer = await readFile(localPath)
  await r2Upload(storagePath, fileBuffer, 'video/mp4')

  await setProgress(videoId, 58)
  return { localPath, storagePath }
}

interface SarvamWord { word: string; start: number; end: number; speaker?: string; confidence?: number }
interface SarvamRawResponse {
  language_code: string
  transcript?: string
  timestamps?: {
    words: string[]
    start_time_seconds: number[]
    end_time_seconds: number[]
  }
}
interface SarvamResponse { language_code: string; transcript?: string; words: SarvamWord[] }

// const CHUNK_SEC = 25  // Sarvam max is 30s; keep at 25s for safety   // old pipeline (disabled)

// const PARALLEL = 3    // concurrent ffmpeg extractions AND Sarvam API calls (higher → 429 rate limit)   // old pipeline (disabled)

const OPENAI_API_KEY = process.env.OPENAI_API_KEY
// const GROQ_API_KEY   = process.env.GROQ_API_KEY   // old pipeline (disabled)
// const SARVAM_API_KEY = process.env.SARVAM_API_KEY   // old pipeline (disabled)

/* ── OLD PIPELINE (disabled): Sarvam saaras:v3 + Groq whisper-large-v3 ──────────
   Replaced by Gemini 3.5 Transcribe (see transcribeAudio below). Kept for rollback:
   uncomment this block and delete the Gemini transcribeAudio to switch back.
// Normalize a word for fuzzy matching: lowercase, strip punctuation.
// Works across scripts (Telugu Unicode, Hindi Unicode, Latin) since we compare code-points.
function normalizeWord(w: string): string {
  return w.toLowerCase().replace(/[^\p{L}\p{N}]/gu, '')
}

// Single-row Levenshtein — O(n) space, fast enough for short words.
function editDistance(a: string, b: string): number {
  if (a === b) return 0
  const la = a.length, lb = b.length
  if (la === 0) return lb
  if (lb === 0) return la
  // Early-exit: if lengths differ by more than 60% they can't be similar
  if (Math.abs(la - lb) > Math.max(la, lb) * 0.6) return Math.max(la, lb)
  const row = Array.from({ length: lb + 1 }, (_, j) => j)
  for (let i = 1; i <= la; i++) {
    let prev = row[0]++
    for (let j = 1; j <= lb; j++) {
      const tmp = row[j]
      row[j] = a[i - 1] === b[j - 1] ? prev : 1 + Math.min(prev, row[j], row[j - 1])
      prev = tmp
    }
  }
  return row[lb]
}

// Two words are "similar" if their normalised edit distance is below threshold.
// Short words (≤2 chars) require an exact match to avoid false positives.
function isSimilar(a: string, b: string, threshold = 0.45): boolean {
  const na = normalizeWord(a), nb = normalizeWord(b)
  if (na === nb) return true
  const maxLen = Math.max(na.length, nb.length)
  if (maxLen === 0) return true
  if (maxLen <= 2) return false           // exact-only for tiny words
  return editDistance(na, nb) / maxLen < threshold
}

// Fuzzy sequence alignment: map M Sarvam words onto N Groq timing anchors.
//
// For each Sarvam word we search a window around the expected proportional
// Groq position for a similar word (by edit-distance).  Matched words receive
// Groq's exact timestamp; unmatched words are time-interpolated between their
// nearest matched neighbours — a much shorter span than the old whole-phrase
// linear interpolation, so drift is far smaller.
//
// This correctly handles the main mismatch cases:
//   • Groq inserts filler words ("uh", "hmm", repeated articles) → skipped
//   • Groq omits a word → Sarvam word interpolated between adjacent anchors
//   • Telugu/Hindi: both APIs emit the same Unicode script so edit-distance works
//   • Hinglish English words: often identical strings → direct match
function alignToGroqTimings(wordList: string[], groqWords: SarvamWord[]): SarvamWord[] {
  const M = wordList.length
  const N = groqWords.length
  if (M === 0) return []
  if (N === 0) return wordList.map(word => ({ word, start: 0, end: 0 }))

  // Window size: at least 3, but scales with Groq list size to catch heavier drift.
  const WINDOW = Math.max(3, Math.round(N / 4))

  // anchors[i] = index into groqWords that sarvam word i matched, or -1
  const anchors = new Array<number>(M).fill(-1)
  let minJ = 0  // enforce monotonic (non-decreasing) matching

  for (let i = 0; i < M; i++) {
    const expected = Math.round(i * (N - 1) / Math.max(M - 1, 1))
    const jStart   = Math.max(minJ, expected - WINDOW)
    const jEnd     = Math.min(N - 1, expected + WINDOW)

    let bestJ = -1, bestDist = Infinity
    for (let j = jStart; j <= jEnd; j++) {
      if (!isSimilar(wordList[i], groqWords[j].word)) continue
      const d = Math.abs(j - expected)
      if (d < bestDist) { bestDist = d; bestJ = j }
    }

    if (bestJ >= 0) { anchors[i] = bestJ; minJ = bestJ + 1 }
  }

  const matched = anchors.filter(a => a >= 0).length
  if (matched > 0) {
    console.log(`[align] ${matched}/${M} words anchored to Groq timestamps (${M - matched} interpolated)`)
  }

  return wordList.map((word, i) => {
    const gIdx = anchors[i]

    // Direct match — use Groq's exact start/end
    if (gIdx >= 0) return { word, start: groqWords[gIdx].start, end: groqWords[gIdx].end }

    // Unmatched — interpolate between nearest anchor on each side
    let lo = i - 1; while (lo >= 0 && anchors[lo] < 0) lo--
    let hi = i + 1; while (hi < M && anchors[hi] < 0) hi++

    const tStart = lo >= 0 ? groqWords[anchors[lo]].end   : groqWords[0].start
    const tEnd   = hi < M  ? groqWords[anchors[hi]].start : groqWords[N - 1].end
    const tRange = Math.max(0, tEnd - tStart)

    // Divide the gap evenly among all unmatched words in this run
    const gapStart = lo + 1
    const gapSize  = (hi < M ? hi : M) - gapStart
    const posInGap = i - gapStart

    const slotStart = tStart + (posInGap       / Math.max(gapSize, 1)) * tRange
    const slotEnd   = tStart + ((posInGap + 1) / Math.max(gapSize, 1)) * tRange

    return { word, start: slotStart, end: slotEnd }
  })
}

// Sarvam and Groq run in parallel per chunk.
// Sarvam: correct Telugu spelling (phrase-level timestamps).
// Groq:   accurate per-word timestamps (may misspell).
// We map each Sarvam word onto Groq's timeline by interpolating position
// (word i of M → Groq position i*(N-1)/(M-1)), giving correct words + real timing.
async function transcribeAudio(audioPath: string, videoId?: string, languageCode?: string): Promise<SarvamResponse> {
  const sarvamKey = SARVAM_API_KEY
  if (!sarvamKey) throw new Error('SARVAM_API_KEY is not set')
  const groqKey = GROQ_API_KEY  // used for per-word timing

  console.log(`[transcribe] sarvam (spelling) + groq (timing) → aligned pipeline`)

  const { stdout } = await execFileAsync('ffprobe', [
    '-v', 'error', '-show_entries', 'format=duration',
    '-of', 'default=noprint_wrappers=1:nokey=1', audioPath,
  ])
  const totalSec = parseFloat(stdout.trim())
  const numChunks = Math.ceil(totalSec / CHUNK_SEC)
  console.log(`[transcribe] audio ${totalSec.toFixed(1)}s → ${numChunks} chunk(s) (${PARALLEL} parallel)`)
  if (videoId) await setProgress(videoId, 62)

  interface ChunkResult { startSec: number; result: SarvamResponse }
  const ordered: ChunkResult[] = []

  async function extractAndTranscribeChunk(i: number, lang: string | undefined): Promise<ChunkResult> {
    const startSec = i * CHUNK_SEC
    const chunkPath = audioPath.replace('.wav', `_chunk${i}.wav`)
    await execFileAsync('ffmpeg', [
      '-i', audioPath, '-ss', String(startSec), '-t', String(CHUNK_SEC),
      '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', chunkPath,
    ])
    const buf = await readFile(chunkPath)
    await unlink(chunkPath).catch(() => {})

    const sarvamLang = lang ?? 'unknown'
    const groqLang = lang ? toWhisperLang(lang) : undefined

    // Run both in parallel — Sarvam for correct words, Groq for word-level timing
    const [sarvam, groqResult] = await Promise.all([
      callSarvamChunk(buf, sarvamKey!, sarvamLang),
      groqKey
        ? withRetry(() => callWhisperChunk(buf, groqKey, groqLang ?? '', 'https://api.groq.com/openai/v1', 'whisper-large-v3'), 3, 1000)
            .catch(err => { console.error(`[groq] all 3 retries failed, falling back to Sarvam timing: ${err instanceof Error ? err.message : err}`); return null })
        : Promise.resolve(null),
    ])

    const groqWords = groqResult?.words ?? []

    const expandedWords: SarvamWord[] = []
    for (const phrase of sarvam.words) {
      const wordList = phrase.word.trim().split(/\s+/).filter(w => w.length > 0)
      if (wordList.length === 0) continue
      const phraseDur = phrase.end - phrase.start

      const phraseGroq = groqWords.filter(gw => gw.start < phrase.end && gw.end > phrase.start)
      if (phraseGroq.length > 0) {
        expandedWords.push(...alignToGroqTimings(wordList, phraseGroq))
      } else {
        wordList.forEach((word, j) => {
          expandedWords.push({
            word,
            start: phrase.start + (j / wordList.length) * phraseDur,
            end: phrase.start + ((j + 1) / wordList.length) * phraseDur,
          })
        })
      }
    }

    return { startSec, result: { language_code: sarvam.language_code, words: expandedWords } }
  }

  // Run first chunk alone to detect language, then lock for remaining chunks.
  let detectedLang = (languageCode && languageCode !== 'unknown') ? languageCode : undefined
  if (numChunks > 1 && !detectedLang) {
    if (videoId) await setProgress(videoId, 63)
    const first = await extractAndTranscribeChunk(0, undefined)
    ordered.push(first)
    if (first.result.language_code && first.result.language_code !== 'unknown') {
      detectedLang = first.result.language_code
      console.log(`[transcribe] detected language: ${detectedLang} — locking for remaining chunks`)
    }
  }

  const startBatch = ordered.length  // 0 if language was known, 1 if we ran first chunk
  for (let b = startBatch; b < numChunks; b += PARALLEL) {
    if (videoId) {
      const pct = Math.round(63 + (b / numChunks) * 33)
      await setProgress(videoId, pct)
    }
    const batchIndices = Array.from({ length: Math.min(PARALLEL, numChunks - b) }, (_, k) => b + k)
    const batchResults = await Promise.all(
      batchIndices.map(i => extractAndTranscribeChunk(i, detectedLang))
    )
    ordered.push(...batchResults)
  }

  ordered.sort((a, b) => a.startSec - b.startSec)
  const allWords: SarvamWord[] = []
  let language = 'unknown'
  for (const { startSec, result } of ordered) {
    if (result.language_code && result.language_code !== 'unknown') language = result.language_code
    for (const w of result.words) {
      allWords.push({ ...w, start: w.start + startSec, end: w.end + startSec })
    }
  }

  // Set each word's end to next word's start (removes gaps and overlaps),
  // but cap at MAX_WORD_HOLD_SEC so captions disappear during long music/silence gaps
  // rather than holding one word for 8-10 seconds. For dense speech all gaps are <4s
  // so the cap never fires; for music-heavy audio it prevents stale captions.
  const MAX_WORD_HOLD_SEC = 4.0
  for (let i = 0; i < allWords.length - 1; i++) {
    allWords[i] = {
      ...allWords[i],
      end: Math.min(allWords[i + 1].start, allWords[i].start + MAX_WORD_HOLD_SEC),
    }
  }
  if (allWords.length > 0 && allWords[allWords.length - 1].end < totalSec) {
    allWords[allWords.length - 1] = { ...allWords[allWords.length - 1], end: totalSec }
  }

  return { language_code: language, words: allWords }
}

// ── Whisper: true per-word timestamps for every language ─────────────────────
// Maps various code forms → ISO-639-1 that Whisper's API accepts.
const LANG_TO_WHISPER: Record<string, string> = {
  // Sarvam xx-IN codes
  'en-IN': 'en', 'hi-IN': 'hi', 'te-IN': 'te', 'ta-IN': 'ta',
  'kn-IN': 'kn', 'ml-IN': 'ml', 'bn-IN': 'bn', 'gu-IN': 'gu',
  'mr-IN': 'mr', 'pa-IN': 'pa', 'od-IN': 'or',
  // Whisper full language names (returned in response.language)
  'telugu': 'te', 'hindi': 'hi', 'tamil': 'ta', 'kannada': 'kn',
  'malayalam': 'ml', 'bengali': 'bn', 'gujarati': 'gu', 'marathi': 'mr',
  'punjabi': 'pa', 'odia': 'or', 'english': 'en',
}

function toWhisperLang(code: string): string {
  return LANG_TO_WHISPER[code.toLowerCase()] ?? code.split('-')[0].toLowerCase()
}

async function withRetry<T>(fn: () => Promise<T>, retries = 3, baseDelayMs = 1000): Promise<T> {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      return await fn()
    } catch (err) {
      if (attempt === retries) throw err
      const delay = baseDelayMs * Math.pow(2, attempt - 1)
      console.warn(`[groq] attempt ${attempt} failed, retrying in ${delay}ms: ${err instanceof Error ? err.message : err}`)
      await new Promise(resolve => setTimeout(resolve, delay))
    }
  }
  throw new Error('unreachable')
}

async function callWhisperChunk(
  buf: Buffer, apiKey: string, languageCode?: string,
  baseUrl = 'https://api.openai.com/v1', model = 'whisper-1',
  prompt?: string,  // Sarvam transcript text — seeds Groq's decoder for better spelling
): Promise<SarvamResponse> {
  const formData = new FormData()
  formData.append('file', new Blob([buf], { type: 'audio/wav' }), 'audio.wav')
  formData.append('model', model)
  formData.append('response_format', 'verbose_json')
  formData.append('timestamp_granularities[]', 'word')

  if (languageCode && languageCode !== 'unknown') {
    formData.append('language', toWhisperLang(languageCode))
  }
  if (prompt) {
    formData.append('prompt', prompt.slice(0, 500))
  }

  const res = await fetch(`${baseUrl}/audio/transcriptions`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}` },
    body: formData,
  })

  if (!res.ok) throw new Error(`Whisper API ${res.status}: ${await res.text()}`)

  const raw = await res.json() as {
    language?: string
    words?: { word: string; start: number; end: number }[]
  }

  const words: SarvamWord[] = (raw.words ?? []).map(w => ({
    word: w.word.trim(),
    start: w.start,
    end: w.end,
  })).filter(w => w.word.length > 0)

  if (words.length > 0) {
    const sample = words.slice(0, 3).map(w => `"${w.word}"(${w.start.toFixed(2)}-${w.end.toFixed(2)}s)`).join(', ')
    console.log(`[whisper] ${words.length} words in ${(words[words.length-1]?.end ?? 0).toFixed(1)}s. Sample: ${sample}`)
  }

  return { language_code: raw.language ?? languageCode ?? 'unknown', words }
}
── end OLD PIPELINE ── */
// ── fetch with timeout (used by Sarvam transliteration) ──────────────────────
async function fetchWithTimeout(url: string, init: RequestInit, timeoutMs: number): Promise<Response> {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), timeoutMs)
  try {
    return await fetch(url, { ...init, signal: ctrl.signal })
  } finally {
    clearTimeout(timer)
  }
}

/* ── OLD PIPELINE (disabled): Sarvam saaras:v3 + Groq whisper-large-v3 ──────────
   Replaced by Gemini 3.5 Transcribe (see transcribeAudio below). Kept for rollback:
   uncomment this block and delete the Gemini transcribeAudio to switch back.
async function callSarvamChunk(buf: Buffer, apiKey: string, languageCode?: string): Promise<SarvamResponse> {
  const TIMEOUT_MS = 90_000

  return withRetry(async () => {
    const formData = new FormData()
    formData.append('file', new Blob([buf], { type: 'audio/wav' }), 'audio.wav')
    formData.append('model', 'saaras:v3')
    formData.append('language_code', languageCode ?? 'unknown')
    formData.append('with_timestamps', 'true')

    const res = await fetchWithTimeout('https://api.sarvam.ai/speech-to-text', {
      method: 'POST',
      headers: { 'api-subscription-key': apiKey },
      body: formData,
    }, TIMEOUT_MS)

    if (res.status === 429) {
      await new Promise(r => setTimeout(r, 5000))
      throw new Error(`Sarvam API 429: rate limit`)
    }
    if (!res.ok) throw new Error(`Sarvam API ${res.status}: ${await res.text()}`)

    const raw = await res.json() as SarvamRawResponse
    const ts = raw.timestamps
    const words: SarvamWord[] = ts?.words?.map((w, i) => ({
      word: w,
      start: ts.start_time_seconds[i] ?? 0,
      end: ts.end_time_seconds[i] ?? 0,
    })) ?? []

    if (words.length > 0) {
      const sample = words.slice(0, 3).map(w => `"${w.word}"(${w.start.toFixed(2)}-${w.end.toFixed(2)}s)`).join(', ')
      console.log(`[sarvam] ${words.length} tokens in ${(words[words.length-1]?.end ?? 0).toFixed(1)}s. Sample: ${sample}`)
    }

    return { language_code: raw.language_code, transcript: raw.transcript, words }
  }, 3, 1000)
}
── end OLD PIPELINE (callSarvamChunk) ── */

// ── Gemini 3.5 Transcribe: words + word-level timestamps in one call ─────────
// Audio is sent in 5-minute chunks: keeps each request under the Tier-1 limit of
// 10k input tokens/min (~25 tokens per second of audio) and limits timestamp drift.
// Each chunk is uploaded via the Files API, transcribed verbatim, then deleted.
const GEMINI_API_KEY    = process.env.GEMINI_API_KEY
const GEMINI_BASE       = 'https://generativelanguage.googleapis.com'
const GEMINI_STT_MODEL  = 'gemini-3.5-transcribe'
const GEMINI_CHUNK_SEC  = 300
const GEMINI_MAX_TRIES  = 8

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

async function geminiTranscribeChunk(buf: Buffer, languageCode?: string): Promise<SarvamWord[]> {
  for (let attempt = 1; ; attempt++) {
    const file = await geminiUploadAudio(buf)
    try {
      const transcription_config: Record<string, unknown> = { mode: { type: 'verbatim', timestamp_granularities: ['word'] } }
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
          steps?: { content?: { annotations?: { type: string; text: string; start_offset: string; end_offset: string }[] }[] }[]
        }
        const words: SarvamWord[] = []
        for (const step of raw.steps ?? []) {
          for (const c of step.content ?? []) {
            for (const a of c.annotations ?? []) {
              if (a.type !== 'word_info' || !a.text?.trim()) continue
              words.push({ word: a.text.trim(), start: parseFloat(a.start_offset), end: parseFloat(a.end_offset) })
            }
          }
        }
        return words
      }

      const body = await res.text()
      const retryable = res.status === 429 || res.status >= 500
      if (!retryable || attempt >= GEMINI_MAX_TRIES) throw new Error(`Gemini transcribe ${res.status}: ${body.slice(0, 300)}`)
      // Rate limited (10k tokens/min on Tier 1) — wait for the window to reset
      const hinted = parseFloat(body.match(/retry in ([\d.]+)s/i)?.[1] ?? '0')
      const delaySec = Math.max(hinted, res.status === 429 ? 20 : 5 * attempt)
      console.warn(`[gemini] ${res.status} on attempt ${attempt}, retrying in ${delaySec}s`)
      await new Promise(r => setTimeout(r, delaySec * 1000))
    } finally {
      fetchWithTimeout(`${GEMINI_BASE}/v1beta/${file.name}`, { method: 'DELETE', headers: { 'x-goog-api-key': GEMINI_API_KEY! } }, 30_000)
        .catch(() => {})
    }
  }
}

async function transcribeAudio(audioPath: string, videoId?: string, languageCode?: string): Promise<SarvamResponse> {
  if (!GEMINI_API_KEY) throw new Error('GEMINI_API_KEY is not set')
  console.log(`[transcribe] ${GEMINI_STT_MODEL} (verbatim, word timestamps)`)

  const { stdout } = await execFileAsync('ffprobe', [
    '-v', 'error', '-show_entries', 'format=duration',
    '-of', 'default=noprint_wrappers=1:nokey=1', audioPath,
  ])
  const totalSec = parseFloat(stdout.trim())
  const numChunks = Math.max(1, Math.ceil(totalSec / GEMINI_CHUNK_SEC))
  console.log(`[transcribe] audio ${totalSec.toFixed(1)}s → ${numChunks} chunk(s)`)
  if (videoId) await setProgress(videoId, 62)

  async function chunkWords(i: number, lang: string | undefined): Promise<SarvamWord[]> {
    const startSec = i * GEMINI_CHUNK_SEC
    const chunkPath = audioPath.replace('.wav', `_gchunk${i}.wav`)
    await execFileAsync('ffmpeg', [
      '-i', audioPath, '-ss', String(startSec), '-t', String(GEMINI_CHUNK_SEC),
      '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', chunkPath,
    ])
    const buf = await readFile(chunkPath)
    await unlink(chunkPath).catch(() => {})
    if (buf.length < 16_000) return []  // < 0.5s of audio — nothing to transcribe

    const words = await geminiTranscribeChunk(buf, lang)
    console.log(`[gemini] chunk ${i + 1}/${numChunks}: ${words.length} words${lang ? ` (${lang})` : ''}`)
    return words.map(w => ({ ...w, start: w.start + startSec, end: w.end + startSec }))
  }

  // Auto-detect drops a lot of Telugu speech, so detect the language from the
  // first chunk's script, then lock it for every chunk (re-running chunk 1).
  let lang = (languageCode && languageCode !== 'unknown') ? languageCode : undefined
  const allWords: SarvamWord[] = []
  let probe: SarvamWord[] | null = null
  if (!lang) {
    probe = await chunkWords(0, undefined)
    lang = detectSarvamLang(probe.map(w => w.word).join(' '))
    if (lang) console.log(`[transcribe] detected language: ${lang} — locking for all chunks`)
  }
  for (let i = 0; i < numChunks; i++) {
    if (i === 0 && probe && !lang) { allWords.push(...probe); continue }  // English / Latin script: keep auto result
    if (videoId) await setProgress(videoId, Math.round(63 + (i / numChunks) * 33))
    const words = await chunkWords(i, lang)
    // Chunk 1 was transcribed twice (auto + locked) — keep whichever caught more speech
    allWords.push(...(i === 0 && probe && probe.length > words.length ? probe : words))
  }
  const language = lang ?? (allWords.length ? 'en-IN' : 'unknown')

  // Gemini occasionally returns a word slightly out of order — keep the timeline monotonic
  allWords.sort((a, b) => a.start - b.start)

  // Set each word's end to next word's start (removes gaps and overlaps),
  // but cap at MAX_WORD_HOLD_SEC so captions disappear during long music/silence gaps
  // rather than holding one word for 8-10 seconds.
  const MAX_WORD_HOLD_SEC = 4.0
  for (let i = 0; i < allWords.length - 1; i++) {
    allWords[i] = {
      ...allWords[i],
      end: Math.min(allWords[i + 1].start, allWords[i].start + MAX_WORD_HOLD_SEC),
    }
  }
  if (allWords.length > 0 && allWords[allWords.length - 1].end < totalSec) {
    allWords[allWords.length - 1] = { ...allWords[allWords.length - 1], end: totalSec }
  }

  return { language_code: language, words: allWords }
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
// Maps short Whisper codes (te, hi, ta…) to Sarvam's xx-IN format
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

// Groq LLM batch transliteration — far more natural than per-character mapping.
// Sends all words in one request so the model has phonetic context.
async function transliterateWithLLM(
  entries: SarvamWord[],
  languageCode: string,
  apiKey: string,
): Promise<(string | undefined)[]> {
  const langName = LANG_NAMES[languageCode] ?? 'Indian'
  const BATCH = 80
  const result: (string | undefined)[] = new Array(entries.length).fill(undefined)

  for (let i = 0; i < entries.length; i += BATCH) {
    const batch = entries.slice(i, i + BATCH)
    const words = batch.map(e => e.word)
    try {
      const res = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
        body: JSON.stringify({
          model: 'gpt-4o-mini',
          temperature: 0,
          max_tokens: 1024,
          messages: [
            {
              role: 'system',
              content: `You are given ${langName} words auto-transcribed by speech recognition. Some words may have errors (repeated syllables, garbled characters from background music).

Steps:
1. Mentally correct any obvious error (e.g. repeated consonants like "న్న్న్న్" → simpler word).
2. Write the corrected word as ENGLISH LETTERS showing how it sounds (phonetic romanization).

IMPORTANT: Output must be in English/Roman letters only — never output the original script.
Return ONLY a JSON array of English strings, same length and order as input. No markdown, no explanations.

Example: ["నేను", "విలన్", "హీరో"] → ["nenu", "villan", "hero"]`,
            },
            { role: 'user', content: JSON.stringify(words) },
          ],
        }),
      })
      if (!res.ok) throw new Error(`OpenAI chat ${res.status}: ${await res.text()}`)
      const data = await res.json() as { choices: Array<{ message: { content: string } }> }
      const raw = data.choices[0]?.message?.content?.trim() ?? '[]'
      const clean = raw.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim()
      const parsed: unknown = JSON.parse(clean)
      const arr: unknown[] = Array.isArray(parsed)
        ? parsed
        : Array.isArray((parsed as Record<string, unknown>).words)
          ? (parsed as Record<string, unknown[]>).words
          : Object.values(parsed as object)
      for (let j = 0; j < batch.length; j++) {
        const v = arr[j]
        result[i + j] = typeof v === 'string' && v.trim() ? v.trim() : undefined
      }
    } catch (err) {
      console.warn(`[transliterate] OpenAI GPT batch ${i}-${i + batch.length} failed:`, err)
    }
  }

  console.log(`[transliterate] ${languageCode} → Roman via OpenAI GPT for ${entries.length} word(s)`)
  return result
}

// Gemini batch transliteration → natural Tenglish / Hinglish (how people type it on
// WhatsApp/YouTube), English loanwords in normal English spelling. One output per
// input word, same order; any batch that doesn't come back 1:1 is left undefined so
// the caller falls back to the rule-based / Sarvam transliterator for those words.
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
    try {
      const res = await fetchWithTimeout(
        `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent`, {
          method: 'POST',
          headers: { 'x-goog-api-key': GEMINI_API_KEY!, 'Content-Type': 'application/json' },
          body: JSON.stringify({
            systemInstruction: { parts: [{ text:
`You convert ${langName} caption words into ${style}: the way ${langName} speakers write their language in English letters on WhatsApp or YouTube comments.
Rules:
- Exactly one output string per input word, same order. Never merge, split, drop or add words.
- Use common, natural spellings (e.g. నుంచి → "nunchi", నేను → "nenu", ఏమైంది → "emaindi", మనదే → "manade"), not academic transliteration and no diacritics.
- English words written in ${langName} script get their normal English spelling (ఇట్స్ → "it's", ట్రూ → "true", కాటన్ → "cotton", యాక్టర్స్ → "actors").
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
      if (!res.ok) throw new Error(`Gemini ${res.status}: ${(await res.text()).slice(0, 200)}`)
      const data = await res.json() as { candidates?: { content?: { parts?: { text?: string }[] } }[] }
      const arr = JSON.parse(data.candidates?.[0]?.content?.parts?.[0]?.text ?? '[]') as unknown[]
      if (!Array.isArray(arr) || arr.length !== batch.length) {
        throw new Error(`expected ${batch.length} words, got ${Array.isArray(arr) ? arr.length : 'non-array'}`)
      }
      for (let j = 0; j < batch.length; j++) {
        const v = arr[j]
        result[i + j] = typeof v === 'string' && v.trim() ? stripDiacritics(v.trim()).replace(/[.,!?।]+$/g, '') : undefined
      }
    } catch (err) {
      console.warn(`[transliterate] Gemini batch ${i}-${i + batch.length} failed, using fallback:`, err)
    }
  }

  console.log(`[transliterate] ${languageCode} → ${style} via Gemini for ${entries.length} word(s)`)
  return result
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

// Sarvam transliterate API — sends words as a sentence with numeric markers
// so Sarvam gets full phonetic context, then splits back on the markers.
async function transliterateWithSarvam(
  entries: SarvamWord[],
  languageCode: string,
  apiKey: string,
): Promise<(string | undefined)[]> {
  const BATCH = 20   // words per sentence call
  const result: (string | undefined)[] = new Array(entries.length).fill(undefined)

  for (let i = 0; i < entries.length; i += BATCH) {
    const batch = entries.slice(i, i + BATCH)

    // Separate ASCII words from Indian-script words so ASCII passes through unchanged
    const isAscii = (w: string) => !/[^\x00-\x7F]/.test(w.replace(/[.,!?।]/g, ''))

    // Build a sentence with [N] markers between words so we can re-split after
    // transliteration.  Sarvam leaves digit tokens like [0] intact.
    const input = batch.map((e, j) => `[${j}] ${e.word.trim()}`).join(' ')

    try {
      const res = await fetchWithTimeout('https://api.sarvam.ai/transliterate', {
        method: 'POST',
        headers: { 'api-subscription-key': apiKey, 'Content-Type': 'application/json' },
        body: JSON.stringify({ input, source_language_code: languageCode, target_language_code: 'en-IN' }),
      }, 20_000)
      if (!res.ok) throw new Error(`${res.status}`)
      const data = await res.json() as { transliterated_text?: string }
      const text = data.transliterated_text ?? ''

      // Split on [N] markers to recover per-word text
      const segments = text.split(/\[\d+\]/).map(s => s.trim())
      // segments[0] is before [0] (usually empty), segments[1] is after [0], etc.
      for (let j = 0; j < batch.length; j++) {
        const raw = segments[j + 1] ?? ''
        const word = batch[j].word.trim()

        // If the original was ASCII, keep it lowercased directly
        const roman = isAscii(word)
          ? word.toLowerCase().replace(/[.,!?।]/g, '').trim()
          : stripDiacritics(raw).replace(/[.,!?।\[\]]/g, '').trim()

        if (roman && /[a-zA-Z]/.test(roman)) result[i + j] = roman
      }
    } catch (err) {
      // Fall back to per-word calls on failure
      console.warn(`[transliterate] Sarvam sentence batch ${i} failed (${err}), falling back to per-word`)
      await Promise.all(batch.map(async (entry, j) => {
        const word = entry.word.trim()
        if (isAscii(word)) { result[i + j] = word.toLowerCase().replace(/[.,!?।]/g, '') || undefined; return }
        try {
          const res2 = await fetchWithTimeout('https://api.sarvam.ai/transliterate', {
            method: 'POST',
            headers: { 'api-subscription-key': apiKey, 'Content-Type': 'application/json' },
            body: JSON.stringify({ input: word, source_language_code: languageCode, target_language_code: 'en-IN' }),
          }, 12_000)
          if (!res2.ok) return
          const d2 = await res2.json() as { transliterated_text?: string }
          const r2 = stripDiacritics((d2.transliterated_text ?? '').trim()).replace(/[.,!?।]/g, '').trim()
          if (r2 && /[a-zA-Z]/.test(r2)) result[i + j] = r2
        } catch { /* skip */ }
      }))
    }
  }

  console.log(`[transliterate] ${languageCode} → Roman via Sarvam for ${entries.length} word(s)`)
  return result
}

async function transliterateToRoman(
  entries: SarvamWord[],
  languageCode: string,
  sarvamApiKey?: string,
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

  // Prefer OpenAI GPT (most natural phrasing)
  if (OPENAI_API_KEY) return transliterateWithLLM(entries, lang, OPENAI_API_KEY)

  // Deterministic fallbacks: rule-based for Telugu, Sarvam API for other languages
  const fallback = async (): Promise<(string | undefined)[]> => {
    if (lang === 'te-IN' || lang === 'te') {
      return entries.map(e => {
        const word = e.word.trim()
        if (!word || !/[ఀ-౿]/.test(word)) return /[a-zA-Z]/.test(word) ? word.toLowerCase() : undefined
        const roman = transliterateTeluguWord(word).replace(/[.,!?।]/g, '').trim()
        return roman || undefined
      })
    }
    if (sarvamApiKey) return transliterateWithSarvam(entries, lang, sarvamApiKey)
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
