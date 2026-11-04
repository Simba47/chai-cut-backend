/**
 * Reference transcripts in the worker (rules: lib/referenceTranscript.ts; plan: 1 Nov 2026).
 *
 * Nothing here runs while REFERENCE_TRANSCRIBER is off (the default). In shadow mode:
 *   - after an upload's audio is kept (jobs/proxy.ts), its audio goes to the transcriber service
 *   - a background check (not a queue job: it holds no job slot) collects finished transcripts
 *   - when Make my clips picks moments, the same picker runs on the reference transcript in the
 *     background and both lists are saved for comparison. Users still get today's clips.
 * On: the same, and Make my clips picks from the reference transcript (jobs/ai_edit.ts waitForWords).
 * Its own tables only: reference_transcripts and reference_compare.
 */
import { HeadObjectCommand } from '@aws-sdk/client-s3'
import db from '../db.js'
import { r2, R2_BUCKET, r2SignedUrl } from '../r2.js'
import type { FoundClip } from '../lib/clipFinder.js'
import {
  referenceMode, submitReferenceTranscript, checkReferenceTranscript, momentAgreement, type ReferenceWord,
} from '../lib/referenceTranscript.js'
import { audioKeyFor } from './proxy.js'

/** How long a submitted transcript may take before it counts as failed */
const GIVE_UP_MS = 3 * 60 * 60_000
/** How often the background check runs, and how many tickets it looks at each time */
const POLL_MS = 20_000
const POLL_BATCH = 10

export async function ensureReferenceSchema(): Promise<void> {
  await db`
    CREATE TABLE IF NOT EXISTS reference_transcripts (
      video_id uuid PRIMARY KEY REFERENCES videos(id) ON DELETE CASCADE,
      status text NOT NULL CHECK (status IN ('submitted', 'done', 'failed')),
      call_id text,
      model text,
      language text,
      words jsonb,
      audio_seconds real,
      gpu_seconds real,
      error text,
      submitted_at timestamptz NOT NULL DEFAULT now(),
      checked_at timestamptz,
      done_at timestamptz
    )`
  await db`CREATE INDEX IF NOT EXISTS reference_transcripts_open_idx ON reference_transcripts (submitted_at) WHERE status = 'submitted'`
  await db`
    CREATE TABLE IF NOT EXISTS reference_compare (
      id bigserial PRIMARY KEY,
      video_id uuid NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
      baseline jsonb NOT NULL,
      reference jsonb NOT NULL,
      agreement jsonb NOT NULL,
      created_at timestamptz NOT NULL DEFAULT now()
    )`
}

/**
 * After an upload's audio is kept: hand it to the transcriber (once per video). Best effort: a
 * failure is recorded and logged, and never affects the upload.
 */
export async function submitReferenceFor(videoId: string, storagePath: string): Promise<void> {
  if (referenceMode() === 'off' || !process.env.TRANSCRIBER_URL) return
  const [existing] = await db`SELECT status FROM reference_transcripts WHERE video_id = ${videoId}`
  if (existing && existing.status !== 'failed') return
  const key = audioKeyFor(storagePath)
  const kept = await r2.send(new HeadObjectCommand({ Bucket: R2_BUCKET, Key: key })).then(() => true, () => false)
  if (!kept) return  // no sound in the video
  try {
    const callId = await submitReferenceTranscript(await r2SignedUrl(key, 6 * 60 * 60), null)
    await db`
      INSERT INTO reference_transcripts (video_id, status, call_id)
      VALUES (${videoId}, 'submitted', ${callId})
      ON CONFLICT (video_id) DO UPDATE SET status = 'submitted', call_id = EXCLUDED.call_id, error = NULL,
        submitted_at = now(), checked_at = NULL, done_at = NULL`
    console.log(`[reference] video ${videoId}: audio sent to the transcriber (${callId})`)
  } catch (e) {
    const error = (e instanceof Error ? e.message : String(e)).slice(0, 500)
    await db`
      INSERT INTO reference_transcripts (video_id, status, error) VALUES (${videoId}, 'failed', ${error})
      ON CONFLICT (video_id) DO UPDATE SET status = 'failed', error = EXCLUDED.error`
    console.warn(`[reference] video ${videoId}: not sent: ${error}`)
  }
}

export type ReferenceTranscript =
  | { state: 'none' | 'pending' | 'failed' }
  | { state: 'done'; words: ReferenceWord[]; language: string | null }

/** A video's reference transcript, for Make my clips (switch on) */
export async function referenceTranscriptOf(videoId: string): Promise<ReferenceTranscript> {
  const [row] = await db<{ status: string; words: ReferenceWord[] | null; language: string | null }[]>`
    SELECT status, CASE WHEN status = 'done' THEN words END AS words, language
    FROM reference_transcripts WHERE video_id = ${videoId}`
  if (!row) return { state: 'none' }
  if (row.status === 'submitted') return { state: 'pending' }
  // a transcript with no words (music only) is no use for picking moments: today's way
  if (row.status === 'done' && row.words?.length) return { state: 'done', words: row.words, language: row.language }
  return { state: 'failed' }
}

/** One round of the background check: collect finished transcripts, give up on stuck ones */
export async function pollReferenceTranscripts(): Promise<void> {
  // Claim a few tickets (several workers may poll: each ticket goes to one of them per round)
  const rows = await db<{ video_id: string; call_id: string; submitted_at: Date }[]>`
    UPDATE reference_transcripts SET checked_at = now()
    WHERE video_id IN (
      SELECT video_id FROM reference_transcripts
      WHERE status = 'submitted' AND (checked_at IS NULL OR checked_at < now() - interval '15 seconds')
      ORDER BY submitted_at LIMIT ${POLL_BATCH} FOR UPDATE SKIP LOCKED
    )
    RETURNING video_id, call_id, submitted_at`
  for (const r of rows) {
    if (Date.now() - new Date(r.submitted_at).getTime() > GIVE_UP_MS) {
      await db`UPDATE reference_transcripts SET status = 'failed', error = 'took too long' WHERE video_id = ${r.video_id}`
      continue
    }
    const check = await checkReferenceTranscript(r.call_id).catch(e => ({ status: 'pending' as const, note: String(e) }))
    if (check.status === 'done') {
      const { reply } = check
      await db`
        UPDATE reference_transcripts SET status = 'done', words = ${db.json(reply.words as never)}, model = ${reply.model},
          language = ${reply.language}, audio_seconds = ${reply.audioSeconds}, gpu_seconds = ${reply.gpuSeconds}, done_at = now()
        WHERE video_id = ${r.video_id}`
      console.log(`[reference] video ${r.video_id}: ${reply.words.length} words (${reply.model}, ${reply.gpuSeconds ?? '?'} GPU s)`)
    } else if (check.status === 'failed') {
      await db`UPDATE reference_transcripts SET status = 'failed', error = ${check.error.slice(0, 500)} WHERE video_id = ${r.video_id}`
      console.warn(`[reference] video ${r.video_id}: failed: ${check.error}`)
    }
  }
}

/** Start the background check (once per worker; only when the switch is not off) */
let polling: NodeJS.Timeout | null = null
export function startReferencePoller(): void {
  if (polling || referenceMode() === 'off' || !process.env.TRANSCRIBER_URL) return
  let busy = false
  polling = setInterval(() => {
    if (busy) return
    busy = true
    pollReferenceTranscripts()
      .catch(e => console.warn('[reference] check failed:', e instanceof Error ? e.message : e))
      .finally(() => { busy = false })
  }, POLL_MS)
  polling.unref()
  console.log(`[reference] transcriber on (${referenceMode()}): checking for finished transcripts every ${POLL_MS / 1000} s`)
}

/**
 * Shadow mode: the moments Make my clips just picked (from today's transcript) against the moments
 * the same picker finds in the reference transcript. `pick` is Make my clips' own picker with this
 * run's settings, so both lists are made the same way. Saved for comparison; never changes the
 * clips. Called without waiting: errors are logged only.
 */
export async function shadowCompareMoments(
  videoId: string,
  baseline: FoundClip[],
  pick: (words: ReferenceWord[], durationMs: number) => Promise<FoundClip[]>,
): Promise<void> {
  if (referenceMode() !== 'shadow') return
  const [row] = await db<{ words: ReferenceWord[] }[]>`SELECT words FROM reference_transcripts WHERE video_id = ${videoId} AND status = 'done'`
  if (!row?.words?.length) return
  const reference = await pick(row.words, row.words[row.words.length - 1].end_ms)
  const slim = (cs: FoundClip[]) => cs.map(c => ({ start_ms: c.start_ms, end_ms: c.end_ms, score: c.score, title: c.title }))
  const agreement = momentAgreement(baseline, reference)
  await db`
    INSERT INTO reference_compare (video_id, baseline, reference, agreement)
    VALUES (${videoId}, ${db.json(slim(baseline) as never)}, ${db.json(slim(reference) as never)}, ${db.json(agreement as never)})`
  console.log(`[reference] video ${videoId}: shadow picks: ${Math.round(agreement.baselineFound * 100)}% of today's picks also found from the reference transcript`)
}
