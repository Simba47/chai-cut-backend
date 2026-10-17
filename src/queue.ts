import db from './db.js'
import type { Job, JobType } from './types.js'
import { JOB_POLL_INTERVAL_MS } from './types.js'

/** `signal` aborts when the job runs past JOB_TIMEOUT_MS: handlers stop their child processes on it */
type JobHandler = (job: Job, signal: AbortSignal) => Promise<void>
const handlers = new Map<JobType, JobHandler>()

const CONCURRENCY = parseInt(process.env.QUEUE_CONCURRENCY ?? '2', 10)
const JOB_TIMEOUT_MS = parseInt(process.env.JOB_TIMEOUT_MS ?? String(15 * 60 * 1000), 10)
// An editing copy (jobs/proxy.ts) of a long video takes longer than other jobs: its own limit
const PROXY_TIMEOUT_MS = parseInt(process.env.PROXY_TIMEOUT_MS ?? String(90 * 60 * 1000), 10)
// Following people (Subjects, Sub-edit) takes well under a minute for a clip (it reads only the
// clip's part of the video): one that runs this long has hung, and is stopped instead of holding
// a worker slot for an hour
const TRACK_TIMEOUT_MS = parseInt(process.env.TRACK_TIMEOUT_MS ?? String(8 * 60 * 1000), 10)
const timeoutFor = (type: string) => (type === 'proxy' ? PROXY_TIMEOUT_MS : type === 'track_subjects' ? TRACK_TIMEOUT_MS : JOB_TIMEOUT_MS)

export function registerHandler(type: JobType, handler: JobHandler) {
  handlers.set(type, handler)
}

// Several workers share this queue (Railway — two at once during a deploy — and a local one in
// development), so a worker must never take back a job another worker is still running. Each
// worker stamps its running jobs' heartbeat (job_heartbeats) every HEARTBEAT_MS; a "processing" job
// whose heartbeat stopped for STUCK_AFTER_MS lost its worker (it died, or a deploy restarted it
// mid-job) and goes back to the queue — in a few minutes, not an hour. A job claimed by a worker
// from before heartbeats (none) keeps the old rule: stuck once it has been "processing" longer than
// its timeout plus a minute (updated_at is set by the jobs_updated_at trigger on claim).
const HEARTBEAT_MS = 30_000
const STUCK_AFTER_MS = 3 * 60 * 1000
const STUCK_SWEEP_MS = 60 * 1000
const NO_HEARTBEAT_STUCK_MS = JOB_TIMEOUT_MS + 60_000
const NO_HEARTBEAT_PROXY_STUCK_MS = PROXY_TIMEOUT_MS + 60_000

async function requeueStuck() {
  const stuck = await db`
    WITH back AS (
      UPDATE jobs j SET status = 'queued'
      FROM jobs o LEFT JOIN job_heartbeats h ON h.job_id = o.id
      WHERE j.id = o.id AND o.status = 'processing' AND (
        h.at < now() - (${STUCK_AFTER_MS}::int * interval '1 millisecond')
        OR (h.job_id IS NULL AND o.updated_at < now()
          - ((CASE WHEN o.type = 'proxy' THEN ${NO_HEARTBEAT_PROXY_STUCK_MS}::int ELSE ${NO_HEARTBEAT_STUCK_MS}::int END) * interval '1 millisecond'))
      )
      RETURNING j.id
    )
    DELETE FROM job_heartbeats WHERE job_id IN (SELECT id FROM back)
    RETURNING job_id
  `
  if (stuck.length) console.log(`[queue] Re-queued ${stuck.length} job(s) whose worker stopped`)
}

export async function startQueue() {
  await requeueStuck()
  // Keep rescuing jobs whose worker died, not only when this one starts
  setInterval(() => { requeueStuck().catch(err => console.error('[queue] Stuck-job sweep failed:', err)) }, STUCK_SWEEP_MS)

  console.log(`[queue] Worker started (concurrency=${CONCURRENCY}), polling for jobs…`)
  await Promise.all(Array.from({ length: CONCURRENCY }, (_, i) => runLoop(i)))
}

async function runLoop(id: number) {
  while (true) {
    try { await tick() } catch (err) { console.error(`[queue] Unexpected error in loop ${id}:`, err) }
    await sleep(JOB_POLL_INTERVAL_MS)
  }
}

// Railway sets RAILWAY_ENVIRONMENT automatically. If present, skip jobs that require
// a residential IP (yt-dlp YouTube downloads) — those must run on the local worker.
// 'ai_edit' runs anywhere: it reads the video from storage and never uses yt-dlp.
const isRailway = !!process.env.RAILWAY_ENVIRONMENT

async function tick() {
  // Only jobs this worker can run (a worker from before a new kind of job leaves those to others).
  // Editing copies (proxy) wait for everything else, and only one is made at a time: renders and
  // captions always have a slot.
  // Picked and claimed in one statement. SKIP LOCKED: with many workers (replicas × CONCURRENCY
  // loops) each takes the next job nobody else is taking, instead of all reading the same oldest
  // job, one winning and the rest sleeping a poll interval for nothing. The heartbeat starts in
  // the same statement, as before.
  const types = [...handlers.keys()]
  const [job] = await db<Job[]>`
    WITH next AS (
      SELECT id FROM jobs
      WHERE status = 'queued' AND type = ANY(${types})
      ${isRailway ? db`AND NOT (type = 'transcribe' AND payload @> '{"requires_ytdlp":true}'::jsonb)` : db``}
      AND NOT (type = 'proxy' AND EXISTS (SELECT 1 FROM jobs p WHERE p.type = 'proxy' AND p.status = 'processing'))
      -- A whole-video reading (AI edit: many minutes for a long video) waits for the quick jobs that
      -- came after it — a clip's own captions (manual edit), renders, uploads — so they never queue
      -- behind it (COALESCE: a job without the field must sort as false, not as NULL, which goes last)
      ORDER BY (type = 'proxy') ASC, (type = 'transcribe' AND COALESCE(payload->>'transcribe_full', '') = 'true') ASC, created_at ASC LIMIT 1
      FOR UPDATE SKIP LOCKED
    ), claimed AS (
      UPDATE jobs j SET status = 'processing' FROM next WHERE j.id = next.id AND j.status = 'queued' RETURNING j.*
    ), beat AS (
      INSERT INTO job_heartbeats (job_id) SELECT id FROM claimed
      ON CONFLICT (job_id) DO UPDATE SET at = now()
    )
    SELECT * FROM claimed
  `
  if (!job) return

  console.log(`[queue] Processing job ${job.id} (${job.type})`)
  const handler = handlers.get(job.type as JobType)
  if (!handler) { await fail(job.id, `No handler for type: ${job.type}`); return }
  // The heartbeat: this job's worker is alive (see requeueStuck)
  const beat = setInterval(() => {
    db`UPDATE job_heartbeats SET at = now() WHERE job_id = ${job.id}`
      .catch(err => console.error(`[queue] Heartbeat for job ${job.id} failed:`, err instanceof Error ? err.message : err))
  }, HEARTBEAT_MS)

  // On timeout the job is marked failed straight away (so the stuck-job sweep never runs it a
  // second time) and told to stop. This loop still waits for the handler to actually finish:
  // racing it against the timer used to free the slot while its ffmpeg/Python kept running, so the
  // worker ran more jobs than CONCURRENCY and a render's clip stayed "rendering".
  const ctrl = new AbortController()
  let timedOut = false
  const timeoutMs = timeoutFor(job.type)
  const timer = setTimeout(() => {
    timedOut = true
    const msg = `job timed out after ${timeoutMs / 1000}s`
    console.error(`[queue] Job ${job.id} ${msg}, stopping it`)
    ctrl.abort(new Error(msg))
    fail(job.id, msg).catch(err => console.error(`[queue] Could not mark job ${job.id} failed:`, err))
  }, timeoutMs)

  try {
    await handler(job, ctrl.signal)
    if (timedOut) { console.warn(`[queue] Job ${job.id} finished after its timeout (left as failed)`); return }
    await db`UPDATE jobs SET status = 'done' WHERE id = ${job.id}`
    console.log(`[queue] Job ${job.id} done`)
  } catch (err) {
    if (timedOut) { console.error(`[queue] Job ${job.id} stopped after its timeout`); return }
    const msg = err instanceof Error ? err.message : String(err)
    console.error(`[queue] Job ${job.id} failed:`, msg)
    await fail(job.id, msg)
  } finally {
    clearTimeout(timer)
    clearInterval(beat)
    await db`DELETE FROM job_heartbeats WHERE job_id = ${job.id}`.catch(() => {})
  }
}

async function fail(jobId: string, error: string) {
  await db`UPDATE jobs SET status = 'failed', error = ${error} WHERE id = ${jobId}`
}

function sleep(ms: number) { return new Promise(resolve => setTimeout(resolve, ms)) }
