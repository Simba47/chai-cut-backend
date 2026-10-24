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
// Reading a whole video (AI edit) grows with its length: a 4-hour video must not be stopped part-way
const READING_TIMEOUT_MS = Math.max(JOB_TIMEOUT_MS, parseInt(process.env.READING_TIMEOUT_MS ?? String(60 * 60 * 1000), 10))
const timeoutFor = (job: Job) => (job.type === 'proxy' ? PROXY_TIMEOUT_MS : job.type === 'track_subjects' ? TRACK_TIMEOUT_MS
  : isWholeReading(job) ? READING_TIMEOUT_MS : JOB_TIMEOUT_MS)

/** A whole-video reading (AI edit's captions): many minutes for a long video */
function isWholeReading(job: Job): boolean {
  const p = (typeof job.payload === 'string' ? JSON.parse(job.payload) : job.payload) as { transcribe_full?: unknown }
  return job.type === 'transcribe' && p?.transcribe_full === true
}
/** A long job: a whole-video reading, or an editing copy (proxy) — both many minutes for a long video */
function isLong(job: Job): boolean {
  return job.type === 'proxy' || isWholeReading(job)
}

// Long jobs — whole-video readings (started as soon as someone picks AI edit) and editing copies
// (proxy) — may take every slot but one together: with more than one slot, one is always kept for
// the quick jobs people wait on (a clip's captions, renders, Subjects), so they never queue behind
// them. (User, 2026-10-24: a 2-hour upload's editing copy and its reading took both slots, and a
// Subjects job waited 7+ minutes — editing copies weren't counted.) AI edit itself is not held
// back: it waits for its video's reading, which must be able to start next to it. Editing copies
// also wait for everything else queued and are made one at a time.
const LONG_SLOTS = CONCURRENCY > 1 ? CONCURRENCY - 1 : 1
let longRunning = 0

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

/** How long the queue is (every worker logs it): when the oldest job has waited over WAIT_WARN_MS, more workers are needed */
const WAIT_WARN_MS = 5 * 60 * 1000
async function reportQueue() {
  // (editing copies count among the running; a queued one waits for everything else by design, so
  // it isn't counted as waiting)
  const [q] = await db<{ waiting: number; running: number; copies: number; oldest_s: number | null }[]>`
    SELECT count(*) FILTER (WHERE status = 'queued' AND type <> 'proxy')::int AS waiting,
      count(*) FILTER (WHERE status = 'processing')::int AS running,
      count(*) FILTER (WHERE status = 'processing' AND type = 'proxy')::int AS copies,
      extract(epoch FROM now() - min(created_at) FILTER (WHERE status = 'queued' AND type <> 'proxy'))::int AS oldest_s
    FROM jobs WHERE status IN ('queued', 'processing')`
  if (!q || !q.waiting) return
  const line = `[queue] ${q.waiting} job(s) waiting, ${q.running} running${q.copies ? ` (${q.copies} an editing copy)` : ''}; the oldest has waited ${q.oldest_s ?? 0} s`
  if ((q.oldest_s ?? 0) * 1000 > WAIT_WARN_MS) console.warn(`${line} — more workers needed`)
  else console.log(line)
}

export async function startQueue() {
  await requeueStuck()
  // Keep rescuing jobs whose worker died, not only when this one starts
  setInterval(() => { requeueStuck().catch(err => console.error('[queue] Stuck-job sweep failed:', err)) }, STUCK_SWEEP_MS)
  setInterval(() => { reportQueue().catch(() => { /* only a log line */ }) }, STUCK_SWEEP_MS)

  console.log(`[queue] Worker started (concurrency=${CONCURRENCY}, long jobs (whole-video readings, editing copies) at most ${LONG_SLOTS} at a time), polling for jobs…`)
  await Promise.all(Array.from({ length: CONCURRENCY }, (_, i) => runLoop(i)))
}

async function runLoop(id: number) {
  while (true) {
    // Straight on to the next job after one is done; wait only when there was nothing to take
    let ran = false
    try { ran = await tick() } catch (err) { console.error(`[queue] Unexpected error in loop ${id}:`, err) }
    if (!ran) await sleep(JOB_POLL_INTERVAL_MS)
  }
}

// Railway sets RAILWAY_ENVIRONMENT automatically. If present, skip jobs that require
// a residential IP (yt-dlp YouTube downloads) — those must run on the local worker.
// 'ai_edit' runs anywhere: it reads the video from storage and never uses yt-dlp.
const isRailway = !!process.env.RAILWAY_ENVIRONMENT

/** Takes and runs the next job; false when there was none */
async function tick(): Promise<boolean> {
  // Only jobs this worker can run (a worker from before a new kind of job leaves those to others).
  // Editing copies (proxy) wait for everything else, and only one is made at a time: renders and
  // captions always have a slot.
  // Picked and claimed in one statement. SKIP LOCKED: with many workers (replicas × CONCURRENCY
  // loops) each takes the next job nobody else is taking, instead of all reading the same oldest
  // job, one winning and the rest sleeping a poll interval for nothing. The heartbeat starts in
  // the same statement, as before.
  const types = [...handlers.keys()]
  // A long job may be taken only while a slot is left for quick jobs (counted before the claim, so
  // two loops never both take the last one)
  const mayLong = longRunning < LONG_SLOTS
  if (mayLong) longRunning++
  let job: Job | undefined
  try {
    job = await claim(types, mayLong)
  } finally {
    if (mayLong && !(job && isLong(job))) longRunning--
  }
  if (!job) return false
  try {
    await run(job)
  } finally {
    if (mayLong && isLong(job)) longRunning--
  }
  return true
}

async function claim(types: string[], mayLong: boolean): Promise<Job | undefined> {
  const [job] = await db<Job[]>`
    WITH next AS (
      SELECT id FROM jobs
      WHERE status = 'queued' AND type = ANY(${types})
      ${isRailway ? db`AND NOT (type = 'transcribe' AND payload @> '{"requires_ytdlp":true}'::jsonb)` : db``}
      AND NOT (type = 'proxy' AND EXISTS (SELECT 1 FROM jobs p WHERE p.type = 'proxy' AND p.status = 'processing'))
      ${mayLong ? db`` : db`AND NOT (type = 'proxy' OR (type = 'transcribe' AND COALESCE(payload->>'transcribe_full', '') = 'true'))`}
      -- One whole-video reading per person at a time (across all workers): one person's videos
      -- never take every slot while someone else waits
      AND NOT (type = 'transcribe' AND COALESCE(payload->>'transcribe_full', '') = 'true' AND EXISTS (
        SELECT 1 FROM jobs p
        JOIN videos pv ON pv.id::text = p.payload->>'video_id'
        JOIN videos v ON v.id::text = jobs.payload->>'video_id'
        WHERE p.type = 'transcribe' AND p.status = 'processing' AND COALESCE(p.payload->>'transcribe_full', '') = 'true'
          AND pv.user_id = v.user_id))
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
  return job
}

async function run(job: Job) {
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
  const timeoutMs = timeoutFor(job)
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
