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
const timeoutFor = (type: string) => (type === 'proxy' ? PROXY_TIMEOUT_MS : JOB_TIMEOUT_MS)

export function registerHandler(type: JobType, handler: JobHandler) {
  handlers.set(type, handler)
}

// Several workers share this queue (Railway, and a local one during development), so a worker
// that starts must not take back jobs another worker is still running. A job only counts as
// stuck once it has been "processing" longer than the job timeout (plus a minute): its worker
// died or was restarted mid-job. updated_at is set by the jobs_updated_at trigger on claim.
const STUCK_AFTER_MS = JOB_TIMEOUT_MS + 60_000
const PROXY_STUCK_AFTER_MS = PROXY_TIMEOUT_MS + 60_000
const STUCK_SWEEP_MS = 5 * 60 * 1000

async function requeueStuck() {
  const stuck = await db`
    UPDATE jobs SET status = 'queued'
    WHERE status = 'processing' AND updated_at < now()
      - ((CASE WHEN type = 'proxy' THEN ${PROXY_STUCK_AFTER_MS}::int ELSE ${STUCK_AFTER_MS}::int END) * interval '1 millisecond')
    RETURNING id
  `
  if (stuck.length) console.log(`[queue] Re-queued ${stuck.length} stuck job(s) (processing for over ${Math.round(STUCK_AFTER_MS / 60000)} min)`)
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
  const types = [...handlers.keys()]
  const [job] = await db<Job[]>`
    SELECT * FROM jobs
    WHERE status = 'queued' AND type = ANY(${types})
    ${isRailway ? db`AND NOT (type = 'transcribe' AND payload @> '{"requires_ytdlp":true}'::jsonb)` : db``}
    AND NOT (type = 'proxy' AND EXISTS (SELECT 1 FROM jobs p WHERE p.type = 'proxy' AND p.status = 'processing'))
    ORDER BY (type = 'proxy') ASC, created_at ASC LIMIT 1
  `
  if (!job) return

  const claimed = await db`
    UPDATE jobs SET status = 'processing' WHERE id = ${job.id} AND status = 'queued' RETURNING id
  `
  if (!claimed.length) return

  console.log(`[queue] Processing job ${job.id} (${job.type})`)
  const handler = handlers.get(job.type as JobType)
  if (!handler) { await fail(job.id, `No handler for type: ${job.type}`); return }

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
  }
}

async function fail(jobId: string, error: string) {
  await db`UPDATE jobs SET status = 'failed', error = ${error} WHERE id = ${jobId}`
}

function sleep(ms: number) { return new Promise(resolve => setTimeout(resolve, ms)) }
