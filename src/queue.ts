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
// also wait for everything else queued.
let LONG_SLOTS = CONCURRENCY > 1 ? CONCURRENCY - 1 : 1
let longRunning = 0

// Editing copies one worker makes at once (PROXY_PER_WORKER, default 1), within its long-job slots.
// Per worker, not across all of them: more workers make more copies (it used to be one at a time
// across every worker, so at ~200 uploads an hour copies queued for hours).
const PROXY_PER_WORKER = Math.max(1, parseInt(process.env.PROXY_PER_WORKER ?? '1', 10) || 1)
let proxyRunning = 0

// ── Worker pools ─────────────────────────────────────────────────────────────
// WORKER_JOBS: the jobs this worker takes, so each kind can have its own machines, e.g.
// "render,track_subjects,detect_silence" (exports and Subjects on big machines) or "proxy" (editing
// copies on their own). "transcribe" is a clip's captions and link imports; "reading" is a whole-video
// reading (AI edit). Not set: every job, as before.
const READING = 'reading'
interface Pool { types: string[]; clipCaptions: boolean; readings: boolean; label: string }
let pool: Pool | null = null
function workerPool(known: string[]): Pool {
  const raw = process.env.WORKER_JOBS?.trim()
  if (!raw) return { types: known, clipCaptions: true, readings: true, label: 'every job' }
  const asked = raw.split(',').map(s => s.trim()).filter(Boolean)
  const unknown = asked.filter(a => a !== READING && !known.includes(a))
  if (unknown.length) console.warn(`[queue] WORKER_JOBS: no such job ${unknown.join(', ')} (known: ${[...known, READING].join(', ')})`)
  const clipCaptions = asked.includes('transcribe'), readings = asked.includes(READING)
  const types = known.filter(t => t !== 'transcribe' && asked.includes(t))
  if (clipCaptions || readings) types.push('transcribe')
  if (!types.length) console.error('[queue] WORKER_JOBS names no job this worker can run: it will take nothing')
  return { types, clipCaptions, readings, label: asked.filter(a => !unknown.includes(a)).join(', ') || 'nothing' }
}
/** Whether this worker's jobs include any quick one (else no slot needs keeping free for them) */
const takesQuickJobs = (p: Pool) => p.clipCaptions || p.types.some(t => t !== 'proxy' && t !== 'transcribe')

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

  pool = workerPool([...handlers.keys()])
  // A worker of long jobs only (editing copies, readings) has no quick job to keep a slot for
  if (!takesQuickJobs(pool)) LONG_SLOTS = CONCURRENCY
  const copies = Math.min(PROXY_PER_WORKER, LONG_SLOTS)
  if (copies < PROXY_PER_WORKER && pool.types.includes('proxy')) console.warn(`[queue] PROXY_PER_WORKER=${PROXY_PER_WORKER} but only ${LONG_SLOTS} slot(s) for long jobs: ${copies} editing copies at a time (raise QUEUE_CONCURRENCY)`)
  await listenForJobs(pool)
  console.log(`[queue] Worker started (concurrency=${CONCURRENCY}, takes: ${pool.label}; long jobs (whole-video readings, editing copies) at most ${LONG_SLOTS} at a time, editing copies at most ${copies}; per person at once: ${EXPORTS_PER_PERSON} exports, ${SUBJECTS_PER_PERSON} Subjects), ${listening ? 'new jobs signalled' : 'polling for jobs'}…`)
  await Promise.all(Array.from({ length: CONCURRENCY }, (_, i) => runLoop(i)))
}

async function runLoop(id: number) {
  while (true) {
    // Straight on to the next job after one is done; wait only when there was nothing to take
    let ran = false
    try { ran = await tick() } catch (err) { console.error(`[queue] Unexpected error in loop ${id}:`, err) }
    if (!ran) await waitForJob(listening ? FALLBACK_POLL_MS : JOB_POLL_INTERVAL_MS)
  }
}

// ── New-job signal ───────────────────────────────────────────────────────────
// A trigger on jobs signals (NOTIFY jobs_new, with the job's type) when a job is queued: added, or
// put back by requeueStuck. Each worker listens and wakes one idle slot per job of its kinds, so a
// job starts at once instead of at the next poll (up to 2 s later, and every idle slot of every
// worker polling every 2 s). Signals can be missed (the listening connection dropped for a moment),
// so idle slots still look every FALLBACK_POLL_MS; without the signal at all, every 2 s as before.
// The signal needs a direct connection to Postgres: PgBouncer in transaction mode doesn't pass it on.
const FALLBACK_POLL_MS = 10_000
const SIGNAL = 'jobs_new'
// Creating the trigger takes its turn (workers starting together would both make it)
const SIGNAL_LOCK = 20261028
let listening = false
const idle: Array<() => void> = []

/** Waits until a job of this worker's kinds is signalled, or `ms` passes */
function waitForJob(ms: number): Promise<void> {
  return new Promise(resolve => {
    const done = () => { clearTimeout(timer); const i = idle.indexOf(done); if (i >= 0) idle.splice(i, 1); resolve() }
    const timer = setTimeout(done, ms)
    idle.push(done)
  })
}

async function listenForJobs(p: Pool) {
  try {
    await db.begin(async tx => {
      await tx`SELECT pg_advisory_xact_lock(${SIGNAL_LOCK})`
      await tx`
        CREATE OR REPLACE FUNCTION jobs_notify_new() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN PERFORM pg_notify('jobs_new', NEW.type); RETURN NULL; END $$`
      // Made only when missing: replacing a trigger locks the jobs table
      const [made] = await tx`SELECT 1 FROM pg_trigger WHERE tgname = 'jobs_new_job' AND tgrelid = 'jobs'::regclass`
      if (!made) await tx`
        CREATE TRIGGER jobs_new_job AFTER INSERT OR UPDATE OF status ON jobs
        FOR EACH ROW WHEN (NEW.status = 'queued') EXECUTE FUNCTION jobs_notify_new()`
    })
    // (postgres.js keeps one connection for this and listens again after it reconnects; on each
    // (re)start one idle slot looks, for any signal missed meanwhile)
    await db.listen(SIGNAL, type => { if (p.types.includes(type)) idle.shift()?.() }, () => { listening = true; idle.shift()?.() })
  } catch (err) {
    listening = false
    console.warn(`[queue] No new-job signal (${err instanceof Error ? err.message : err}): looking for jobs every ${JOB_POLL_INTERVAL_MS / 1000} s instead`)
  }
}

// Railway sets RAILWAY_ENVIRONMENT automatically. If present, skip jobs that require
// a residential IP (yt-dlp YouTube downloads) — those must run on the local worker.
// 'ai_edit' runs anywhere: it reads the video from storage and never uses yt-dlp.
const isRailway = !!process.env.RAILWAY_ENVIRONMENT

/** Takes and runs the next job; false when there was none (true: look again now) */
async function tick(): Promise<boolean> {
  // Only jobs this worker can run (a worker from before a new kind of job leaves those to others).
  // Editing copies (proxy) wait for everything else, at most PROXY_PER_WORKER at a time: renders and
  // captions always have a slot.
  // Picked and claimed in one statement. SKIP LOCKED: with many workers (replicas × CONCURRENCY
  // loops) each takes the next job nobody else is taking, instead of all reading the same oldest
  // job, one winning and the rest sleeping a poll interval for nothing. The heartbeat starts in
  // the same statement, as before.
  const p = pool ?? workerPool([...handlers.keys()])
  // A long job may be taken only while a slot is left for quick jobs (counted before the claim, so
  // two loops never both take the last one)
  const mayLong = longRunning < LONG_SLOTS
  if (mayLong) longRunning++
  const mayProxy = mayLong && proxyRunning < PROXY_PER_WORKER
  if (mayProxy) proxyRunning++
  let job: Job | undefined
  try {
    job = await claim(p, mayLong, mayProxy)
  } finally {
    if (mayLong && !(job && isLong(job))) longRunning--
    if (mayProxy && job?.type !== 'proxy') proxyRunning--
  }
  // Nothing, but only because another slot was holding a long-job place while it picked (and has
  // let go of it): look again now, not at the next poll
  if (!job) return (!mayLong && longRunning < LONG_SLOTS) || (mayLong && !mayProxy && proxyRunning < PROXY_PER_WORKER)
  busyHere++
  // One more idle slot looks too: signals for jobs added together arrive as one
  idle.shift()?.()
  try {
    await run(job)
  } finally {
    busyHere--
    if (mayLong && isLong(job)) longRunning--
    if (mayProxy && job.type === 'proxy') proxyRunning--
  }
  return true
}

// ── Spreading jobs across machines ───────────────────────────────────────────
// Every worker hears every new job, and the first to claim it wins, so a machine already running
// jobs took new ones as readily as an idle one. Exports on one machine slow each other (one ~21 s
// alone, ~35 s with 4 at once; on 2026-10-28 a busy machine took several bursts while others sat
// idle). So a worker waits SPREAD_MS for each job it is running or claiming (at most SPREAD_MAX_MS)
// before it claims: an idle machine claims at once and gets the job first. Nobody else took it
// meanwhile: this worker still does. 100 ms is far more than one machine's head start in hearing a
// job, and adds ~0.2 s to a start (tested: 3 machines × 4, uneven starts 12–16% → 0–1%; 250 ms
// spread no better and added ~0.5 s). SPREAD_MS=0: no waiting.
const SPREAD_MS = Math.max(0, parseInt(process.env.SPREAD_MS ?? '100', 10) || 0)
const SPREAD_MAX_MS = 400
let busyHere = 0   // jobs this worker is running or claiming

// ── Fair limits per person ───────────────────────────────────────────────────
// Running at once for one person, across all workers (the person is the job's payload.user_id, set
// by the website): their next ones wait their turn, nothing is refused. Exports: Make my clips gives
// up to 10 clips, so all 10 can export together (EXPORTS_PER_PERSON). Subjects / Sub-edit: one at a
// time (SUBJECTS_PER_PERSON). Among the waiting, the person with fewest of that kind running goes
// first, so one person's 10 exports never hold up someone else's one. (A job from before user_id was
// sent has no limit.) The top plan's exports (payload.priority) go before all of these, each person
// still within their limit.
const EXPORTS_PER_PERSON = Math.max(1, parseInt(process.env.EXPORTS_PER_PERSON ?? '10', 10) || 10)
const SUBJECTS_PER_PERSON = Math.max(1, parseInt(process.env.SUBJECTS_PER_PERSON ?? '1', 10) || 1)
// Claims take turns (a transaction lock, held for the few ms a claim takes): two workers claiming at
// the same moment would otherwise both see "none running" and both take one, past a limit
const CLAIM_LOCK = 20261027

async function claim(p: Pool, mayLong: boolean, mayProxy: boolean): Promise<Job | undefined> {
  // Nothing waiting (most polls of an idle worker): no turn on the lock
  const [{ waiting }] = await db<{ waiting: number }[]>`
    SELECT count(*)::int AS waiting FROM (SELECT 1 FROM jobs WHERE status = 'queued' AND type = ANY(${p.types}) LIMIT 2) w`
  if (!waiting) return undefined
  // A busy worker lets idle ones go first (see SPREAD_MS)
  const ahead = busyHere++
  // Several waiting (a burst): the next idle slot starts its own, longer wait now, side by side,
  // rather than after this claim (6 exports at one machine: the last started 1.2 s late, not 0.4 s)
  if (waiting > 1 && SPREAD_MS) idle.shift()?.()
  try {
    if (ahead && SPREAD_MS) await new Promise(r => setTimeout(r, Math.min(ahead * SPREAD_MS, SPREAD_MAX_MS)))
    return await claimNow(p, mayLong, mayProxy)
  } finally {
    busyHere--
  }
}

async function claimNow(p: Pool, mayLong: boolean, mayProxy: boolean): Promise<Job | undefined> {
  return db.begin(async tx => {
    await tx`SELECT pg_advisory_xact_lock(${CLAIM_LOCK})`
    const reading = tx`(type = 'transcribe' AND COALESCE(payload->>'transcribe_full', '') = 'true')`
    // How many of this job's kind its person has running (counted once, from the few running jobs)
    const theirs = tx`COALESCE(running.n, 0)`
    const [job] = await tx<Job[]>`
      WITH running AS (
        SELECT type AS kind, payload->>'user_id' AS person, count(*) AS n FROM jobs
        WHERE status = 'processing' AND payload->>'user_id' IS NOT NULL GROUP BY 1, 2
      ), next AS (
        SELECT jobs.id FROM jobs LEFT JOIN running ON running.kind = jobs.type AND running.person = jobs.payload->>'user_id'
        WHERE status = 'queued' AND type = ANY(${p.types})
        ${p.clipCaptions && !p.readings ? tx`AND NOT ${reading}` : tx``}
        ${p.readings && !p.clipCaptions ? tx`AND (type <> 'transcribe' OR ${reading})` : tx``}
        ${isRailway ? tx`AND NOT (type = 'transcribe' AND payload @> '{"requires_ytdlp":true}'::jsonb)` : tx``}
        ${mayProxy ? tx`` : tx`AND type <> 'proxy'`}
        ${mayLong ? tx`` : tx`AND NOT (type = 'proxy' OR ${reading})`}
        AND NOT (type = 'render' AND ${theirs} >= ${EXPORTS_PER_PERSON})
        AND NOT (type = 'track_subjects' AND ${theirs} >= ${SUBJECTS_PER_PERSON})
        -- One whole-video reading per person at a time (across all workers): one person's videos
        -- never take every slot while someone else waits
        AND NOT (${reading} AND EXISTS (
          SELECT 1 FROM jobs p
          JOIN videos pv ON pv.id::text = p.payload->>'video_id'
          JOIN videos v ON v.id::text = jobs.payload->>'video_id'
          WHERE p.type = 'transcribe' AND p.status = 'processing' AND COALESCE(p.payload->>'transcribe_full', '') = 'true'
            AND pv.user_id = v.user_id))
        -- A whole-video reading (AI edit: many minutes for a long video) waits for the quick jobs that
        -- came after it — a clip's own captions (manual edit), renders, uploads — so they never queue
        -- behind it (COALESCE: a job without the field must sort as false, not as NULL, which goes last).
        -- Then the top plan's exports (payload.priority: "priority rendering"), then the person with
        -- fewest of that kind running, then the oldest.
        ORDER BY (type = 'proxy') ASC, ${reading} ASC, (type = 'render' AND payload @> '{"priority":true}'::jsonb) DESC,
          ${theirs} ASC, created_at ASC LIMIT 1
        FOR UPDATE OF jobs SKIP LOCKED
      ), claimed AS (
        UPDATE jobs j SET status = 'processing' FROM next WHERE j.id = next.id AND j.status = 'queued' RETURNING j.*
      ), beat AS (
        INSERT INTO job_heartbeats (job_id) SELECT id FROM claimed
        ON CONFLICT (job_id) DO UPDATE SET at = now()
      )
      SELECT * FROM claimed
    `
    return job
  })
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

