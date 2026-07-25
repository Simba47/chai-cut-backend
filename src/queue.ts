import db from './db.js'
import type { Job, JobType } from './types.js'
import { JOB_POLL_INTERVAL_MS } from './types.js'

type JobHandler = (job: Job) => Promise<void>
const handlers = new Map<JobType, JobHandler>()

export function registerHandler(type: JobType, handler: JobHandler) {
  handlers.set(type, handler)
}

export async function startQueue() {
  const stuck = await db`UPDATE jobs SET status = 'queued' WHERE status = 'processing' RETURNING id`
  if (stuck.length) console.log(`[queue] Reset ${stuck.length} stuck processing job(s) to queued`)

  console.log('[queue] Worker started, polling for jobs…')
  while (true) {
    try { await tick() } catch (err) { console.error('[queue] Unexpected error in tick:', err) }
    await sleep(JOB_POLL_INTERVAL_MS)
  }
}

async function tick() {
  const [job] = await db<Job[]>`
    SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1
  `
  if (!job) return

  const claimed = await db`
    UPDATE jobs SET status = 'processing' WHERE id = ${job.id} AND status = 'queued' RETURNING id
  `
  if (!claimed.length) return

  console.log(`[queue] Processing job ${job.id} (${job.type})`)
  const handler = handlers.get(job.type as JobType)
  if (!handler) { await fail(job.id, `No handler for type: ${job.type}`); return }

  try {
    await handler(job)
    await db`UPDATE jobs SET status = 'done' WHERE id = ${job.id}`
    console.log(`[queue] Job ${job.id} done`)
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    console.error(`[queue] Job ${job.id} failed:`, msg)
    await fail(job.id, msg)
  }
}

async function fail(jobId: string, error: string) {
  await db`UPDATE jobs SET status = 'failed', error = ${error} WHERE id = ${jobId}`
}

function sleep(ms: number) { return new Promise(resolve => setTimeout(resolve, ms)) }
