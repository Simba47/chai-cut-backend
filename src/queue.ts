import { supabase } from './supabase.js'
import type { Job, JobType } from './types.js'
import { JOB_POLL_INTERVAL_MS } from './types.js'

type JobHandler = (job: Job) => Promise<void>

const handlers = new Map<JobType, JobHandler>()

export function registerHandler(type: JobType, handler: JobHandler) {
  handlers.set(type, handler)
}

export async function startQueue() {
  // Reset any jobs left in `processing` from a previous crashed run
  const { count } = await supabase
    .from('jobs')
    .update({ status: 'queued' }, { count: 'exact' })
    .eq('status', 'processing')
  if (count) console.log(`[queue] Reset ${count} stuck processing job(s) to queued`)

  console.log('[queue] Worker started, polling for jobs…')

  while (true) {
    try {
      await tick()
    } catch (err) {
      console.error('[queue] Unexpected error in tick:', err)
    }
    await sleep(JOB_POLL_INTERVAL_MS)
  }
}

async function tick() {
  // Claim one queued job atomically using update + returning pattern
  const { data: jobs } = await supabase
    .from('jobs')
    .select('*')
    .eq('status', 'queued')
    .order('created_at', { ascending: true })
    .limit(1)

  if (!jobs?.length) return

  const job = jobs[0] as Job

  // Mark as processing
  const { error: claimError } = await supabase
    .from('jobs')
    .update({ status: 'processing' })
    .eq('id', job.id)
    .eq('status', 'queued')  // guard against double-claim

  if (claimError) {
    console.warn('[queue] Failed to claim job', job.id)
    return
  }

  console.log(`[queue] Processing job ${job.id} (${job.type})`)

  const handler = handlers.get(job.type as JobType)
  if (!handler) {
    await fail(job.id, `No handler registered for type: ${job.type}`)
    return
  }

  try {
    await handler(job)
    await supabase.from('jobs').update({ status: 'done' }).eq('id', job.id)
    console.log(`[queue] Job ${job.id} done`)
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    console.error(`[queue] Job ${job.id} failed:`, msg)
    await fail(job.id, msg)
  }
}

async function fail(jobId: string, error: string) {
  await supabase.from('jobs').update({ status: 'failed', error }).eq('id', jobId)
}

function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms))
}
