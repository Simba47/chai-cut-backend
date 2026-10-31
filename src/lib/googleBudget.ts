/**
 * One Google speech-to-text budget shared by every worker (C3). Google lets the project send so many
 * audio tokens a minute (Gemini 3.5 Transcribe: ~32 tokens per second of audio; ~10K a minute on our
 * Vertex quota, 100K on Tier 2). Each worker used to find the limit on its own, by being refused
 * (429) and backing off, so an AI edit's whole-video reading and an editor's "captions on" fought
 * for it at random — on 2026-10-29 a reading met 429s on almost every chunk.
 *
 * Now each request waits its turn in one line, in Postgres, and goes when the budget has room:
 *   rank 0  an editor's captions, top plan     (someone is waiting in the editor)
 *   rank 1  an editor's captions
 *   rank 2  an AI edit's reading, top plan
 *   rank 3  an AI edit's reading
 * then first come, first served. The budget refills at GOOGLE_STT_TPM tokens a minute (a minute's
 * worth at most). Not set: no line, no budget (each worker on its own, as before). The website reads
 * the line to tell people where they are in it (google_waiting).
 */
import { AsyncLocalStorage } from 'node:async_hooks'
import { randomUUID } from 'node:crypto'
import db from '../db.js'

/** Audio tokens a minute the whole project may send (0: no shared budget) */
const TPM = Math.max(0, Number(process.env.GOOGLE_STT_TPM) || 0)
/** Gemini counts 32 tokens per second of audio */
const TOKENS_PER_SEC = 32
/** A waiter not seen for this long has gone (its worker stopped): the line skips it */
const GONE_AFTER_S = 30
const LANE = 'stt'

export interface SttTurn {
  /** Place in line (see above) */
  rank: number
  /** The job this request is for (the website looks up its place in line by it) */
  jobId?: string
}
/** Who the speech-to-text requests made inside it are for (handleTranscribeJob sets it) */
export const sttTurn = new AsyncLocalStorage<SttTurn>()

/** The place in line for a transcribe job: editors before readings, the top plan first within each */
export function sttRank(reading: boolean, topPlan: boolean): number {
  return (reading ? 2 : 0) + (topPlan ? 0 : 1)
}

let ready: Promise<void> | null = null
/** The two tables, made once (the first worker to start; the others find them) */
function tables(): Promise<void> {
  ready ??= db.begin(async tx => {
    await tx`SELECT pg_advisory_xact_lock(hashtext('google_budget_tables'))`
    await tx`CREATE TABLE IF NOT EXISTS google_budget (lane text PRIMARY KEY, tokens double precision NOT NULL, at timestamptz NOT NULL DEFAULT clock_timestamp())`
    await tx`CREATE TABLE IF NOT EXISTS google_waiting (
      id uuid PRIMARY KEY, lane text NOT NULL, rank smallint NOT NULL, job_id uuid,
      since timestamptz NOT NULL DEFAULT clock_timestamp(), seen timestamptz NOT NULL DEFAULT clock_timestamp())`
    await tx`CREATE INDEX IF NOT EXISTS google_waiting_line ON google_waiting (lane, rank, since)`
  }).then(() => undefined, e => { ready = null; throw e })
  return ready
}

/** Seconds of audio in a 16 kHz mono 16-bit WAV (what every chunk is) */
export function wavSeconds(buf: Buffer): number {
  return Math.max(0, buf.length - 44) / 32_000
}

/**
 * Waits for this request's turn and its share of the budget (`seconds` of audio), then returns.
 * At once when no budget is set (GOOGLE_STT_TPM).
 */
export async function waitForStt(seconds: number, signal?: AbortSignal): Promise<void> {
  if (!TPM) return
  await tables()
  const turn = sttTurn.getStore() ?? { rank: 1 }
  // A minute's worth at most: a bigger request (a 5-minute chunk at 9,600 tokens on a 10K budget)
  // waits for a full budget rather than forever
  const need = Math.min(TPM, Math.ceil(seconds * TOKENS_PER_SEC))
  const perSec = TPM / 60
  const me = randomUUID()
  const [{ since }] = await db<{ since: Date }[]>`
    INSERT INTO google_waiting (id, lane, rank, job_id) VALUES (${me}, ${LANE}, ${turn.rank}, ${turn.jobId ?? null}) RETURNING since`
  try {
    for (;;) {
      signal?.throwIfAborted()
      const waitMs = await db.begin(async tx => {
        await tx`INSERT INTO google_budget (lane, tokens) VALUES (${LANE}, ${TPM}) ON CONFLICT (lane) DO NOTHING`
        const [b] = await tx<{ tokens: number; secs: number }[]>`
          SELECT tokens, extract(epoch FROM clock_timestamp() - at)::float8 AS secs FROM google_budget WHERE lane = ${LANE} FOR UPDATE`
        // Someone ahead in line (a better rank, or the same rank sooner) goes first
        const [ahead] = await tx`
          SELECT 1 FROM google_waiting
          WHERE lane = ${LANE} AND id <> ${me} AND seen > clock_timestamp() - make_interval(secs => ${GONE_AFTER_S})
            AND (rank < ${turn.rank} OR (rank = ${turn.rank} AND (since, id) < (${since}, ${me})))
          LIMIT 1`
        const have = Math.min(TPM, b.tokens + b.secs * perSec)
        if (!ahead && have >= need) {
          await tx`UPDATE google_budget SET tokens = ${have - need}, at = clock_timestamp() WHERE lane = ${LANE}`
          return 0
        }
        await tx`UPDATE google_waiting SET seen = clock_timestamp() WHERE id = ${me}`
        // First in line: until the budget has refilled enough (looking again at least every 5 s);
        // behind someone: soon, to follow them
        return ahead ? 500 : Math.min(5000, Math.max(200, ((need - have) / perSec) * 1000))
      })
      if (!waitMs) return
      await new Promise(r => setTimeout(r, waitMs + Math.random() * 100))
    }
  } finally {
    await db`DELETE FROM google_waiting WHERE id = ${me}`.catch(() => {})
  }
}
