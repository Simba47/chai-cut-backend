/**
 * The reference transcript (architecture plan, 1 Nov 2026: Shortcut-Transcription-Architecture-Plan.pdf).
 *
 * A self-hosted model (the transcriber service, Indic-Transcribe on a serverless GPU) reads every
 * upload once, cheaply and without Google's rate limits. Gemini then reads that TEXT to pick the
 * best moments, and Gemini audio is needed only for the captions of the clips people make.
 *
 * REFERENCE_TRANSCRIBER switches it (the same pattern as CREDITS_MODE):
 *   off     today's behaviour, nothing new runs (the default)
 *   shadow  reference transcripts are made and compared with today's picks, users still get today's
 *   on      the new flow (not wired yet: shadow first)
 * TRANSCRIBER_URL and TRANSCRIBER_TOKEN point at the service (the token is a secret: set by a person).
 *
 * This file has no database or network imports so its rules are tested on their own.
 */

export type ReferenceMode = 'off' | 'shadow' | 'on'

export function referenceMode(value: string | undefined = process.env.REFERENCE_TRANSCRIBER): ReferenceMode {
  const v = (value ?? '').trim().toLowerCase()
  return v === 'shadow' || v === 'on' ? v : 'off'
}

export type ReferenceWord = { word: string; start_ms: number; end_ms: number }

export interface TranscriberReply {
  words: ReferenceWord[]
  model: string
  language: string | null
  audioSeconds: number | null
  /** GPU time the service spent, for the cost log */
  gpuSeconds: number | null
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

/** The service's answer, checked: words with text and sane times, in time order */
export function parseTranscriberReply(body: unknown): TranscriberReply {
  const b = (body ?? {}) as Record<string, unknown>
  if (!Array.isArray(b.words)) throw new Error('The transcriber answered with no words list')
  const words: ReferenceWord[] = []
  for (const w of b.words as Array<Record<string, unknown>>) {
    const word = typeof w?.word === 'string' ? w.word.trim() : ''
    const start = num(w?.start_ms), end = num(w?.end_ms)
    if (!word || start === null || end === null || end < start || start < 0) continue
    words.push({ word, start_ms: Math.round(start), end_ms: Math.round(end) })
  }
  words.sort((a, b) => a.start_ms - b.start_ms)
  return {
    words,
    model: typeof b.model === 'string' ? b.model : 'unknown',
    language: typeof b.language === 'string' ? b.language : null,
    audioSeconds: num(b.audio_seconds),
    gpuSeconds: num(b.gpu_seconds),
  }
}

type Span = { start_ms: number; end_ms: number }

/** The same moment: they share more than half of the shorter one (as in the 1 Nov comparison) */
function sameMoment(a: Span, b: Span): boolean {
  const shared = Math.min(a.end_ms, b.end_ms) - Math.max(a.start_ms, b.start_ms)
  return shared > 0.5 * Math.min(a.end_ms - a.start_ms, b.end_ms - b.start_ms)
}

/**
 * Shadow mode's measure: how many of today's picks (baseline) the reference transcript also found,
 * and how many of its picks today's list contains.
 */
export function momentAgreement(baseline: Span[], reference: Span[]) {
  const found = baseline.filter(b => reference.some(r => sameMoment(b, r))).length
  const inBase = reference.filter(r => baseline.some(b => sameMoment(b, r))).length
  return {
    baselineFound: baseline.length ? found / baseline.length : 0,
    referenceInBaseline: reference.length ? inBase / reference.length : 0,
    baselineCount: baseline.length,
    referenceCount: reference.length,
  }
}

// ── The service: submit, then collect ─────────────────────────────────────────
//
// The worker never waits for a transcript inside a queue job (that would hold one of its few job
// slots for minutes and slow exports). It submits the audio and gets a ticket at once; a light
// background check collects finished transcripts by ticket.

export type ResultCheck = { status: 'pending' } | { status: 'done'; reply: TranscriberReply } | { status: 'failed'; error: string }

/** The service's answer to "is ticket X done?": 202 = still working, 200 = the transcript */
export function parseResultCheck(httpStatus: number, body: unknown): ResultCheck {
  if (httpStatus === 202) return { status: 'pending' }
  if (httpStatus === 200) {
    try {
      return { status: 'done', reply: parseTranscriberReply(body) }
    } catch (e) {
      return { status: 'failed', error: e instanceof Error ? e.message : String(e) }
    }
  }
  const error = typeof (body as { error?: unknown } | null)?.error === 'string' ? (body as { error: string }).error : `the transcriber answered ${httpStatus}`
  return { status: 'failed', error }
}

function serviceUrl(path: string): string {
  const base = process.env.TRANSCRIBER_URL
  if (!base) throw new Error('TRANSCRIBER_URL is not set')
  return base.replace(/\/+$/, '') + path
}
const authHeaders = (): Record<string, string> =>
  process.env.TRANSCRIBER_TOKEN ? { Authorization: `Bearer ${process.env.TRANSCRIBER_TOKEN}` } : {}

/** Hand an audio file (a link the service can download) to the transcriber; returns its ticket */
export async function submitReferenceTranscript(audioUrl: string, language: string | null): Promise<string> {
  const res = await fetch(serviceUrl('/submit'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ audio_url: audioUrl, language }),
    signal: AbortSignal.timeout(30_000),
  })
  if (!res.ok) throw new Error(`The transcriber refused the audio (${res.status}): ${(await res.text().catch(() => '')).slice(0, 200)}`)
  const body = (await res.json()) as { call_id?: unknown }
  if (typeof body.call_id !== 'string' || !body.call_id) throw new Error('The transcriber gave no ticket')
  return body.call_id
}

/** Is the transcript for this ticket ready? */
export async function checkReferenceTranscript(callId: string): Promise<ResultCheck> {
  const res = await fetch(serviceUrl(`/result?call_id=${encodeURIComponent(callId)}`), {
    headers: authHeaders(),
    signal: AbortSignal.timeout(60_000),
  })
  return parseResultCheck(res.status, await res.json().catch(() => null))
}
