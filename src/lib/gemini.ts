/**
 * One way to ask Gemini, for every AI call outside speech-to-text (clip finding, hooks and titles,
 * B-roll ideas, caption emphasis, script conversion, visuals).
 *
 * With GOOGLE_CLOUD_PROJECT set the request goes to Vertex AI, billed to that Google Cloud
 * project; no Gemini API key is needed. Login: the service account key in GOOGLE_CREDENTIALS_JSON
 * (the key file's contents, as JSON or base64), else the Application Default Credentials file
 * (`gcloud auth application-default login` on a developer PC). Without GOOGLE_CLOUD_PROJECT the
 * request goes to the Gemini API with GEMINI_API_KEY, as before.
 *
 * Built for many users at once: a cap on requests in flight per process (AI_MAX_CONCURRENT) and
 * retries spread out at random (see "Many users at once" below).
 *
 * Only Node built-ins, so this file is copied as-is between the repos:
 * chai-cut-backend/src/lib/gemini.ts and chai-cut-frontend/src/lib/gemini.ts. Keep the two identical.
 */
import { createSign } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'

export type GeminiPart = { text: string } | { inlineData: { mimeType: string; data: string } }

export interface AskOptions {
  /** Default gemini-2.5-flash */
  model?: string
  /** System instruction */
  system?: string
  /** Answer as JSON (responseMimeType application/json) */
  json?: boolean
  /** JSON shape the answer must follow (Gemini responseSchema) */
  schema?: unknown
  temperature?: number
  /** 0 switches thinking off (cheaper and faster for simple tasks) */
  thinkingBudget?: number
  /** Default 120 s */
  timeoutMs?: number
}

export const GEMINI_MODEL = 'gemini-2.5-flash'

/** Where AI requests go: 'vertex', 'gemini-api', or null when neither is configured */
export function aiProvider(): 'vertex' | 'gemini-api' | null {
  if (process.env.GOOGLE_CLOUD_PROJECT) return 'vertex'
  if (process.env.GEMINI_API_KEY) return 'gemini-api'
  return null
}

/** An error from Gemini that carries the HTTP status (429 / 5xx are worth retrying) */
export class GeminiError extends Error {
  constructor(message: string, readonly status: number) { super(message) }
}

/** Sends one request and returns the answer's text (thinking parts left out) */
export async function askGemini(input: string | GeminiPart[], opts: AskOptions = {}): Promise<string> {
  const provider = aiProvider()
  if (!provider) throw new Error('AI is not configured: set GOOGLE_CLOUD_PROJECT (Vertex AI) or GEMINI_API_KEY')
  const model = opts.model ?? GEMINI_MODEL

  const generationConfig: Record<string, unknown> = {}
  if (opts.json || opts.schema) generationConfig.responseMimeType = 'application/json'
  if (opts.schema) generationConfig.responseSchema = opts.schema
  if (opts.temperature !== undefined) generationConfig.temperature = opts.temperature
  if (opts.thinkingBudget !== undefined) generationConfig.thinkingConfig = { thinkingBudget: opts.thinkingBudget }
  const body = JSON.stringify({
    contents: [{ role: 'user', parts: typeof input === 'string' ? [{ text: input }] : input }],
    ...(opts.system ? { systemInstruction: { parts: [{ text: opts.system }] } } : {}),
    ...(Object.keys(generationConfig).length ? { generationConfig } : {}),
  })

  let url: string
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (provider === 'vertex') {
    const project = process.env.GOOGLE_CLOUD_PROJECT!
    const location = process.env.GOOGLE_CLOUD_LOCATION || 'global'
    const host = location === 'global' ? 'aiplatform.googleapis.com' : `${location}-aiplatform.googleapis.com`
    url = `https://${host}/v1/projects/${project}/locations/${location}/publishers/google/models/${model}:generateContent`
    headers.Authorization = `Bearer ${await vertexToken()}`
    headers['x-goog-user-project'] = project
  } else {
    url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`
    headers['x-goog-api-key'] = process.env.GEMINI_API_KEY!
  }

  // Vertex answers 429 when shared capacity is briefly busy: wait and try again
  for (let attempt = 1; ; attempt++) {
    const res = await inTurn(() => fetch(url, { method: 'POST', headers, body, signal: AbortSignal.timeout(opts.timeoutMs ?? 120_000) }))
    if (res.ok) {
      const data = await res.json() as { candidates?: { content?: { parts?: { text?: string; thought?: boolean }[] } }[] }
      return (data.candidates?.[0]?.content?.parts ?? []).filter(p => !p.thought).map(p => p.text ?? '').join('')
    }
    const text = await res.text()
    if ((res.status === 429 || res.status >= 500) && attempt < TRIES) {
      await new Promise(r => setTimeout(r, retryDelayMs(attempt, retryAfterMs(res.headers.get('retry-after')))))
      continue
    }
    throw new GeminiError(`Gemini ${res.status} (${provider}): ${text.slice(0, 300)}`, res.status)
  }
}

// ── Many users at once ───────────────────────────────────────────────────────
// Every user's AI calls share one Google Cloud project's capacity. A burst from one process only
// earns "busy" (429) answers, and many workers retrying on the same schedule hit the same busy
// moment again. So each process sends at most AI_MAX_CONCURRENT requests at a time (the rest wait
// their turn), and retries wait a random part of a growing delay, or as long as Google asks.

/** Tries per request: about a minute of waiting in all before giving up */
const TRIES = 6
/** Requests one process has in flight at most (AI_MAX_CONCURRENT, default 8) */
const MAX_IN_FLIGHT = Math.max(1, Number(process.env.AI_MAX_CONCURRENT) || 8)
let inFlight = 0
const waiting: Array<() => void> = []

/** Runs `send` when fewer than MAX_IN_FLIGHT requests are out; a finished one hands its turn to the next in line */
async function inTurn<T>(send: () => Promise<T>): Promise<T> {
  if (inFlight < MAX_IN_FLIGHT) inFlight++
  else await new Promise<void>(r => waiting.push(r))
  try {
    return await send()
  } finally {
    const next = waiting.shift()
    if (next) next()
    else inFlight--
  }
}

/**
 * How long to wait before try `attempt + 1`: half of a delay that doubles from 2 s (up to 30 s)
 * plus a random part of the other half, so workers that were refused together come back apart;
 * never less than Google's own hint
 */
export function retryDelayMs(attempt: number, hintMs = 0): number {
  const cap = Math.min(30_000, 2000 * 2 ** (attempt - 1))
  return Math.max(hintMs, cap / 2 + Math.random() * (cap / 2))
}

/** A Retry-After header (seconds, or an HTTP date) in ms; 0 when there is none, capped at 60 s */
export function retryAfterMs(header: string | null): number {
  if (!header) return 0
  const sec = Number(header)
  const ms = Number.isFinite(sec) ? sec * 1000 : Date.parse(header) - Date.now()
  return Number.isFinite(ms) && ms > 0 ? Math.min(60_000, ms) : 0
}

// ── Vertex AI login ──────────────────────────────────────────────────────────

type Credentials =
  | { type: 'service_account'; client_email: string; private_key: string; token_uri?: string }
  | { type: 'authorized_user'; client_id: string; client_secret: string; refresh_token: string }

let cached: { token: string; expires: number } | null = null

async function loadCredentials(): Promise<Credentials> {
  const raw = process.env.GOOGLE_CREDENTIALS_JSON?.trim()
  if (raw) {
    try { return JSON.parse(raw.startsWith('{') ? raw : Buffer.from(raw, 'base64').toString('utf8')) }
    catch { throw new Error('GOOGLE_CREDENTIALS_JSON is set but is not a valid service account key (paste the whole JSON file)') }
  }
  const file = process.env.GOOGLE_APPLICATION_CREDENTIALS || join(homedir(), '.config', 'gcloud', 'application_default_credentials.json')
  try { return JSON.parse(await readFile(file, 'utf8')) }
  catch { throw new Error('No Google Cloud login: set GOOGLE_CREDENTIALS_JSON, or run `gcloud auth application-default login`') }
}

/** An access token for Vertex AI, kept until 5 minutes before it expires */
async function vertexToken(): Promise<string> {
  if (cached && Date.now() < cached.expires) return cached.token
  const creds = await loadCredentials()
  let form: URLSearchParams
  let tokenUrl = 'https://oauth2.googleapis.com/token'
  if (creds.type === 'service_account') {
    tokenUrl = creds.token_uri || tokenUrl
    const now = Math.floor(Date.now() / 1000)
    const b64 = (o: unknown) => Buffer.from(JSON.stringify(o)).toString('base64url')
    const unsigned = `${b64({ alg: 'RS256', typ: 'JWT' })}.${b64({
      iss: creds.client_email, scope: 'https://www.googleapis.com/auth/cloud-platform', aud: tokenUrl, iat: now, exp: now + 3600,
    })}`
    const signature = createSign('RSA-SHA256').update(unsigned).sign(creds.private_key, 'base64url')
    form = new URLSearchParams({ grant_type: 'urn:ietf:params:oauth:grant-type:jwt-bearer', assertion: `${unsigned}.${signature}` })
  } else if (creds.type === 'authorized_user') {
    form = new URLSearchParams({
      grant_type: 'refresh_token', client_id: creds.client_id, client_secret: creds.client_secret, refresh_token: creds.refresh_token,
    })
  } else {
    throw new Error(`Unsupported Google credentials type: ${(creds as { type?: string }).type}`)
  }
  const res = await fetch(tokenUrl, { method: 'POST', body: form, signal: AbortSignal.timeout(30_000) })
  const data = await res.json().catch(() => ({})) as { access_token?: string; expires_in?: number; error_description?: string; error?: string }
  if (!res.ok || !data.access_token) throw new Error(`Google Cloud login failed (${res.status}): ${data.error_description ?? data.error ?? 'no token'}`)
  cached = { token: data.access_token, expires: Date.now() + ((data.expires_in ?? 3600) - 300) * 1000 }
  return cached.token
}
