/**
 * Checks that captions can run on Vertex AI: the service account logs in and the transcription
 * model answers. Sends 2 seconds of a test tone (costs a fraction of a paisa); touches no database.
 *
 *   GOOGLE_CLOUD_PROJECT=<project-id> GOOGLE_CREDENTIALS_JSON='<key json>' npx tsx src/scripts/check-vertex.ts
 * (or put both in .env and run: npx tsx src/scripts/check-vertex.ts)
 */
import 'dotenv/config'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { GoogleAuth } from 'google-auth-library'

const project = process.env.GOOGLE_CLOUD_PROJECT
const location = process.env.GOOGLE_CLOUD_LOCATION || 'global'
if (!project) { console.error('✗ GOOGLE_CLOUD_PROJECT is not set'); process.exit(1) }

const raw = process.env.GOOGLE_CREDENTIALS_JSON?.trim()
let credentials: Record<string, unknown> | undefined
if (raw) {
  try { credentials = JSON.parse(raw.startsWith('{') ? raw : Buffer.from(raw, 'base64').toString('utf8')) }
  catch { console.error('✗ GOOGLE_CREDENTIALS_JSON is not valid JSON (paste the whole key file)'); process.exit(1) }
  console.log(`• Service account: ${credentials!.client_email}`)
} else {
  console.log('• No GOOGLE_CREDENTIALS_JSON: using Application Default Credentials')
}

const auth = new GoogleAuth({ scopes: ['https://www.googleapis.com/auth/cloud-platform'], ...(credentials ? { credentials } : {}) })
let token: string | null | undefined
try {
  token = await auth.getAccessToken()
  console.log('✓ Logged in to Google Cloud')
} catch (e) {
  console.error('✗ Could not log in:', e instanceof Error ? e.message : e)
  process.exit(1)
}

const wav = join(tmpdir(), `vertex-check-${Date.now()}.wav`)
await promisify(execFile)('ffmpeg', ['-v', 'error', '-f', 'lavfi', '-i', 'sine=f=440:d=2', '-ar', '16000', '-ac', '1', '-acodec', 'pcm_s16le', '-y', wav])
const audio = (await readFile(wav)).toString('base64')
await rm(wav, { force: true })

const host = location === 'global' ? 'aiplatform.googleapis.com' : `${location}-aiplatform.googleapis.com`
const model = 'gemini-3.5-transcribe-preview'
const res = await fetch(`https://${host}/v1/projects/${project}/locations/${location}/publishers/google/models/${model}:generateContent`, {
  method: 'POST',
  headers: { Authorization: `Bearer ${token}`, 'x-goog-user-project': project, 'Content-Type': 'application/json' },
  body: JSON.stringify({
    contents: [{ role: 'user', parts: [{ inlineData: { mimeType: 'audio/wav', data: audio } }] }],
    generationConfig: { audioTranscriptionConfig: { wordTimestamp: true, diarization: true } },
  }),
})
const text = await res.text()
if (res.ok) {
  console.log(`✓ ${model} answered on Vertex AI (${location}). Captions can be switched to Vertex.`)
  process.exit(0)
}
console.error(`✗ Vertex AI answered ${res.status}: ${text.slice(0, 400)}`)
if (res.status === 403) console.error('  → Enable the Vertex AI API on the project and give the service account the "Vertex AI User" role.')
if (res.status === 404) console.error('  → The model or location was not found: keep GOOGLE_CLOUD_LOCATION unset (global).')
process.exit(1)
