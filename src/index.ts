import 'dotenv/config' // reload env

const REQUIRED_ENV = ['DATABASE_URL', 'SARVAM_API_KEY', 'GROQ_API_KEY', 'R2_ENDPOINT', 'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY', 'GEMINI_API_KEY']
const missing = REQUIRED_ENV.filter(k => !process.env[k])
if (missing.length) {
  console.error(`[startup] Missing required env vars: ${missing.join(', ')}`)
  process.exit(1)
}

import db from './db.js'
import { registerHandler, startQueue } from './queue.js'
import { handleTranscribeJob } from './jobs/transcribe.js'
import { handleRenderJob } from './jobs/render.js'
import { handleAiEditJob } from './jobs/ai_edit.js'

// Safe schema migrations — idempotent, run on every startup
async function runMigrations() {
  await db`ALTER TABLE caption_styles ADD COLUMN IF NOT EXISTS timing_offset_ms INTEGER DEFAULT 0`
  await db`ALTER TABLE videos ADD COLUMN IF NOT EXISTS role TEXT CHECK (role IN ('project','asset')) NOT NULL DEFAULT 'project'`
  await db`
    CREATE TABLE IF NOT EXISTS ai_edit_jobs (
      id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      video_id uuid NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
      clip_count integer NOT NULL,
      status text CHECK (status IN ('queued', 'running', 'done', 'failed')) NOT NULL DEFAULT 'queued',
      error text,
      created_at timestamptz NOT NULL DEFAULT now()
    )
  `
  await db`ALTER TABLE clips ADD COLUMN IF NOT EXISTS ai_edit_job_id uuid REFERENCES ai_edit_jobs(id) ON DELETE SET NULL`
  console.log('[startup] migrations ok')
}

registerHandler('transcribe', handleTranscribeJob)
registerHandler('render', handleRenderJob)
registerHandler('ai_edit', handleAiEditJob)

runMigrations()
  .then(() => startQueue())
  .catch(err => {
    console.error('[worker] Fatal error:', err)
    process.exit(1)
  })
