import 'dotenv/config' // reload env

const REQUIRED_ENV = ['DATABASE_URL', 'SARVAM_API_KEY', 'GROQ_API_KEY', 'R2_ENDPOINT', 'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY']
const missing = REQUIRED_ENV.filter(k => !process.env[k])
if (missing.length) {
  console.error(`[startup] Missing required env vars: ${missing.join(', ')}`)
  process.exit(1)
}

import db from './db.js'
import { registerHandler, startQueue } from './queue.js'
import { handleTranscribeJob } from './jobs/transcribe.js'
import { handleRenderJob } from './jobs/render.js'

// Safe schema migrations — idempotent, run on every startup
async function runMigrations() {
  await db`ALTER TABLE caption_styles ADD COLUMN IF NOT EXISTS timing_offset_ms INTEGER DEFAULT 0`
  console.log('[startup] migrations ok')
}

registerHandler('transcribe', handleTranscribeJob)
registerHandler('render', handleRenderJob)

runMigrations()
  .then(() => startQueue())
  .catch(err => {
    console.error('[worker] Fatal error:', err)
    process.exit(1)
  })
