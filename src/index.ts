import 'dotenv/config'
import { registerHandler, startQueue } from './queue.js'
import { handleTranscribeJob } from './jobs/transcribe.js'
import { handleRenderJob } from './jobs/render.js'

registerHandler('transcribe', handleTranscribeJob)
registerHandler('render', handleRenderJob)

startQueue().catch(err => {
  console.error('[worker] Fatal error:', err)
  process.exit(1)
})
