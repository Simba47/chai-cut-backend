import 'dotenv/config' // reload env

// Gemini is the only AI vendor: GEMINI_API_KEY does captions and everything else.
// SARVAM_API_KEY, GROQ_API_KEY and OPENAI_API_KEY are no longer read.
const REQUIRED_ENV = ['DATABASE_URL', 'R2_ENDPOINT', 'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY', 'GEMINI_API_KEY']
const missing = REQUIRED_ENV.filter(k => !process.env[k])
if (missing.length) {
  console.error(`[startup] Missing required env vars: ${missing.join(', ')}`)
  process.exit(1)
}
// Captions: Vertex AI when GOOGLE_CLOUD_PROJECT is set (no daily request cap), else the Gemini API key
console.log(process.env.GOOGLE_CLOUD_PROJECT
  ? `[startup] Captions via Vertex AI (project ${process.env.GOOGLE_CLOUD_PROJECT}, ${process.env.GOOGLE_CREDENTIALS_JSON ? 'service account key from GOOGLE_CREDENTIALS_JSON' : 'default credentials'})`
  : '[startup] Captions via the Gemini API key (100 requests/day on Tier 1). Set GOOGLE_CLOUD_PROJECT to use Vertex AI.')

// Every job shells out to ffmpeg. A worker without it would still claim jobs from the shared
// queue and fail each one (users see "Render failed"), so refuse to start instead.
import { execFileSync } from 'node:child_process'
try {
  execFileSync('ffmpeg', ['-version'], { stdio: 'ignore' })
} catch {
  console.error('[startup] ffmpeg is not installed or not on PATH. This worker would fail every render and caption job, so it will not start.')
  console.error('[startup] Install ffmpeg (and ideally point DATABASE_URL at a test database) before running the worker locally.')
  process.exit(1)
}

import db from './db.js'
import { registerHandler, startQueue } from './queue.js'
import { handleTranscribeJob } from './jobs/transcribe.js'
import { handleRenderJob } from './jobs/render.js'
import { handleAiEditJob } from './jobs/ai_edit.js'
import { handleProxyJob } from './jobs/proxy.js'

// Safe schema migrations — idempotent, run on every startup
async function runMigrations() {
  await db`ALTER TABLE caption_styles ADD COLUMN IF NOT EXISTS timing_offset_ms INTEGER DEFAULT 0`
  // The transcript comes from reading the whole video (not just a clip of it): Make my clips,
  // Best moments and Ask AI then use it as it is and never send the video for captions again
  await db`ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS whole_video BOOLEAN NOT NULL DEFAULT false`
  // Captions on/off per clip. Turning captions off used to delete the style, and the editor then
  // switched them back on for any clip whose video has a transcript. Existing styles count as on.
  await db`ALTER TABLE caption_styles ADD COLUMN IF NOT EXISTS enabled BOOLEAN NOT NULL DEFAULT true`
  await db`ALTER TABLE videos ADD COLUMN IF NOT EXISTS role TEXT CHECK (role IN ('project','asset')) NOT NULL DEFAULT 'project'`
  await db`ALTER TABLE videos ADD COLUMN IF NOT EXISTS title TEXT`
  // Why a video failed, in words the user can act on (e.g. a link that isn't shared publicly)
  await db`ALTER TABLE videos ADD COLUMN IF NOT EXISTS error TEXT`
  // Frames: five frame layouts, letterbox band settings per format, and per-slot photo / motion / audio mix
  await db`ALTER TABLE segments DROP CONSTRAINT IF EXISTS segments_layout_check`
  await db`ALTER TABLE segments ADD CONSTRAINT segments_layout_check CHECK (layout IN (
    'vertical','split','trio','spotlight','centered','horizontal',
    'frame_single','frame_video_photo','frame_dual','frame_dual_letterbox','frame_triple',
    'frame_title_caption','frame_big_small','frame_photo_story'))`
  await db`ALTER TABLE segments ADD COLUMN IF NOT EXISTS frame JSONB`
  await db`ALTER TABLE crop_boxes ADD COLUMN IF NOT EXISTS image_path TEXT`
  await db`ALTER TABLE crop_boxes ADD COLUMN IF NOT EXISTS image_motion TEXT`
  await db`ALTER TABLE crop_boxes ADD COLUMN IF NOT EXISTS volume REAL NOT NULL DEFAULT 1`
  await db`ALTER TABLE crop_boxes ADD COLUMN IF NOT EXISTS muted BOOLEAN NOT NULL DEFAULT false`
  // Timeline controls in the editor: hide (not shown, not exported), mute (silent), lock (can't be
  // moved, trimmed or deleted). Music also keeps its trims (where in the song it starts, where it stops).
  await db`ALTER TABLE segments ADD COLUMN IF NOT EXISTS hidden BOOLEAN NOT NULL DEFAULT false`
  await db`ALTER TABLE segments ADD COLUMN IF NOT EXISTS muted BOOLEAN NOT NULL DEFAULT false`
  await db`ALTER TABLE segments ADD COLUMN IF NOT EXISTS locked BOOLEAN NOT NULL DEFAULT false`
  await db`ALTER TABLE overlays ADD COLUMN IF NOT EXISTS hidden BOOLEAN NOT NULL DEFAULT false`
  await db`ALTER TABLE overlays ADD COLUMN IF NOT EXISTS muted BOOLEAN NOT NULL DEFAULT false`
  await db`ALTER TABLE overlays ADD COLUMN IF NOT EXISTS locked BOOLEAN NOT NULL DEFAULT false`
  await db`ALTER TABLE text_overlays ADD COLUMN IF NOT EXISTS hidden BOOLEAN NOT NULL DEFAULT false`
  await db`ALTER TABLE text_overlays ADD COLUMN IF NOT EXISTS locked BOOLEAN NOT NULL DEFAULT false`
  await db`ALTER TABLE audio_tracks ADD COLUMN IF NOT EXISTS muted BOOLEAN NOT NULL DEFAULT false`
  await db`ALTER TABLE audio_tracks ADD COLUMN IF NOT EXISTS locked BOOLEAN NOT NULL DEFAULT false`
  await db`ALTER TABLE audio_tracks ADD COLUMN IF NOT EXISTS offset_ms INTEGER`
  await db`ALTER TABLE audio_tracks ADD COLUMN IF NOT EXISTS end_ms INTEGER`
  // Music: the fade-in / fade-out buttons (0.5 s each, in the preview and the export)
  await db`ALTER TABLE audio_tracks ADD COLUMN IF NOT EXISTS fade_in BOOLEAN NOT NULL DEFAULT false`
  await db`ALTER TABLE audio_tracks ADD COLUMN IF NOT EXISTS fade_out BOOLEAN NOT NULL DEFAULT false`
  // The clip's own sound (top of the Music panel): its volume and mute, saved and exported
  await db`ALTER TABLE clips ADD COLUMN IF NOT EXISTS original_volume REAL NOT NULL DEFAULT 1`
  await db`ALTER TABLE clips ADD COLUMN IF NOT EXISTS original_muted BOOLEAN NOT NULL DEFAULT false`
  // Clip board: a clip marked as a favourite (the heart on its card)
  await db`ALTER TABLE clips ADD COLUMN IF NOT EXISTS favorite BOOLEAN NOT NULL DEFAULT false`
  // An added video (B-roll) hidden on its own; segments.hidden is then the main video's
  await db`ALTER TABLE crop_boxes ADD COLUMN IF NOT EXISTS hidden BOOLEAN NOT NULL DEFAULT false`
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
  // The original schema only allowed 'transcribe' and 'render' jobs, so an 'ai_edit' job could not be queued
  await db`ALTER TABLE jobs DROP CONSTRAINT IF EXISTS jobs_type_check`
  await db`ALTER TABLE jobs ADD CONSTRAINT jobs_type_check CHECK (type IN ('transcribe','render','ai_edit','proxy'))`
  // The editing copy of a video (jobs/proxy.ts): what the editor plays, quick to drag through
  await db`ALTER TABLE videos ADD COLUMN IF NOT EXISTS proxy_path TEXT`
  // "Make my clips": how far the job has got (0–100), and why the AI picked each clip
  await db`ALTER TABLE ai_edit_jobs ADD COLUMN IF NOT EXISTS progress INTEGER NOT NULL DEFAULT 0`
  // Make my clips asks before making fewer clips than asked for: the run stops in 'confirm' with
  // the moments it found, and a yes from the user queues it again to make exactly those
  await db`ALTER TABLE ai_edit_jobs DROP CONSTRAINT IF EXISTS ai_edit_jobs_status_check`
  await db`ALTER TABLE ai_edit_jobs ADD CONSTRAINT ai_edit_jobs_status_check CHECK (status IN ('queued', 'running', 'done', 'failed', 'confirm'))`
  await db`ALTER TABLE ai_edit_jobs ADD COLUMN IF NOT EXISTS found JSONB`
  await db`ALTER TABLE clips ADD COLUMN IF NOT EXISTS ai_score INTEGER`
  await db`ALTER TABLE clips ADD COLUMN IF NOT EXISTS ai_reason TEXT`
  // Animated caption presets (render.py _preset_events, editor drawPresetCaptions)
  await db`ALTER TABLE caption_styles DROP CONSTRAINT IF EXISTS caption_styles_animation_check`
  await db`ALTER TABLE caption_styles ADD CONSTRAINT caption_styles_animation_check
    CHECK (animation IN ('karaoke','fade','none','pop','highlight','bounce','word','hormozi','box','glow'))`
  await db`ALTER TABLE caption_styles ADD COLUMN IF NOT EXISTS highlight_color TEXT DEFAULT '#FFE700'`
  await db`ALTER TABLE caption_styles ADD COLUMN IF NOT EXISTS words_per_line INTEGER`
  await db`ALTER TABLE caption_styles ADD COLUMN IF NOT EXISTS uppercase BOOLEAN NOT NULL DEFAULT false`
  await db`ALTER TABLE caption_styles ADD COLUMN IF NOT EXISTS stroke_width INTEGER DEFAULT 4`
  // Words the AI marked as important ({ "<word start_ms>": true }), made once per clip at export
  await db`ALTER TABLE caption_styles ADD COLUMN IF NOT EXISTS emphasis JSONB`
  await db`ALTER TABLE caption_styles ADD COLUMN IF NOT EXISTS emoji BOOLEAN NOT NULL DEFAULT false`
  // Remove pauses and filler words: the switch per clip, and the source-time ranges it cut
  await db`ALTER TABLE clips ADD COLUMN IF NOT EXISTS remove_fillers BOOLEAN NOT NULL DEFAULT false`
  await db`ALTER TABLE clips ADD COLUMN IF NOT EXISTS cut_ranges JSONB`
  // Parts of the video the user removed from a clip in the editor ("delete this part"):
  // [[start, end], …] in ms of the source video (lib/trims.ts). The export cuts them out.
  await db`ALTER TABLE clips ADD COLUMN IF NOT EXISTS trim_ranges JSONB`
  // Where the clip started and ended when it was made, kept the first time its start / end is
  // dragged in the editor: Reset brings the clip back to it
  // Videos on top (B-roll) are layers in the editor but saved as one row of sections that never
  // overlap (what this worker renders). This is the editor's own note of the sections it cut to
  // do that, so it can show them whole again; nothing here reads it.
  await db`ALTER TABLE clips ADD COLUMN IF NOT EXISTS layers JSONB`
  await db`ALTER TABLE clips ADD COLUMN IF NOT EXISTS original_start_ms INTEGER`
  await db`ALTER TABLE clips ADD COLUMN IF NOT EXISTS original_end_ms INTEGER`
  // Width of a text box (share of the frame's width) dragged in the editor: the text wraps inside it
  await db`ALTER TABLE text_overlays ADD COLUMN IF NOT EXISTS w REAL`
  // Height of a text box (share of the frame's height): the text sits in its middle
  await db`ALTER TABLE text_overlays ADD COLUMN IF NOT EXISTS h REAL`
  // AI post text per clip: hook (also a text overlay), post caption, hashtags
  await db`ALTER TABLE clips ADD COLUMN IF NOT EXISTS hook_text TEXT`
  await db`ALTER TABLE clips ADD COLUMN IF NOT EXISTS post_caption TEXT`
  await db`ALTER TABLE clips ADD COLUMN IF NOT EXISTS hashtags TEXT[]`
  // Stock videos saved as the user's assets for auto B-roll ("pexels:<id>" / "pixabay:<id>"), reused when picked again
  await db`ALTER TABLE videos ADD COLUMN IF NOT EXISTS stock_ref TEXT`
  // What users do with AI suggestions (to improve clip picking later)
  await db`
    CREATE TABLE IF NOT EXISTS ai_suggestion_events (
      id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      user_id uuid REFERENCES users(id) ON DELETE CASCADE,
      video_id uuid REFERENCES videos(id) ON DELETE CASCADE,
      clip_id uuid,
      source text CHECK (source IN ('best_moments','clip_search','auto_clips')),
      suggestion jsonb,
      event text CHECK (event IN ('shown','previewed','used','exported','deleted')) NOT NULL,
      created_at timestamptz NOT NULL DEFAULT now()
    )
  `
  await db`CREATE INDEX IF NOT EXISTS ai_suggestion_events_video_created ON ai_suggestion_events (video_id, created_at)`
  console.log('[startup] migrations ok')
}

registerHandler('transcribe', handleTranscribeJob)
registerHandler('render', handleRenderJob)
registerHandler('ai_edit', handleAiEditJob)
registerHandler('proxy', handleProxyJob)

runMigrations()
  .then(() => startQueue())
  .catch(err => {
    console.error('[worker] Fatal error:', err)
    process.exit(1)
  })
