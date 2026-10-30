import 'dotenv/config' // reload env

// Gemini is the only AI vendor. With GOOGLE_CLOUD_PROJECT set, captions and every other AI call
// (clip finding, hooks, B-roll, visuals) go to Vertex AI and no Gemini API key is needed; without
// it, GEMINI_API_KEY is used for all of them. SARVAM_API_KEY, GROQ_API_KEY and OPENAI_API_KEY are no longer read.
const REQUIRED_ENV = ['DATABASE_URL', 'R2_ENDPOINT', 'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY']
const missing = REQUIRED_ENV.filter(k => !process.env[k])
if (!process.env.GOOGLE_CLOUD_PROJECT && !process.env.GEMINI_API_KEY) missing.push('GOOGLE_CLOUD_PROJECT (Vertex AI) or GEMINI_API_KEY')
if (missing.length) {
  console.error(`[startup] Missing required env vars: ${missing.join(', ')}`)
  process.exit(1)
}
console.log(process.env.GOOGLE_CLOUD_PROJECT
  ? `[startup] AI via Vertex AI (project ${process.env.GOOGLE_CLOUD_PROJECT}, ${process.env.GOOGLE_CREDENTIALS_JSON ? 'service account key from GOOGLE_CREDENTIALS_JSON' : 'default credentials'})`
  : '[startup] AI via the Gemini API key (captions: 100 requests/day on Tier 1). Set GOOGLE_CLOUD_PROJECT to use Vertex AI.')

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
import { handleTrackSubjectsJob } from './jobs/track_subjects.js'
import { handleDetectSilenceJob } from './jobs/silence.js'

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
  // Frames: the frame layouts, letterbox band settings per format, and per-slot photo / motion / audio mix
  // (15 more frames added 2026-10: blur fill, gameplay split, picture in picture, … — frames.py)
  await db`ALTER TABLE segments DROP CONSTRAINT IF EXISTS segments_layout_check`
  await db`ALTER TABLE segments ADD CONSTRAINT segments_layout_check CHECK (layout IN (
    'vertical','split','trio','spotlight','centered','horizontal',
    'frame_single','frame_video_photo','frame_dual','frame_dual_letterbox','frame_triple',
    'frame_title_caption','frame_big_small','frame_photo_story',
    'frame_blur_fill','frame_gameplay','frame_pip','frame_screenshot','frame_comment','frame_letterbox_wide',
    'frame_two_crops','frame_side_by_side','frame_name_tags','frame_brand_card','frame_top_list',
    'frame_screen_face','frame_audiogram','frame_grid','frame_news','frame_trio_main'))`
  await db`ALTER TABLE segments ADD COLUMN IF NOT EXISTS frame JSONB`
  // Where a segment's video starts when a true INSERT pushed it forward (render.py, audio.py, frames.py).
  // Production got this column by hand; databases built from schema alone (QA) lacked it.
  await db`ALTER TABLE segments ADD COLUMN IF NOT EXISTS video_offset_ms INTEGER`
  // A part played faster or slower (the editor's Speed): its speed, smooth slow-mo, and whether
  // its sound's pitch follows the speed (else the voice stays natural)
  await db`ALTER TABLE segments ADD COLUMN IF NOT EXISTS speed REAL NOT NULL DEFAULT 1`
  await db`ALTER TABLE segments ADD COLUMN IF NOT EXISTS smooth BOOLEAN NOT NULL DEFAULT false`
  await db`ALTER TABLE segments ADD COLUMN IF NOT EXISTS pitch BOOLEAN NOT NULL DEFAULT false`
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
  await db`ALTER TABLE jobs ADD CONSTRAINT jobs_type_check CHECK (type IN ('transcribe','render','ai_edit','proxy','track_subjects','detect_silence'))`
  // One device per account: the current sign-in's id (the site's modules/auth/one-device.ts)
  await db`ALTER TABLE users ADD COLUMN IF NOT EXISTS session_id TEXT`
  // A running job's worker is alive (queue.ts heartbeat); a job whose heartbeat stops is taken back.
  // Its own table: an update of jobs sets its updated_at (the jobs_updated_at trigger), which the
  // site reads as when the job started
  await db`CREATE TABLE IF NOT EXISTS job_heartbeats (
    job_id uuid PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    at timestamptz NOT NULL DEFAULT now()
  )`
  // Subjects: people the user marked in a clip (editor Subjects tool), followed through it by
  // jobs/track_subjects.ts. path: [[source ms, x, y, w, h], …] — their face 4× a second,
  // fractions of the frame, only where they were seen.
  await db`
    CREATE TABLE IF NOT EXISTS clip_subjects (
      id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      clip_id uuid NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
      label text NOT NULL,
      color text NOT NULL,
      mark_ms integer NOT NULL,
      mark_x real NOT NULL,
      mark_y real NOT NULL,
      status text NOT NULL DEFAULT 'new' CHECK (status IN ('new', 'queued', 'running', 'done', 'failed')),
      error text,
      progress integer NOT NULL DEFAULT 0,
      scan_start_ms integer,
      scan_end_ms integer,
      path jsonb,
      created_at timestamptz NOT NULL DEFAULT now()
    )
  `
  await db`CREATE INDEX IF NOT EXISTS clip_subjects_clip ON clip_subjects (clip_id)`
  // The camera cuts found while tracking (source ms): the editor never carries a face across one,
  // and switches the view between people exactly on them
  await db`ALTER TABLE clip_subjects ADD COLUMN IF NOT EXISTS cuts jsonb`
  // When the subject is talking ([start, end] source ms, to the word): the view cuts to them on their first word
  await db`ALTER TABLE clip_subjects ADD COLUMN IF NOT EXISTS speaking jsonb`
  // A group subject: a camera shot (these people, in these places) marked with the format and views
  // the user gave it (shot: { layout, boxes }); spans: [start, end] source ms wherever the clip shows
  // that shot again (jobs/track_subjects.ts shotSpans) — the editor puts that format there
  await db`ALTER TABLE clip_subjects ADD COLUMN IF NOT EXISTS shot jsonb`
  await db`ALTER TABLE clip_subjects ADD COLUMN IF NOT EXISTS spans jsonb`
  // The clip's camera angles (jobs/track_subjects.ts findAngles), listed in the editor's Subjects
  // tool so every one gets its subject or group. angles: [{ people, at, faces, spans }, …] in source ms
  await db`
    CREATE TABLE IF NOT EXISTS clip_angles (
      clip_id uuid PRIMARY KEY REFERENCES clips(id) ON DELETE CASCADE,
      status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'done', 'failed')),
      error text,
      progress integer NOT NULL DEFAULT 0,
      scan_start_ms integer,
      scan_end_ms integer,
      angles jsonb,
      updated_at timestamptz NOT NULL DEFAULT now()
    )
  `
  // The editing copy of a video (jobs/proxy.ts): what the editor plays, quick to drag through
  await db`ALTER TABLE videos ADD COLUMN IF NOT EXISTS proxy_path TEXT`
  // "Make my clips": how far the job has got (0–100), and why the AI picked each clip
  await db`ALTER TABLE ai_edit_jobs ADD COLUMN IF NOT EXISTS progress INTEGER NOT NULL DEFAULT 0`
  // Make my clips asks before making fewer clips than asked for: the run stops in 'confirm' with
  // the moments it found, and a yes from the user queues it again to make exactly those
  await db`ALTER TABLE ai_edit_jobs DROP CONSTRAINT IF EXISTS ai_edit_jobs_status_check`
  await db`ALTER TABLE ai_edit_jobs ADD CONSTRAINT ai_edit_jobs_status_check CHECK (status IN ('queued', 'running', 'done', 'failed', 'confirm'))`
  // Transitions between sections (the editor's Transitions tool): the kinds render.py draws with
  // ffmpeg's xfade, besides the first three. ('fade' = a dissolve, 'wipe' = wipeleft.)
  await db`ALTER TABLE transitions DROP CONSTRAINT IF EXISTS transitions_type_check`
  await db`ALTER TABLE transitions ADD CONSTRAINT transitions_type_check CHECK (type IN ('cut', 'fade', 'wipe', 'fadeblack', 'fadewhite', 'slideleft', 'slideup', 'zoomin', 'circleopen', 'slideright', 'slidedown', 'smoothleft', 'hblur', 'radial', 'pixelize'))`
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
  // The caption style ("template": a preset in its own font and look, see src/python/caption_templates.json).
  // `animation` keeps the style's base preset, so a worker from before this draws that instead.
  await db`ALTER TABLE caption_styles ADD COLUMN IF NOT EXISTS template TEXT`
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
  // How far a text or a photo is turned (degrees, clockwise) with its rotate handle on the preview
  await db`ALTER TABLE text_overlays ADD COLUMN IF NOT EXISTS rotation REAL`
  await db`ALTER TABLE overlays ADD COLUMN IF NOT EXISTS rotation REAL`
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
  // A clip's silences (the editor's Cleanup tool: jobs/silence.ts): its silent stretches, source ms
  await db`
    CREATE TABLE IF NOT EXISTS clip_silences (
      clip_id uuid PRIMARY KEY REFERENCES clips(id) ON DELETE CASCADE,
      status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'done', 'failed')),
      error text,
      ranges jsonb,
      scan_start_ms integer,
      scan_end_ms integer,
      updated_at timestamptz NOT NULL DEFAULT now()
    )
  `
  // A clip's B-roll ideas (editor Media ▸ B-roll / Photos: the AI's searches for what is said in it,
  // frontend services/stock.ts), kept so opening the panels again — after a redeploy too — doesn't
  // ask Gemini again. Good while the clip's span and its video's transcript are the ones they came from.
  await db`
    CREATE TABLE IF NOT EXISTS clip_broll_ideas (
      clip_id uuid PRIMARY KEY REFERENCES clips(id) ON DELETE CASCADE,
      transcript_id uuid NOT NULL,
      start_ms integer NOT NULL,
      end_ms integer NOT NULL,
      ideas jsonb NOT NULL,
      created_at timestamptz NOT NULL DEFAULT now()
    )
  `
  // Brand kits (editor Brand kit tool): a user's saved looks — colours, logo, @handle, title style —
  // put on a clip with Apply. data: the kit's settings (frontend lib/shared/brandKit.ts). How many a
  // user can have depends on their plan (frontend server/services/brandKits.ts)
  await db`
    CREATE TABLE IF NOT EXISTS brand_kits (
      id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      name text NOT NULL,
      data jsonb NOT NULL DEFAULT '{}'::jsonb,
      created_at timestamptz NOT NULL DEFAULT now(),
      updated_at timestamptz NOT NULL DEFAULT now()
    )
  `
  await db`CREATE INDEX IF NOT EXISTS brand_kits_user ON brand_kits (user_id, created_at)`
  // How see-through a photo on the clip is, 0–1 (a brand kit's logo; null = solid)
  await db`ALTER TABLE overlays ADD COLUMN IF NOT EXISTS opacity REAL`
  // The brand kit on a clip (the editor's Brand kit switch): { kit_id, before } — what the brand
  // changed, so switching it off puts it back; null = off
  await db`ALTER TABLE clips ADD COLUMN IF NOT EXISTS brand JSONB`
  // A bar along the top or bottom edge filling up as the clip plays (a brand kit's progress bar):
  // { color, height (px on a 1080 × 1920 frame), position: 'top' | 'bottom' }; null = none
  await db`ALTER TABLE clips ADD COLUMN IF NOT EXISTS progress_bar JSONB`
  // Where the captions are centred left to right (0–1 of the frame's width, dragged in the editor's
  // preview like text); null = the middle
  await db`ALTER TABLE caption_styles ADD COLUMN IF NOT EXISTS position_x REAL`
  // A brand kit shared with other accounts (its owner's team): they can use it on their clips, not
  // change it; with the kit's brand lock on, their editor offers only its colours and fonts
  await db`
    CREATE TABLE IF NOT EXISTS brand_kit_shares (
      kit_id uuid NOT NULL REFERENCES brand_kits(id) ON DELETE CASCADE,
      user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      created_at timestamptz NOT NULL DEFAULT now(),
      PRIMARY KEY (kit_id, user_id)
    )
  `
  await db`CREATE INDEX IF NOT EXISTS brand_kit_shares_user ON brand_kit_shares (user_id)`
  // Brand kit teams (Phase 6): what each person a kit is shared with can do — 'use' it on their clips,
  // or also 'edit' it (frontend lib/shared/brandTeam.ts)
  await db`ALTER TABLE brand_kit_shares ADD COLUMN IF NOT EXISTS role text NOT NULL DEFAULT 'use'`
  // People invited to a kit by email (an account, or not yet), until they accept or decline in the app
  // (no email is sent: the owner can copy a link to it). email: lower case
  await db`
    CREATE TABLE IF NOT EXISTS brand_kit_invites (
      id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      kit_id uuid NOT NULL REFERENCES brand_kits(id) ON DELETE CASCADE,
      email text NOT NULL,
      role text NOT NULL DEFAULT 'use',
      invited_by uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      created_at timestamptz NOT NULL DEFAULT now(),
      UNIQUE (kit_id, email)
    )
  `
  await db`CREATE INDEX IF NOT EXISTS brand_kit_invites_email ON brand_kit_invites (email)`
  // The lookups the website makes on every editor open, autosave and dashboard visit (Postgres doesn't
  // index a link column by itself: without these each one read the whole table). Each on its own, so
  // one that can't be made is logged and the worker still starts
  const indexes: Array<[string, ReturnType<typeof db>]> = [
    ['caption_styles_clip', db`CREATE INDEX IF NOT EXISTS caption_styles_clip ON caption_styles (clip_id)`],
    ['text_overlays_clip', db`CREATE INDEX IF NOT EXISTS text_overlays_clip ON text_overlays (clip_id)`],
    ['overlays_clip', db`CREATE INDEX IF NOT EXISTS overlays_clip ON overlays (clip_id)`],
    ['audio_tracks_clip', db`CREATE INDEX IF NOT EXISTS audio_tracks_clip ON audio_tracks (clip_id)`],
    ['transitions_clip', db`CREATE INDEX IF NOT EXISTS transitions_clip ON transitions (clip_id)`],
    ['transcripts_video_created', db`CREATE INDEX IF NOT EXISTS transcripts_video_created ON transcripts (video_id, created_at)`],
    ['videos_user_created', db`CREATE INDEX IF NOT EXISTS videos_user_created ON videos (user_id, created_at)`],
    ['ai_edit_jobs_video', db`CREATE INDEX IF NOT EXISTS ai_edit_jobs_video ON ai_edit_jobs (video_id)`],
  ]
  for (const [name, q] of indexes) {
    try { await q } catch (e) { console.warn(`[startup] index ${name} not made:`, e instanceof Error ? e.message : e) }
  }
  console.log('[startup] migrations ok')
}

// Workers starting together (a deploy of several copies) run the migrations one after another: two
// at once collide (DROP then ADD CONSTRAINT: "already exists", and the second worker stopped; found
// in a two-worker test, 2026-10-27). A lock held on one connection for the whole run.
const MIGRATION_LOCK = 20261029
async function migrateInTurn() {
  const conn = await db.reserve()
  try {
    await conn`SELECT pg_advisory_lock(${MIGRATION_LOCK})`
    await runMigrations()
  } finally {
    await conn`SELECT pg_advisory_unlock(${MIGRATION_LOCK})`.catch(() => {})
    conn.release()
  }
}

registerHandler('transcribe', handleTranscribeJob)
registerHandler('render', handleRenderJob)
registerHandler('ai_edit', handleAiEditJob)
registerHandler('proxy', handleProxyJob)
registerHandler('track_subjects', handleTrackSubjectsJob)
registerHandler('detect_silence', handleDetectSilenceJob)

/**
 * Whether this worker may change the database and take jobs. A worker on a PC whose database isn't
 * on that PC (a local .env pointing at QA's) doesn't, unless LOCAL_WORKER=on: it took QA's jobs
 * before Railway's worker did — with code not deployed yet and the PC's own AI settings (2026-10-17:
 * a QA transcript ran on a PC, through its Vertex project, and never showed on Railway) — and ran
 * its migrations on QA. A worker on Railway (QA or live) always does.
 */
function mayTakeJobs(): boolean {
  if (process.env.RAILWAY_PROJECT_ID || process.env.RAILWAY_SERVICE_ID) return true
  if (process.env.LOCAL_WORKER === 'on') return true
  let host = ''
  try { host = new URL(process.env.DATABASE_URL ?? '').hostname } catch { /* no database set: nothing to take */ }
  return ['localhost', '127.0.0.1', '::1', '[::1]'].includes(host)
}

if (!mayTakeJobs()) {
  console.log('[worker] Not taking jobs: this PC\'s database is somewhere else (QA\'s?), so Railway\'s worker does them. '
    + 'Set LOCAL_WORKER=on in .env to run this worker on it anyway.')
} else {
  migrateInTurn()
    .then(() => startQueue())
    .catch(err => {
      console.error('[worker] Fatal error:', err)
      process.exit(1)
    })
}
