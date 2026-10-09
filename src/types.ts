export type VideoStatus = 'uploaded' | 'transcribing' | 'ready' | 'failed'
export type ClipStatus = 'draft' | 'rendering' | 'done' | 'failed'
export type JobType = 'transcribe' | 'render' | 'ai_edit' | 'proxy'
export type JobStatus = 'queued' | 'processing' | 'done' | 'failed'
export type RenderQuality = '480p' | '720p' | '1080p' | '2160p'

export interface Job {
  id: string
  type: JobType
  payload: Record<string, unknown>
  status: JobStatus
  error: string | null
  created_at: string
  updated_at: string
}

export interface TranscribeJobPayload {
  video_id: string
  storage_path: string
  language_code?: string
  is_retranscribe?: boolean
  clip_id?: string
  clip_start_ms?: number
  clip_end_ms?: number
  // Re-render with fresh captions: this render job is queued once transcription finishes
  render_after?: RenderJobPayload
  // Upload/link: caption the whole video in the background once it's ready
  transcribe_full?: boolean
  // Link imports from Google Drive / Dropbox: downloaded directly (no yt-dlp)
  link_source?: 'gdrive' | 'dropbox'
  // Largest file the user's plan allows (link imports stop past this)
  max_bytes?: number
}

/** The editing copy of a video (jobs/proxy.ts) */
export interface ProxyJobPayload {
  video_id: string
}

export interface RenderJobPayload {
  clip_id: string
  video_storage_path: string
  quality?: RenderQuality
  watermark?: boolean
}

export interface AiEditJobPayload {
  ai_edit_job_id: string
  video_id: string
  clip_count: number
  /** Add stock B-roll (also needs AUTO_BROLL=on and PEXELS_API_KEY on the worker) */
  add_broll?: boolean
  // The Make my clips checkboxes; a missing one counts as on (jobs queued before they existed)
  /** Karaoke captions (off: no caption style is saved, so none show or export) */
  captions?: boolean
  /** The AI hook as an on-screen title over the first 3 seconds */
  title?: boolean
  /** The crop follows people (off: it holds still within each shot) */
  motion?: boolean
  /** Split and trio layouts (off: always vertical, on one person) */
  layouts?: boolean
  /** Captions in the speaker's own script ('native': Telugu, Hindi letters) or in English letters ('roman': Tenglish, Hinglish…). Missing = native where a font is bundled */
  caption_language?: 'native' | 'roman'
  /** The title, hook and post caption: English letters in the speaker's language ('roman', the default), English, or the speaker's own script */
  title_language?: 'roman' | 'english' | 'native'
  /** The user said yes to making fewer clips than asked for: the moments already found (ai_edit_jobs.found) are used */
  confirmed?: boolean
}

export const JOB_POLL_INTERVAL_MS = 2000
