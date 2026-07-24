export type VideoStatus = 'uploaded' | 'transcribing' | 'ready' | 'failed'
export type ClipStatus = 'draft' | 'rendering' | 'done' | 'failed'
export type JobType = 'transcribe' | 'render'
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
}

export interface RenderJobPayload {
  clip_id: string
  video_storage_path: string
  quality?: RenderQuality
}

export const JOB_POLL_INTERVAL_MS = 2000
