import db from '../db.js'

/**
 * Brand kits on the clips Make my clips makes (user, 2026-10-17). A kit is a user's saved look
 * (the frontend's lib/shared/brandKit.ts: the editor's Brand kit tool, the brand_kits table). The
 * job's payload says which: 'none', a kit's id, or 'default' — the account's default kit, if it has
 * one. What it puts on each clip mirrors what the frontend puts on a clip made by hand
 * (services/brandKits.ts brandNewClips): its logo where and when the kit says, its caption style
 * when its captions are on, and here also its title style on the hook. Keep the two in step.
 */

export interface ClipBrand {
  title: { font: string | null; color: string; size: number }
  /** The caption style fields (null: the kit's captions are off — the job's own style is kept) */
  captions: {
    template: string | null; animation: string; font: string | null; size: number; color: string
    highlight_color: string; uppercase: boolean; words_per_line: number | null
  } | null
  /** How the AI writes the clip's text (lib/clipText.ts BrandVoice); null: none set */
  voice: { tone: string; avoid: string; hashtags: string[]; cta: string } | null
  /** A bar filling up as the clip plays (the clip's progress_bar); null: none */
  progress: ProgressBar | null
  /** Its logo on the frame (shares of it) and when it shows; null: no logo */
  logo: { path: string; x: number; y: number; w: number; h: number; opacity: number; show: 'all' | 'start' | 'end'; seconds: number } | null
}

/** A clip's progress bar (clips.progress_bar): its colour, thickness in px on a 1080 × 1920 frame, and edge */
export interface ProgressBar { color: string; height: number; position: 'top' | 'bottom' }

const HEX = /^#[0-9a-f]{6}$/i
const ANIMATIONS = ['karaoke', 'fade', 'none', 'pop', 'highlight', 'bounce', 'word', 'hormozi', 'box', 'glow']
const num = (v: unknown, lo: number, hi: number, fallback: number) => (typeof v === 'number' && isFinite(v) ? Math.min(hi, Math.max(lo, v)) : fallback)
const color = (v: unknown, fallback: string) => (typeof v === 'string' && HEX.test(v) ? v.toLowerCase() : fallback)
const r4 = (v: number) => Math.round(v * 10000) / 10000
/** The logo's margin from the frame's edges in its corner (share of the frame's width): the frontend's LOGO_MARGIN */
const LOGO_MARGIN = 0.04

/** A kit's saved settings (as the brand_kits row keeps them) as what goes on a clip; anything missing or wrong takes the default */
export function clipBrand(raw: unknown, userId: string): ClipBrand {
  const d = (raw && typeof raw === 'object' ? raw : {}) as Record<string, any>
  const t = (d.title && typeof d.title === 'object' ? d.title : {}) as Record<string, unknown>
  const c = (d.captions && typeof d.captions === 'object' ? d.captions : {}) as Record<string, unknown>
  const l = (d.logo && typeof d.logo === 'object' ? d.logo : {}) as Record<string, unknown>
  const path = typeof l.path === 'string' && l.path.startsWith(`brand/${userId}/`) && !l.path.includes('..') ? l.path : null
  let logo: ClipBrand['logo'] = null
  if (path) {
    // Where the frontend's logoBox puts it on the 1080 × 1920 frame
    const w = num(l.size, 0.06, 0.5, 0.18)
    const h = Math.min(0.9, (w * 1080 * num(l.ratio, 0.05, 20, 1)) / 1920)
    const corner = ['tl', 'tr', 'bl', 'br'].includes(l.corner as string) ? (l.corner as string) : 'tr'
    const mx = LOGO_MARGIN, my = (LOGO_MARGIN * 1080) / 1920
    const placed = typeof l.x === 'number' && typeof l.y === 'number' && isFinite(l.x) && isFinite(l.y)
    const x = placed ? Math.min(1 - w, Math.max(0, l.x as number)) : corner.endsWith('l') ? mx : 1 - mx - w
    const y = placed ? Math.min(1 - h, Math.max(0, l.y as number)) : corner.startsWith('t') ? my : 1 - my - h
    logo = {
      path, x: r4(x), y: r4(y), w: r4(w), h: r4(h), opacity: num(l.opacity, 0.1, 1, 0.9),
      show: (['all', 'start', 'end'] as const).find(s => s === l.show) ?? 'all', seconds: Math.round(num(l.seconds, 1, 15, 3)),
    }
  }
  const p = (d.progress && typeof d.progress === 'object' ? d.progress : {}) as Record<string, unknown>
  const v = (d.voice && typeof d.voice === 'object' ? d.voice : {}) as Record<string, unknown>
  const txt = (x: unknown, max: number) => (typeof x === 'string' ? x.slice(0, max) : '')
  const voice = {
    tone: txt(v.tone, 200), avoid: txt(v.avoid, 200), cta: txt(v.cta, 120),
    hashtags: (Array.isArray(v.hashtags) ? v.hashtags : []).map(h => txt(h, 40).replace(/^#+/, '').replace(/\s+/g, '')).filter(Boolean).slice(0, 5),
  }
  return {
    voice: voice.tone || voice.avoid || voice.cta || voice.hashtags.length ? voice : null,
    title: { font: typeof t.font === 'string' && t.font.length <= 60 ? t.font : null, color: color(t.color, '#ffffff'), size: Math.round(num(t.size, 24, 160, 64)) },
    captions: c.on === true ? {
      template: typeof c.template === 'string' && c.template.length <= 60 ? c.template : null,
      animation: ANIMATIONS.includes(c.animation as string) ? (c.animation as string) : 'karaoke',
      font: typeof c.font === 'string' && c.font.length <= 60 ? c.font : null,
      size: Math.round(num(c.size, 20, 100, 52)),
      color: color(c.color, '#ffffff'),
      highlight_color: color(c.highlight_color, '#ffe700'),
      uppercase: c.uppercase === true,
      words_per_line: typeof c.words_per_line === 'number' ? Math.round(num(c.words_per_line, 1, 10, 3)) : null,
    } : null,
    progress: p.on === true ? cleanProgressBar(p) : null,
    logo,
  }
}

/** A saved progress bar (clips.progress_bar, or a kit's) made safe; null when it is not one */
export function cleanProgressBar(raw: unknown): ProgressBar | null {
  if (!raw || typeof raw !== 'object') return null
  const p = raw as Record<string, unknown>
  return { color: color(p.color, '#ffffff'), height: Math.round(num(p.height, 4, 40, 12)), position: p.position === 'top' ? 'top' : 'bottom' }
}

/**
 * The kit a job's clips get: 'none' → none; a kit's id → that kit (the user's own, or one shared with
 * them: brand_kit_shares); else the user's default kit, if any. A shared kit's logo is its owner's.
 */
export async function loadClipBrand(userId: string, choice: string | undefined): Promise<ClipBrand | null> {
  if (choice === 'none') return null
  try {
    const [row] = choice && /^[0-9a-f-]{36}$/i.test(choice)
      ? await db<{ data: unknown; user_id: string }[]>`
          SELECT bk.data, bk.user_id FROM brand_kits bk WHERE bk.id = ${choice}
            AND (bk.user_id = ${userId} OR EXISTS (SELECT 1 FROM brand_kit_shares s WHERE s.kit_id = bk.id AND s.user_id = ${userId}))`
      : await db<{ data: unknown; user_id: string }[]>`SELECT data, user_id FROM brand_kits WHERE user_id = ${userId} AND (data->>'is_default')::boolean LIMIT 1`
    return row ? clipBrand(row.data, row.user_id) : null
  } catch (e) {
    // (no brand_kits table yet, or the database hiccuped: the clips are made without a kit)
    console.warn('[brand] the brand kit could not be read:', e instanceof Error ? e.message : e)
    return null
  }
}

/** When the logo shows on a clip `lengthMs` long: [start, end] ms (the frontend's logoSpan) */
export function logoSpan(logo: NonNullable<ClipBrand['logo']>, lengthMs: number): [number, number] {
  const len = Math.max(0, Math.round(lengthMs))
  const s = Math.min(len, logo.seconds * 1000)
  return logo.show === 'start' ? [0, s] : logo.show === 'end' ? [len - s, len] : [0, len]
}

/**
 * The caption style a kit gives a clip whose captions the job made in `font` (its script's font)
 * and `language`. The kit's font only where the captions are in English letters: a font for
 * Telugu or Hindi letters is kept (a brand font usually has no such letters).
 */
export function brandCaptionRow(c: NonNullable<ClipBrand['captions']>, caption: { font: string; language: string | null }) {
  const latin = caption.language === 'roman' || caption.font === 'roboto'
  return {
    font: latin && c.font ? c.font : caption.font, language: caption.language, position: 'bottom-center',
    size: c.size, color: c.color, animation: c.animation, highlight_color: c.highlight_color,
    uppercase: c.uppercase, words_per_line: c.words_per_line, template: c.template,
  }
}
