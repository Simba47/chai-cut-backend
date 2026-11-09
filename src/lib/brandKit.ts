import db from '../db.js'
import { cleanBrandWords, type BrandWords } from './brandWords.js'

/**
 * Brand kits on the clips Make my clips makes (user, 2026-10-17). A kit is a user's saved look
 * (the frontend's lib/shared/brandKit.ts: the editor's Brand kit tool, the brand_kits table). The
 * job's payload says which: 'none', a kit's id, or 'default' — the account's default kit, if it has
 * one. What it puts on each clip mirrors what the frontend puts on a clip made by hand
 * (services/brandKits.ts brandNewClips): its logo where and when the kit says, its caption style
 * when its captions are on, and here also its title style on the hook. Keep the two in step.
 * The rest of a kit (its end card, intro and outro: brand kit Phase 4) the clip gets from the frontend's brand
 * engine the first time it catches up with its kit — when it's opened in the editor or exported —
 * because the version saved with the clip (`applied`) has it off (appliedHere).
 */

export interface ClipBrand {
  /** The kit (its brand_kits id): the clip is saved with it switched on (clips.brand) */
  kit_id?: string
  /**
   * The kit's settings as saved, kept with the clip (clips.brand.applied) as the version it got: when
   * the kit changes, the clip follows it (the frontend's brand engine, modules/brand/engine.ts)
   */
  applied?: unknown
  title: { font: string | null; color: string; size: number }
  /** The caption style fields (null: the kit's captions are off — the job's own style is kept) */
  captions: {
    template: string | null; animation: string; font: string | null; size: number; color: string
    highlight_color: string; uppercase: boolean; words_per_line: number | null
  } | null
  /** Its logo on the frame (shares of it) and when it shows; null: no logo */
  logo: ClipLogo | null
  /** Its words (brand kit Phase 5, lib/brandWords.ts): its spellings and tone in the post text the job writes */
  words: BrandWords
}

type LogoCorner = 'tl' | 'tr' | 'bl' | 'br'
export interface LogoBox { x: number; y: number; w: number; h: number }
/** `smart`: moved to the clearest corner on each clip (placedLogo); never one placed by hand */
export interface ClipLogo extends LogoBox { path: string; opacity: number; show: 'all' | 'start' | 'end'; seconds: number; corner: LogoCorner; safe: boolean; smart: boolean }

const HEX = /^#[0-9a-f]{6}$/i
/** The plans whose kits can be shared with a team (the frontend's lib/shared/brandTeam.ts TEAM_SEATS): a kit shared by someone on another plan waits */
const TEAM_PLANS = ['agency']
const ANIMATIONS = ['karaoke', 'fade', 'none', 'pop', 'highlight', 'bounce', 'word', 'hormozi', 'box', 'glow']
const num = (v: unknown, lo: number, hi: number, fallback: number) => (typeof v === 'number' && isFinite(v) ? Math.min(hi, Math.max(lo, v)) : fallback)
const color = (v: unknown, fallback: string) => (typeof v === 'string' && HEX.test(v) ? v.toLowerCase() : fallback)
const r4 = (v: number) => Math.round(v * 10000) / 10000
/** The logo's margin from the frame's edges in its corner (share of the frame's width): the frontend's LOGO_MARGIN */
const LOGO_MARGIN = 0.04
/** What Reels and Shorts draw over a video, for a logo kept clear of it (kit logo.safe): the frontend's LOGO_SAFE */
const LOGO_SAFE = { top: 0.08, bottom: 0.13, right: 0.14, edge: 0.035 }

/** Where a logo sits in a corner of the 1080 × 1920 frame: the frontend's logoBox */
function cornerSpot(w: number, h: number, safe: boolean, corner: LogoCorner) {
  const mx = safe ? Math.max(LOGO_MARGIN, LOGO_SAFE.edge) : LOGO_MARGIN
  const top = safe ? LOGO_SAFE.top : (LOGO_MARGIN * 1080) / 1920, bottom = safe ? LOGO_SAFE.bottom : (LOGO_MARGIN * 1080) / 1920
  const right = safe && corner === 'br' ? LOGO_SAFE.right : mx
  return { x: r4(corner.endsWith('l') ? mx : 1 - right - w), y: r4(corner.startsWith('t') ? top : 1 - bottom - h) }
}

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
    const corner = (['tl', 'tr', 'bl', 'br'] as const).find(c => c === l.corner) ?? 'tr'
    const safe = l.safe !== false
    const placed = typeof l.x === 'number' && typeof l.y === 'number' && isFinite(l.x) && isFinite(l.y)
    const spot = placed ? { x: r4(Math.min(1 - w, Math.max(0, l.x as number))), y: r4(Math.min(1 - h, Math.max(0, l.y as number))) } : cornerSpot(w, h, safe, corner)
    logo = {
      path, ...spot, w: r4(w), h: r4(h), opacity: num(l.opacity, 0.1, 1, 0.9),
      show: (['all', 'start', 'end'] as const).find(s => s === l.show) ?? 'all', seconds: Math.round(num(l.seconds, 1, 15, 3)),
      corner, safe, smart: !placed && l.smart !== false,
    }
  }
  return {
    words: cleanBrandWords(d.words),
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
    logo,
  }
}

/**
 * The kit a job's clips get: 'none' → none; a kit's id → that kit (the user's own, or one shared with
 * them: brand_kit_shares); else the user's default kit, if any. A shared kit's logo is its owner's.
 */
export async function loadClipBrand(userId: string, choice: string | undefined): Promise<ClipBrand | null> {
  if (choice === 'none') return null
  try {
    const [row] = choice && /^[0-9a-f-]{36}$/i.test(choice)
      ? await db<{ id: string; data: unknown; user_id: string }[]>`
          SELECT bk.id, bk.data, bk.user_id FROM brand_kits bk WHERE bk.id = ${choice}
            AND (bk.user_id = ${userId} OR (EXISTS (SELECT 1 FROM brand_kit_shares s WHERE s.kit_id = bk.id AND s.user_id = ${userId})
              AND EXISTS (SELECT 1 FROM users o WHERE o.id = bk.user_id AND o.plan = ANY(${TEAM_PLANS}))))`
      : await db<{ id: string; data: unknown; user_id: string }[]>`SELECT id, data, user_id FROM brand_kits WHERE user_id = ${userId} AND (data->>'is_default')::boolean LIMIT 1`
    return row ? { ...clipBrand(row.data, row.user_id), kit_id: row.id, applied: appliedHere(row.data) } : null
  } catch (e) {
    // (no brand_kits table yet, or the database hiccuped: the clips are made without a kit)
    console.warn('[brand] the brand kit could not be read:', e instanceof Error ? e.message : e)
    return null
  }
}

/**
 * The brand words a clip's captions are exported with (brand kit Phase 5): those of the kit switched on
 * for it (one its owner can still use), else of the owner's default kit — as the frontend's
 * services/brandKits.ts brandWordsFor. Null: none, or the kits couldn't be read (the words then go as they are).
 */
export async function clipBrandWords(clipId: string): Promise<BrandWords | null> {
  try {
    const [row] = await db<{ words: unknown }[]>`
      SELECT bk.data->'words' AS words FROM clips c JOIN videos v ON v.id = c.video_id
      JOIN brand_kits bk ON (bk.id::text = to_jsonb(c)->'brand'->>'kit_id'
          AND (bk.user_id = v.user_id OR (EXISTS (SELECT 1 FROM brand_kit_shares s WHERE s.kit_id = bk.id AND s.user_id = v.user_id)
            AND EXISTS (SELECT 1 FROM users o WHERE o.id = bk.user_id AND o.plan = ANY(${TEAM_PLANS})))))
        OR (bk.user_id = v.user_id AND (bk.data->>'is_default')::boolean)
      WHERE c.id = ${clipId}
      ORDER BY (bk.id::text = to_jsonb(c)->'brand'->>'kit_id') DESC LIMIT 1`
    return row ? cleanBrandWords(row.words) : null
  } catch (e) {
    console.warn('[brand] the brand words could not be read:', e instanceof Error ? e.message : e)
    return null
  }
}

/**
 * A kit's settings as this job puts them on: what it doesn't put on (the kit's end card, intro and outro) off.
 * Saved with the clip as the version it got, so the frontend's brand engine adds the rest later as it
 * would a kit's newly switched-on end card.
 */
export function appliedHere(data: unknown): unknown {
  if (!data || typeof data !== 'object') return data
  const d = data as Record<string, unknown>
  const ends = d.ends && typeof d.ends === 'object' ? d.ends as Record<string, unknown> : null
  return { ...d, ...(ends ? { ends: { ...ends, card: false, intro: null, outro: null } } : {}) }
}

const overlapArea = (a: LogoBox, b: LogoBox) =>
  Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)) * Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y))

/**
 * The logo in the corner it's clearest in on this clip (kit logo.smart): the frontend's pickLogoCorner —
 * the kit's own corner when nothing is in the way there, else the one covering the least of `avoid`
 */
export function placedLogo(logo: ClipLogo, avoid: LogoBox[]): ClipLogo {
  if (!logo.smart || !avoid.length) return logo
  const scored = [...new Set<LogoCorner>([logo.corner, 'tr', 'tl', 'br', 'bl'])].map(c => {
    const spot = cornerSpot(logo.w, logo.h, logo.safe, c)
    return { spot, v: avoid.reduce((t, a) => t + overlapArea({ ...spot, w: logo.w, h: logo.h }, a), 0) }
  })
  const best = scored.find(s => s.v < 1e-4) ?? scored.reduce((b, s) => (s.v < b.v - 1e-6 ? s : b))
  return { ...logo, ...best.spot }
}

type LogoView = LogoBox & { t_ms: number }
function viewAt(kfs: LogoView[], t: number): LogoBox | null {
  const after = kfs.findIndex(k => k.t_ms > t)
  if (after <= 0) return kfs[after === 0 ? 0 : kfs.length - 1] ?? null
  const a = kfs[after - 1], b = kfs[after], p = (t - a.t_ms) / (b.t_ms - a.t_ms)
  const mix = (u: number, v: number) => u + (v - u) * p
  return { x: mix(a.x, b.x), y: mix(a.y, b.y), w: mix(a.w, b.w), h: mix(a.h, b.h) }
}

/**
 * What a clip's logo keeps off (the frontend's EditorShell brandAvoid): the captions' band, and the
 * faces in its full-frame parts, about once a second — a face seen longer weighs more.
 * `frames`: the faces read, at ms from the clip's start, as shares of the source picture.
 */
export function logoAvoid(
  segments: Array<{ start_ms: number; end_ms: number; layout: string; slotKfs: LogoView[][]; slotSources?: Array<number | null> }>,
  frames: Array<{ t_ms: number; faces: LogoBox[] }>,
  captions: boolean,
): LogoBox[] {
  // (the band the frontend's captionBand gives captions at their usual height)
  const out: LogoBox[] = captions ? [{ x: 0.06, y: 0.76, w: 0.88, h: 0.16 }] : []
  if (!frames.length) return out
  for (const seg of segments) {
    if (seg.layout !== 'vertical' || seg.slotKfs.length !== 1 || seg.slotSources?.[0] != null) continue
    for (let t = seg.start_ms; t < seg.end_ms; t += 1000) {
      const v = viewAt(seg.slotKfs[0], t)
      if (!v || v.w <= 0 || v.h <= 0) continue
      const near = frames.reduce((b, f) => (Math.abs(f.t_ms - t) < Math.abs(b.t_ms - t) ? f : b))
      if (Math.abs(near.t_ms - t) > 500) continue
      for (const f of near.faces) {
        const x = (f.x - v.x) / v.w, y = (f.y - v.y) / v.h, w = f.w / v.w, h = f.h / v.h
        if (x + w > 0 && x < 1 && y + h > 0 && y < 1) out.push({ x, y, w, h })
      }
    }
  }
  return out
}

/** When the logo shows on a clip `lengthMs` long: [start, end] ms (the frontend's logoSpan) */
export function logoSpan(logo: ClipLogo, lengthMs: number): [number, number] {
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
