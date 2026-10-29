/**
 * Automatic B-roll (Make my clips' "B-rolls"): short stock shots over the moments where a picture
 * helps. The B-roll plays on top and the speaker stays below, talking (the user, 2026-10-29: "speaker
 * below and broll top" — a Gameplay-split frame with its rows swapped); its own sound is muted.
 *
 * Picking (a wrong shot is worse than none):
 *   1. Gemini reads the clip's lines and picks up to 3 moments, each with 2–3 English searches.
 *   2. Each moment gets up to CANDIDATES stock videos from those searches.
 *   3. One Gemini call looks at every candidate's picture and picks the one that shows what is said
 *      — or none.
 * The candidates are kept on the shot (its frame item's `broll`), so the editor can swap to another.
 *
 * Stock: Pixabay with PIXABAY_API_KEY (users are shown the videos are from Pixabay; searches cached
 * 24 h as its API terms ask), Pexels too when PEXELS_API_KEY is set. Both free for commercial use.
 * Env: AUTO_BROLL=on to enable (default off). PEXELS_API_URL / PIXABAY_API_URL only for testing.
 */
import { createWriteStream } from 'node:fs'
import { pipeline } from 'node:stream/promises'
import { Readable } from 'node:stream'
import { join } from 'node:path'
import { buildLines, type FinderWord } from './clipFinder.js'
import { askGemini, type GeminiPart } from './gemini.js'

export const MAX_BROLL = 3
const MIN_MS = 1500, MAX_MS = 3000
const HOOK_MS = 3000       // never over the hook at the start…
const TAIL_MS = 2000       // …or the last 2 s
const APART_MS = 5000
/** Stock videos looked at for one moment */
export const CANDIDATES = 6

export interface BrollMoment {
  start_ms: number; end_ms: number
  /** English stock searches, the best first */
  queries: string[]
  /** What is said there (for the pick) */
  said: string
}
/** A stock video, as the editor's stock search lists it (server/services/stock.ts StockItem) */
export interface StockCandidate {
  /** "pixabay:<id>" / "pexels:<id>" — the same video is saved once per user */
  ref: string
  provider: 'pixabay' | 'pexels'
  title: string
  /** A small picture of it, and a small file to preview it */
  thumb: string; preview: string
  /** The file used (720p or more) */
  url: string
  width: number; height: number; duration: number
}

export function brollEnabled() {
  return process.env.AUTO_BROLL === 'on' && !!(process.env.PEXELS_API_KEY || process.env.PIXABAY_API_KEY)
}

/**
 * Up to 3 moments in the clip (clip-relative ms) where a stock shot of what is being described
 * would help, each with English searches. Gemini reads the clip's lines once; moments are then held
 * to 1.5–3 s, kept out of the first 3 s and last 2 s, and at least 5 s apart.
 */
export async function pickBrollMoments(words: FinderWord[], clipStart: number, clipEnd: number): Promise<BrollMoment[]> {
  const lines = buildLines(words.filter(w => w.start_ms >= clipStart && w.end_ms <= clipEnd))
  if (lines.length < 2) return []
  const text = lines.map(l => `L${l.index} [${((l.start_ms - clipStart) / 1000).toFixed(1)}s] ${l.text}`).join('\n')
  const prompt = `You add B-roll (short stock video shots) to a vertical clip. The speech can be Telugu, Hindi, Tamil or another Indian language mixed with English.

Clip lines (L<number> [seconds from clip start] text):
${text}

Pick up to ${MAX_BROLL} lines where the speaker describes a concrete thing, place or action that a stock video could show (e.g. "a crowded cinema hall", "rain on a city street", "cooking biryani"). Skip moments where the speaker's own face or reaction matters (a joke's punchline, strong emotion, a direct address to the viewer), and anything about specific real people or brands. Picking nothing is better than a weak pick.
For each, write 2 or 3 short English stock-footage searches (2-4 words each), the most exact first, then a broader one. When the talk is about India or an Indian place, food, festival or custom, say so in the search (e.g. "Hyderabad street traffic", "Indian wedding dance").
Return ONLY JSON: [{"start_line": n, "end_line": n, "queries": ["...", "..."]}]. [] is fine.`

  const out = await askGemini(prompt, { json: true, temperature: 0 })
  const match = out.match(/\[[\s\S]*\]/)
  if (!match) return []
  const raw = JSON.parse(match[0]) as Array<{ start_line?: number; end_line?: number; queries?: unknown; query?: unknown }>
  const dur = clipEnd - clipStart
  const moments: BrollMoment[] = []
  for (const r of raw) {
    const a = lines[Number(r.start_line)], b = lines[Number(r.end_line ?? r.start_line)]
    const list = Array.isArray(r.queries) ? r.queries : [r.query]
    const queries = [...new Set(list.filter((q): q is string => typeof q === 'string').map(q => q.trim().slice(0, 60)).filter(Boolean))].slice(0, 3)
    if (!a || !b || !queries.length) continue
    const start = Math.max(HOOK_MS, a.start_ms - clipStart)
    const end = Math.min(start + MAX_MS, Math.max(start + MIN_MS, b.end_ms - clipStart), dur - TAIL_MS)
    if (end - start < MIN_MS) continue
    if (moments.some(m => start < m.end_ms + APART_MS && m.start_ms < end + APART_MS)) continue
    const said = lines.slice(a.index, b.index + 1).map(l => l.text).join(' ').slice(0, 300)
    moments.push({ start_ms: Math.round(start), end_ms: Math.round(end), queries, said })
    if (moments.length >= MAX_BROLL) break
  }
  return moments.sort((x, y) => x.start_ms - y.start_ms)
}

/**
 * Up to CANDIDATES stock videos for a moment: the searches' results taken in turn (the first search's
 * best, the second's best, …), each video once, Pixabay first, then Pexels when it is set up
 */
export async function searchCandidates(queries: string[]): Promise<StockCandidate[]> {
  const lists = await Promise.all(queries.flatMap(q => [
    ...(process.env.PIXABAY_API_KEY ? [searchPixabay(q).catch(() => [])] : []),
    ...(process.env.PEXELS_API_KEY ? [searchPexels(q).catch(() => [])] : []),
  ]))
  const out: StockCandidate[] = []
  for (let k = 0; out.length < CANDIDATES && lists.some(l => l.length > k); k++) {
    for (const l of lists) {
      const c = l[k]
      if (c && !out.some(o => o.ref === c.ref) && out.length < CANDIDATES) out.push(c)
    }
  }
  return out
}

/**
 * Which candidate shows what is said, for each moment — or none. One Gemini call with every
 * candidate's picture. The answer per moment: an index into its candidates, or null.
 */
export async function chooseBroll(moments: Array<{ said: string; queries: string[]; candidates: StockCandidate[] }>): Promise<Array<number | null>> {
  const parts: GeminiPart[] = []
  const shown: number[][] = []
  for (const [mi, m] of moments.entries()) {
    parts.push({ text: `Moment M${mi + 1}. The speaker says (may be Telugu/Hindi/Tamil mixed with English): "${m.said}"\nSearched for: ${m.queries.join(' / ')}` })
    const ok: number[] = []
    for (const [ci, c] of m.candidates.entries()) {
      const img = await thumbnail(c.thumb)
      if (!img) continue
      parts.push({ text: `M${mi + 1} candidate C${ci + 1}: ${c.title}` })
      parts.push({ inlineData: { mimeType: 'image/jpeg', data: img } })
      ok.push(ci)
    }
    shown.push(ok)
  }
  if (!shown.some(s => s.length)) return moments.map(() => null)
  parts.push({ text: `Each candidate picture is the first frame of a stock video that would play above the speaker for 2-3 seconds while they say the words of its moment.
For each moment, pick the ONE candidate that clearly shows what the speaker is talking about, as a viewer would expect to see it. Answer null when none clearly fits — a wrong or generic shot is worse than none. Never pick: a picture whose main subject is a person's face (it would look like someone in the video), text or logos as the main subject, anything clearly from another country when the talk is about an Indian place or custom, or anything unsafe for work.
Return ONLY JSON: [{"moment": "M1", "pick": "C2"}, {"moment": "M2", "pick": null}, ...], one entry per moment.` })
  const out = await askGemini(parts, { json: true, temperature: 0 })
  const match = out.match(/\[[\s\S]*\]/)
  const raw = match ? (JSON.parse(match[0]) as Array<{ moment?: unknown; pick?: unknown }>) : []
  return moments.map((_, mi) => {
    const r = raw.find(x => String(x.moment ?? '').replace(/\D/g, '') === String(mi + 1))
    const ci = typeof r?.pick === 'string' ? Number(r.pick.replace(/\D/g, '')) - 1 : NaN
    return Number.isInteger(ci) && shown[mi].includes(ci) ? ci : null
  })
}

/** A candidate's picture as base64 JPEG (null when it can't be fetched) */
async function thumbnail(url: string): Promise<string | null> {
  if (!url) return null
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(15_000) })
    if (!res.ok) return null
    const body = Buffer.from(await res.arrayBuffer())
    return body.length > 0 && body.length < 2_000_000 ? body.toString('base64') : null
  } catch {
    return null
  }
}

type PexelsFile = { link: string; width: number; height: number; file_type?: string }
type PexelsVideo = { id: number; url: string; image: string; duration: number; video_files: PexelsFile[] }

/** Pexels results with a file of at least 720p (portrait first, then landscape) */
async function searchPexels(query: string): Promise<StockCandidate[]> {
  const base = process.env.PEXELS_API_URL ?? 'https://api.pexels.com'
  const out: StockCandidate[] = []
  for (const orientation of ['portrait', 'landscape']) {
    const res = await fetch(`${base}/videos/search?${new URLSearchParams({ query, orientation, per_page: '10' })}`, {
      headers: { Authorization: process.env.PEXELS_API_KEY! },
    })
    if (!res.ok) throw new Error(`Pexels ${res.status}`)
    const data = await res.json() as { videos?: PexelsVideo[] }
    for (const v of data.videos ?? []) {
      // The smallest mp4 that is still 720p or more; the smallest of all to preview
      const files = (v.video_files ?? []).filter(f => (f.file_type ?? 'video/mp4') === 'video/mp4')
      const big = files.filter(f => Math.min(f.width, f.height) >= 720).sort((a, b) => a.width * a.height - b.width * b.height)[0]
      const small = [...files].sort((a, b) => a.width * a.height - b.width * b.height)[0]
      if (!big || !small) continue
      const title = v.url?.split('/').filter(Boolean).pop()?.replace(/-\d+$/, '').replace(/-/g, ' ') || query
      out.push({ ref: `pexels:${v.id}`, provider: 'pexels', title, thumb: v.image, preview: small.link, url: big.link, width: big.width, height: big.height, duration: v.duration })
    }
  }
  return out
}

type PixabayFile = { url: string; width: number; height: number; thumbnail?: string }
type PixabayHit = { id: number; tags?: string; duration?: number; isAiGenerated?: boolean; isLowQuality?: boolean; videos: Record<string, PixabayFile> }
const pixabayCache = new Map<string, { at: number; hits: PixabayHit[] }>()

/**
 * Pixabay results with a file of at least 720p, vertical videos first (the API has no orientation
 * filter); AI-made and low-quality ones left out. Searches are cached for 24 h, as Pixabay's API terms ask.
 */
async function searchPixabay(query: string): Promise<StockCandidate[]> {
  const key = query.toLowerCase()
  let hits = pixabayCache.get(key)
  if (!hits || Date.now() - hits.at > 24 * 3600_000) {
    const base = process.env.PIXABAY_API_URL ?? 'https://pixabay.com'
    const params = new URLSearchParams({ key: process.env.PIXABAY_API_KEY!, q: query.slice(0, 100), safesearch: 'true', per_page: '20' })
    const res = await fetch(`${base}/api/videos/?${params}`)
    if (!res.ok) throw new Error(`Pixabay ${res.status}`)
    hits = { at: Date.now(), hits: ((await res.json()) as { hits?: PixabayHit[] }).hits ?? [] }
    pixabayCache.set(key, hits)
  }
  const found = hits.hits.flatMap(h => {
    if (h.isAiGenerated || h.isLowQuality) return []
    const files = Object.values(h.videos ?? {}).filter(f => f?.url)
    const big = files.filter(f => Math.min(f.width, f.height) >= 720).sort((a, b) => a.width * a.height - b.width * b.height)[0]
    const small = [...files].sort((a, b) => a.width * a.height - b.width * b.height)[0]
    if (!big || !small) return []
    return [{
      ref: `pixabay:${h.id}`, provider: 'pixabay' as const, title: (h.tags ?? query).split(',').slice(0, 3).map(t => t.trim()).join(', '),
      thumb: small.thumbnail ?? big.thumbnail ?? '', preview: small.url, url: big.url, width: big.width, height: big.height, duration: h.duration ?? 0,
    }]
  })
  return [...found.filter(c => c.height > c.width), ...found.filter(c => c.height <= c.width)]
}

/** Download a stock file to disk */
export async function downloadStock(url: string, dir: string, ref: string): Promise<string> {
  const res = await fetch(url)
  if (!res.ok || !res.body) throw new Error(`Stock download ${res.status}`)
  const path = join(dir, `${ref.replace(':', '-')}.mp4`)
  await pipeline(Readable.fromWeb(res.body as Parameters<typeof Readable.fromWeb>[0]), createWriteStream(path))
  return path
}

// ── Laying it: B-roll on top, the speaker below ────────────────────────────────

/** The frame a B-roll shot is laid in: the Gameplay split (speaker | footage) with its rows swapped */
export const BROLL_LAYOUT = 'frame_gameplay'
/** A clip's framing part (ai_edit.ts ClipSegment) */
export interface FramedPart<K> {
  start_ms: number; end_ms: number
  layout: 'vertical' | 'split' | 'trio' | 'frame_blur_fill' | 'frame_gameplay'
  slotKfs: K[][]; slotSources?: Array<number | null>
  /** How many people it shows (the Rulebook's formats); unknown: the old framing */
  people?: number
  /** A frame's settings (segments.frame) */
  frame?: Record<string, unknown>
}
export interface BrollShot {
  start_ms: number; end_ms: number
  /** The stock video (the user's asset copy) */
  videoId: string
  /** The speaker's view in the bottom row (fractions of the source; the row's 9:8 shape) */
  view: { x: number; y: number; w: number; h: number }
  /** For the editor's swap: the searches, the candidates and which one is shown */
  query: string
  alts: StockCandidate[]
  at: number
}

/** A part B-roll can go over: one person in a Vertical (people not known: the old framing's Vertical), not a borrowed-footage part */
const takesBroll = <K>(s: FramedPart<K>) => s.layout === 'vertical' && !s.slotSources && (s.people === undefined || s.people === 1)

/**
 * Where a moment can take B-roll: over one person in a Vertical (never over a Split, a Trio, a group or
 * a frame — they show people side by side — nor over a part with no one in it), on one camera shot (the
 * speaker's view below holds still): the longest stretch of the moment inside one such part and between
 * camera cuts, at least MIN_MS. Null: it can't.
 */
export function brollWindow<K>(segments: FramedPart<K>[], cuts: number[], m: { start_ms: number; end_ms: number }): { start_ms: number; end_ms: number } | null {
  let best: { start_ms: number; end_ms: number } | null = null
  for (const seg of segments) {
    const a0 = Math.max(m.start_ms, seg.start_ms), b0 = Math.min(m.end_ms, seg.end_ms)
    if (b0 - a0 < MIN_MS || !takesBroll(seg)) continue
    const edges = [a0, ...cuts.filter(c => c > a0 && c < b0).sort((x, y) => x - y), b0]
    for (let i = 0; i < edges.length - 1; i++) {
      const a = edges[i], b = edges[i + 1]
      if (b - a >= MIN_MS && (!best || b - a > best.end_ms - best.start_ms)) best = { start_ms: a, end_ms: b }
    }
  }
  return best
}

/**
 * The clip's parts with each B-roll shot laid in: the part under it is cut around it (its keyframes
 * split at the shot), and the shot becomes a Gameplay-split frame with its rows swapped — the stock
 * video on top (a muted video on the frame's lane 1, from its start), the main video below in the
 * speaker's view (lane 0). Shots go in only where brollWindow allows.
 */
export function withBroll<K extends { t_ms: number; x: number; y: number; w: number; h: number }>(
  segments: FramedPart<K>[], shots: BrollShot[],
): FramedPart<K>[] {
  let out = segments
  for (const [n, shot] of shots.entries()) {
    const seg = out.find(s => s.start_ms <= shot.start_ms && s.end_ms >= shot.end_ms)
    if (!seg || !takesBroll(seg)) continue
    const keep = (from: number, to: number): FramedPart<K>[] => {
      if (to - from < 100) return []
      return [{
        ...seg, start_ms: from, end_ms: to,
        slotKfs: seg.slotKfs.map(kfs => {
          const before = [...kfs].reverse().find(k => k.t_ms <= from) ?? kfs[0]
          return [{ ...before, t_ms: from }, ...kfs.filter(k => k.t_ms > from && k.t_ms < to)]
        }),
      }]
    }
    const a = shot.start_ms, b = shot.end_ms
    const laid: FramedPart<K> = {
      start_ms: a, end_ms: b, layout: BROLL_LAYOUT, people: 1,
      slotKfs: [[{ t_ms: a, ...shot.view } as K]],
      frame: {
        main_slots: [0],
        // Rows swapped: the footage's row (1) on top, the speaker's (0) below
        row_order: [1, 0],
        items: [{
          id: `broll-${a}-${n}`, lane: 1, kind: 'video', start_ms: a, end_ms: b,
          source_video_id: shot.videoId, source_offset_ms: 0, volume: 1, muted: true,
          broll: { query: shot.query, alts: shot.alts, at: shot.at },
        }],
      },
    }
    out = out.flatMap(s => (s === seg ? [...keep(seg.start_ms, a), laid, ...keep(b, seg.end_ms)] : [s]))
  }
  return out.sort((x, y) => x.start_ms - y.start_ms)
}
