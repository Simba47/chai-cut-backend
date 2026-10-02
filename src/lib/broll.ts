/**
 * Automatic B-roll: short stock shots over the moments where a picture helps. The speaker keeps
 * talking underneath: the shots are cutaways, not inserts, and their own sound is muted.
 *
 * Stock source: Pexels when PEXELS_API_KEY is set (attribution "Videos from Pexels"), otherwise
 * Pixabay with PIXABAY_API_KEY (show users the videos are from Pixabay; search results are cached
 * 24 h as its API terms ask). Both free for commercial use.
 * Env: AUTO_BROLL=on to enable (default off). PEXELS_API_URL / PIXABAY_API_URL only for testing.
 */
import { GoogleGenerativeAI } from '@google/generative-ai'
import { createWriteStream } from 'node:fs'
import { pipeline } from 'node:stream/promises'
import { Readable } from 'node:stream'
import { join } from 'node:path'
import { buildLines, type FinderWord } from './clipFinder.js'

export const MAX_BROLL = 3
const MIN_MS = 1500, MAX_MS = 3000
const HOOK_MS = 3000       // never over the hook at the start…
const TAIL_MS = 2000       // …or the last 2 s
const APART_MS = 5000

export interface BrollMoment { start_ms: number; end_ms: number; query: string }
export interface StockVideo {
  /** "pexels:<id>" / "pixabay:<id>" — the same video is saved once per user */
  ref: string
  url: string; width: number; height: number
}

export function brollEnabled() {
  return process.env.AUTO_BROLL === 'on' && !!(process.env.PEXELS_API_KEY || process.env.PIXABAY_API_KEY)
}

/** The first result with a file of at least 720p, from whichever stock library is set up */
export async function searchStock(query: string): Promise<StockVideo | null> {
  return process.env.PEXELS_API_KEY ? searchPexels(query) : searchPixabay(query)
}

/**
 * Up to 3 moments in the clip (clip-relative ms) where a stock shot of what is being described
 * would help, each with an English search query. Gemini reads the clip's lines once; moments
 * are then held to 1.5–3 s, kept out of the first 3 s and last 2 s, and at least 5 s apart.
 */
export async function pickBrollMoments(words: FinderWord[], clipStart: number, clipEnd: number, apiKey: string): Promise<BrollMoment[]> {
  const lines = buildLines(words.filter(w => w.start_ms >= clipStart && w.end_ms <= clipEnd))
  if (lines.length < 2) return []
  const text = lines.map(l => `L${l.index} [${((l.start_ms - clipStart) / 1000).toFixed(1)}s] ${l.text}`).join('\n')
  const prompt = `You add B-roll (short stock video shots) to a vertical clip. The speech can be Telugu, Hindi, Tamil or another Indian language mixed with English.

Clip lines (L<number> [seconds from clip start] text):
${text}

Pick up to ${MAX_BROLL} lines where the speaker describes a concrete thing, place or action that a stock video could show (e.g. "a crowded cinema hall", "rain on a city street", "cooking biryani"). Skip moments where the speaker's own face or reaction matters (a joke's punchline, strong emotion, a direct address to the viewer), and anything about specific real people or brands.
For each, write a short English stock-footage search query (2-4 words).
Return ONLY JSON: [{"start_line": n, "end_line": n, "query": "..."}]. [] is fine.`

  const model = new GoogleGenerativeAI(apiKey).getGenerativeModel({ model: 'gemini-2.5-flash' })
  const out = (await model.generateContent(prompt)).response.text()
  const match = out.match(/\[[\s\S]*\]/)
  if (!match) return []
  const raw = JSON.parse(match[0]) as Array<{ start_line?: number; end_line?: number; query?: string }>
  const dur = clipEnd - clipStart
  const moments: BrollMoment[] = []
  for (const r of raw) {
    const a = lines[Number(r.start_line)], b = lines[Number(r.end_line ?? r.start_line)]
    const query = typeof r.query === 'string' ? r.query.trim().slice(0, 60) : ''
    if (!a || !b || !query) continue
    const start = Math.max(HOOK_MS, a.start_ms - clipStart)
    const end = Math.min(start + MAX_MS, Math.max(start + MIN_MS, b.end_ms - clipStart), dur - TAIL_MS)
    if (end - start < MIN_MS) continue
    if (moments.some(m => start < m.end_ms + APART_MS && m.start_ms < end + APART_MS)) continue
    moments.push({ start_ms: Math.round(start), end_ms: Math.round(end), query })
    if (moments.length >= MAX_BROLL) break
  }
  return moments.sort((x, y) => x.start_ms - y.start_ms)
}

type PexelsFile = { link: string; width: number; height: number; file_type?: string }
type PexelsVideo = { id: number; width: number; height: number; video_files: PexelsFile[] }

/** The first Pexels result with a file of at least 720p (portrait results first, then landscape) */
async function searchPexels(query: string): Promise<StockVideo | null> {
  const base = process.env.PEXELS_API_URL ?? 'https://api.pexels.com'
  for (const orientation of ['portrait', 'landscape']) {
    const res = await fetch(`${base}/videos/search?${new URLSearchParams({ query, orientation, per_page: '10' })}`, {
      headers: { Authorization: process.env.PEXELS_API_KEY! },
    })
    if (!res.ok) throw new Error(`Pexels ${res.status}`)
    const data = await res.json() as { videos?: PexelsVideo[] }
    for (const v of data.videos ?? []) {
      // The smallest mp4 that is still 720p or more
      const files = (v.video_files ?? [])
        .filter(f => (f.file_type ?? 'video/mp4') === 'video/mp4' && Math.min(f.width, f.height) >= 720)
        .sort((a, b) => a.width * a.height - b.width * b.height)
      if (files[0]) return { ref: `pexels:${v.id}`, url: files[0].link, width: files[0].width, height: files[0].height }
    }
  }
  return null
}

type PixabayFile = { url: string; width: number; height: number }
type PixabayHit = { id: number; videos: Record<string, PixabayFile> }
const pixabayCache = new Map<string, { at: number; hits: PixabayHit[] }>()

/**
 * The first Pixabay result with a file of at least 720p, vertical videos first (the API has no
 * orientation filter). Searches are cached for 24 h, as Pixabay's API terms ask.
 */
async function searchPixabay(query: string): Promise<StockVideo | null> {
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
  const pick = (hit: PixabayHit) => Object.values(hit.videos ?? {})
    .filter(f => f?.url && Math.min(f.width, f.height) >= 720)
    .sort((a, b) => a.width * a.height - b.width * b.height)[0]
  const withFile = hits.hits.map(h => ({ h, f: pick(h) })).filter((x): x is { h: PixabayHit; f: PixabayFile } => !!x.f)
  const best = withFile.find(x => x.f.height > x.f.width) ?? withFile[0]
  return best ? { ref: `pixabay:${best.h.id}`, url: best.f.url, width: best.f.width, height: best.f.height } : null
}

/** Download a stock file to disk */
export async function downloadStock(url: string, dir: string, ref: string): Promise<string> {
  const res = await fetch(url)
  if (!res.ok || !res.body) throw new Error(`Stock download ${res.status}`)
  const path = join(dir, `${ref.replace(':', '-')}.mp4`)
  await pipeline(Readable.fromWeb(res.body as Parameters<typeof Readable.fromWeb>[0]), createWriteStream(path))
  return path
}

/** A kept part of a framing segment, or a stock shot over it */
export interface FramedPart<K> { start_ms: number; end_ms: number; layout: 'vertical' | 'split' | 'trio'; slotKfs: K[][]; broll?: string; slotSources?: Array<number | null> }

/**
 * Cuts the clip's framing segments around the B-roll shots: each shot becomes its own vertical
 * segment showing the stock video (full frame; the export crops it to 9:16), and the framing on
 * either side is kept, its keyframes split at the shot.
 */
export function withBroll<K extends { t_ms: number; x: number; y: number; w: number; h: number }>(
  segments: FramedPart<K>[], shots: Array<{ start_ms: number; end_ms: number; videoId: string }>,
): FramedPart<K>[] {
  let out = segments
  for (const shot of shots) {
    // Never over a split or trio (a reaction or a related visual is on screen there)
    if (out.some(seg => seg.layout !== 'vertical' && seg.start_ms < shot.end_ms && seg.end_ms > shot.start_ms)) continue
    const next: FramedPart<K>[] = []
    for (const seg of out) {
      if (seg.broll || shot.end_ms <= seg.start_ms || shot.start_ms >= seg.end_ms) { next.push(seg); continue }
      const keep = (from: number, to: number) => {
        if (to - from < 100) return
        next.push({
          ...seg, start_ms: from, end_ms: to,
          // Borrowed footage moves on with the cut: a later piece starts later in it
          ...(seg.slotSources ? { slotSources: seg.slotSources.map(v => (v === null ? null : v + (from - seg.start_ms))) } : {}),
          slotKfs: seg.slotKfs.map(kfs => {
            const before = [...kfs].reverse().find(k => k.t_ms <= from) ?? kfs[0]
            const inside = kfs.filter(k => k.t_ms > from && k.t_ms < to)
            return [{ ...before, t_ms: from }, ...inside]
          }),
        })
      }
      keep(seg.start_ms, Math.max(seg.start_ms, shot.start_ms))
      const a = Math.max(seg.start_ms, shot.start_ms), b = Math.min(seg.end_ms, shot.end_ms)
      next.push({ start_ms: a, end_ms: b, layout: 'vertical', broll: shot.videoId,
        slotKfs: [[{ t_ms: a, x: 0, y: 0, w: 1, h: 1 } as K]] })
      keep(Math.min(seg.end_ms, shot.end_ms), seg.end_ms)
    }
    out = next
  }
  // A shot that crossed a framing boundary became two pieces: join them, or the second would
  // restart the stock video from its first frame
  const joined: FramedPart<K>[] = []
  for (const part of out.sort((x, y) => x.start_ms - y.start_ms)) {
    const last = joined[joined.length - 1]
    if (last?.broll && last.broll === part.broll && last.end_ms === part.start_ms) last.end_ms = part.end_ms
    else joined.push(part)
  }
  return joined
}
