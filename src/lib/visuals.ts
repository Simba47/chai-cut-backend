/**
 * Related visuals: while the speaker talks about something the video itself shows elsewhere (a
 * phone, an app screen, a product page, a chart), the clip shows the speaker on top and that
 * visual below (split; the visual fitted whole, muted, borrowed from its own moment).
 *
 * 1. findVisuals (once per video): stretches with no face on screen, from key frames
 *    (src/python/cutaways.py) — screen recordings, product shots, slides, cutaway footage.
 * 2. describeVisuals (once per video): Gemini looks at one picture of each and says what it
 *    shows, and whether a viewer would want to see it when it comes up.
 * 3. pickVisualMoments (per clip): Gemini reads the clip's lines and the list of visuals and
 *    picks where the speaker talks about something one of them shows.
 */
import { GoogleGenerativeAI } from '@google/generative-ai'
import { spawn, execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { mkdir, readdir, readFile, rm } from 'node:fs/promises'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { buildLines, type FinderWord } from './clipFinder.js'

const __dirname = dirname(fileURLToPath(import.meta.url))

/** The JSON array in a model's answer, or [] when there is none or it doesn't parse */
function jsonArray<T>(text: string): T[] {
  const match = text.match(/\[[\s\S]*\]/)
  if (!match) return []
  try { const v = JSON.parse(match[0]); return Array.isArray(v) ? v as T[] : [] } catch { return [] }
}
const MODEL = 'gemini-2.5-flash'

export interface Visual {
  /** Source-video ms */
  start_ms: number; end_ms: number
  image: string
  /** What it shows (from describeVisuals) */
  about?: string
}

/**
 * A graphic the video's editor put on screen next to the speaker (a product page, a screenshot,
 * a photo, a chart): where (fractions of the frame) and when (clip-relative ms)
 */
export interface Panel { start_ms: number; end_ms: number; x: number; y: number; w: number; h: number }

const PANEL_FPS = 1
const MAX_PANEL_FRAMES = 120

/**
 * Graphics placed on top of the camera picture during the clip, from one small frame a second
 * (one Gemini call with all of them). A vertical crop on the speaker would cut them off; the clip
 * shows them under the speaker instead. [] on any failure.
 */
export async function detectPanels(videoPath: string, startMs: number, endMs: number, tmp: string, apiKey: string, signal?: AbortSignal): Promise<Panel[]> {
  const dir = join(tmp, `panels-${startMs}`)
  await mkdir(dir, { recursive: true })
  try {
    await promisify(execFile)('ffmpeg', ['-v', 'error', '-ss', String(startMs / 1000), '-t', String((endMs - startMs) / 1000), '-i', videoPath,
      '-an', '-vf', `fps=${PANEL_FPS},scale=384:-1`, '-q:v', '5', join(dir, 'p_%04d.jpg')], { signal })
    const files = (await readdir(dir)).filter(f => f.endsWith('.jpg')).sort().slice(0, MAX_PANEL_FRAMES)
    if (!files.length) return []
    const parts: Array<{ inlineData: { mimeType: string; data: string } } | { text: string }> = []
    for (const [i, f] of files.entries()) {
      parts.push({ text: `F${i}:` })
      parts.push({ inlineData: { mimeType: 'image/jpeg', data: (await readFile(join(dir, f))).toString('base64') } })
    }
    parts.push({ text: `These are frames (one per second) from a video of people talking. In which frames is a graphic placed on top of the camera picture — a screenshot, web page, product image, photo, chart or slide shown in a box or panel beside or over the people? Ignore captions, logos, name tags and things that are physically in the room (TVs, posters, laptops on the table).
For each such frame give the graphic's box as box_2d [ymin, xmin, ymax, xmax] on a 0–1000 scale.
Return ONLY JSON: [{"frame": "F3", "box_2d": [ymin, xmin, ymax, xmax]}]. [] if there are none.` })
    const model = new GoogleGenerativeAI(apiKey).getGenerativeModel({ model: MODEL, generationConfig: { responseMimeType: 'application/json', temperature: 0 } })
    const out = (await model.generateContent(parts)).response.text()
    const hits = jsonArray<{ frame?: string; box_2d?: number[] }>(out)
      .map(r => ({ i: Number(String(r.frame ?? '').replace(/\D/g, '')), b: r.box_2d }))
      .filter((r): r is { i: number; b: number[] } => Number.isFinite(r.i) && Array.isArray(r.b) && r.b.length === 4)
      .map(r => ({ t: r.i * 1000 / PANEL_FPS, x: r.b[1] / 1000, y: r.b[0] / 1000, w: (r.b[3] - r.b[1]) / 1000, h: (r.b[2] - r.b[0]) / 1000 }))
      .filter(r => r.w >= 0.12 && r.h >= 0.12 && r.w * r.h <= 0.7)
      .sort((a, b) => a.t - b.t)
    // Neighbouring frames with the panel in about the same place are one panel
    const panels: Panel[] = []
    for (const h of hits) {
      const last = panels[panels.length - 1]
      const near = last && h.t - last.end_ms <= 1000 / PANEL_FPS + 100 && Math.abs(h.x - last.x) < 0.1 && Math.abs(h.y - last.y) < 0.1
      if (near) {
        last.end_ms = h.t + 1000 / PANEL_FPS
        // Its settled place: the largest box (it may slide in)
        if (h.w * h.h > last.w * last.h) Object.assign(last, { x: h.x, y: h.y, w: h.w, h: h.h })
      } else {
        panels.push({ start_ms: h.t, end_ms: h.t + 1000 / PANEL_FPS, x: h.x, y: h.y, w: h.w, h: h.h })
      }
    }
    return panels.filter(p => p.end_ms - p.start_ms >= 1500)
      .map(p => ({ ...p, end_ms: Math.min(p.end_ms, endMs - startMs) }))
  } finally {
    await rm(dir, { recursive: true, force: true }).catch(() => {})
  }
}

/**
 * The panels' exact boxes: the difference between a moment of the same camera shot without the
 * panel and one with it (src/python/panel_box.py), so the crop is tight on it. `cuts` are the
 * clip's camera cuts (clip ms); `videoStartMs` is where the clip starts in `video`.
 */
export async function refinePanels(panels: Panel[], cuts: number[], clipMs: number, video: string, videoStartMs: number): Promise<Panel[]> {
  const script = join(__dirname, '../../src/python/panel_box.py')
  const out: Panel[] = []
  for (const p of panels) {
    const shotStart = Math.max(0, ...cuts.filter(c => c <= p.start_ms + 500))
    const shotEnd = Math.min(clipMs, ...cuts.filter(c => c >= p.end_ms - 500))
    const without = p.start_ms - shotStart >= 600 ? (shotStart + p.start_ms) / 2
      : shotEnd - p.end_ms >= 600 ? (p.end_ms + shotEnd) / 2 : null
    if (without === null) { out.push(p); continue }
    const at = (ms: number) => ((videoStartMs + ms) / 1000).toFixed(3)
    const box = await new Promise<{ x: number; y: number; w: number; h: number } | null>(resolve => {
      const proc = spawn('python3', [script, '--video', video, '--without', at(without), '--with', at((p.start_ms + p.end_ms) / 2),
        '--box', [p.x, p.y, p.w, p.h].map(v => v.toFixed(4)).join(',')], { stdio: ['ignore', 'pipe', 'ignore'] })
      let s = ''
      proc.stdout.on('data', (d: Buffer) => { s += d })
      proc.on('close', () => { try { resolve(JSON.parse(s)) } catch { resolve(null) } })
      proc.on('error', () => resolve(null))
    })
    out.push(box ? { ...p, ...box } : p)
  }
  return out
}

/** A stretch of a clip (clip-relative ms) to show a visual under the speaker */
export interface VisualMoment { start_ms: number; end_ms: number; visual: Visual }

const MAX_MOMENTS = 2
const MIN_MS = 1500, MAX_MS = 4500
const LEAD_MS = 1500     // never in the clip's first moments (the hook)
const APART_MS = 4000

/** The faceless stretches of a video (source ms) with a picture of each; [] on any failure */
export async function findVisuals(videoPath: string, outDir: string, signal?: AbortSignal): Promise<Visual[]> {
  const script = join(__dirname, '../../src/python/cutaways.py')
  return new Promise(resolve => {
    const proc = spawn('python3', [script, '--video', videoPath, '--out-dir', outDir], {
      stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' },
    })
    const kill = () => proc.kill('SIGKILL')
    signal?.addEventListener('abort', kill, { once: true })
    let out = '', err = ''
    proc.stdout.on('data', (d: Buffer) => { out += d })
    proc.stderr.on('data', (d: Buffer) => { err += d })
    proc.on('close', () => {
      signal?.removeEventListener('abort', kill)
      const line = err.split(/\r?\n/).find(l => l.startsWith('[visuals]'))
      if (line) console.log(`[ai_edit] ${line}`)
      try { resolve(JSON.parse(out) as Visual[]) } catch { resolve([]) }
    })
    proc.on('error', () => resolve([]))
  })
}

/**
 * What each visual shows, and only those worth showing when they come up (a specific product,
 * app screen, page, chart, place or object — not logos, title cards or blank frames). One
 * Gemini call with all the pictures.
 */
export async function describeVisuals(visuals: Visual[], apiKey: string): Promise<Visual[]> {
  if (!visuals.length) return []
  const parts: Array<{ inlineData: { mimeType: string; data: string } } | { text: string }> = []
  for (const [i, v] of visuals.entries()) {
    parts.push({ text: `Image V${i + 1}:` })
    parts.push({ inlineData: { mimeType: 'image/jpeg', data: (await readFile(v.image)).toString('base64') } })
  }
  parts.push({ text: `These images come from cutaways in a video (screen recordings, product shots, footage shown while people talk).
For each image, write in simple English what it shows in at most 12 words, naming products, apps or brands you can read (e.g. "Apple website page for the iPhone Duo Folio case"). Mark useful=true when it shows a specific product, app screen, web page, document, chart, place or object that a viewer would want to see when the speakers mention it; useful=false for logos, title cards, intros, blank or blurry frames.
Return ONLY JSON: [{"id": "V1", "about": "...", "useful": true}, ...] with one entry per image.` })
  const model = new GoogleGenerativeAI(apiKey).getGenerativeModel({ model: MODEL, generationConfig: { responseMimeType: 'application/json', temperature: 0 } })
  const out = (await model.generateContent(parts)).response.text()
  const raw = jsonArray<{ id?: string; about?: string; useful?: boolean }>(out)
  const described: Visual[] = []
  for (const r of raw) {
    const i = Number(String(r.id ?? '').replace(/\D/g, '')) - 1
    const about = typeof r.about === 'string' ? r.about.trim().slice(0, 120) : ''
    if (visuals[i] && about && r.useful !== false) described.push({ ...visuals[i], about })
  }
  return described
}

/**
 * Where in the clip the speaker talks about something one of the visuals shows (clip-relative
 * ms, at most MAX_MOMENTS, 1.5–4.5 s each, not in the first moments, apart from each other).
 * Visuals already on screen in the clip at that moment are left out (they show anyway).
 */
export async function pickVisualMoments(words: FinderWord[], clipStart: number, clipEnd: number, visuals: Visual[], apiKey: string): Promise<VisualMoment[]> {
  const near = visuals.filter(v => !(v.start_ms < clipEnd && v.end_ms > clipStart))
  const lines = buildLines(words.filter(w => w.start_ms >= clipStart && w.end_ms <= clipEnd))
  if (!near.length || lines.length < 2) return []
  const text = lines.map(l => `L${l.index} [${((l.start_ms - clipStart) / 1000).toFixed(1)}s] ${l.text}`).join('\n')
  const list = near.map((v, i) => `V${i + 1}: ${v.about}`).join('\n')
  const prompt = `A vertical clip of people talking. The speech can be Telugu, Hindi, Tamil or another Indian language mixed with English; judge meaning in the original language.

Clip lines (L<number> [seconds from clip start] text):
${text}

Visuals available from the same video:
${list}

Pick up to ${MAX_MOMENTS} lines where the speaker is clearly talking about the very thing one of these visuals shows (the same product, app, page, object or event — not just the same topic). The visual will be shown under the speaker while they say it. Skip it when no visual truly matches.
Return ONLY JSON: [{"start_line": n, "end_line": n, "visual": "V<number>"}]. [] is fine.`
  const model = new GoogleGenerativeAI(apiKey).getGenerativeModel({ model: MODEL, generationConfig: { responseMimeType: 'application/json', temperature: 0 } })
  const out = (await model.generateContent(prompt)).response.text()
  const raw = jsonArray<{ start_line?: number; end_line?: number; visual?: string }>(out)
  const dur = clipEnd - clipStart
  const moments: VisualMoment[] = []
  for (const r of raw) {
    const a = lines.find(l => l.index === Number(r.start_line)), b = lines.find(l => l.index === Number(r.end_line ?? r.start_line))
    const visual = near[Number(String(r.visual ?? '').replace(/\D/g, '')) - 1]
    if (!a || !b || !visual) continue
    const start = Math.max(LEAD_MS, a.start_ms - clipStart)
    // As long as the line(s), within limits and within the visual's own length
    const end = Math.min(start + MAX_MS, Math.max(start + MIN_MS, b.end_ms - clipStart), dur, start + (visual.end_ms - visual.start_ms))
    if (end - start < MIN_MS) continue
    if (moments.some(m => start < m.end_ms + APART_MS && m.start_ms < end + APART_MS)) continue
    moments.push({ start_ms: Math.round(start), end_ms: Math.round(end), visual })
    if (moments.length >= MAX_MOMENTS) break
  }
  return moments.sort((x, y) => x.start_ms - y.start_ms)
}
