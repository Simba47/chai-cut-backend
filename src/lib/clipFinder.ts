/**
 * Clip finding over a whole transcript.
 *
 * The transcript is cut into numbered lines that end on sentence boundaries or pauses. The model
 * reads the lines in ~10-minute windows (in parallel) and answers with line numbers, never
 * times, so every clip starts where a line starts and ends where a line ends. Each candidate
 * carries five 0–10 sub-scores that add up to a 0–99 viral score. Candidates from all windows
 * are merged, de-duplicated and held to 15–90 s.
 *
 * No framework or database imports: the model call is passed in, so this file is copied as-is
 * between the repos: chai-cut-frontend/src/server/services/clipFinder.ts and
 * chai-cut-backend/src/lib/clipFinder.ts. Keep the two identical.
 */

export type FinderWord = { word: string; start_ms: number; end_ms: number }

export interface TranscriptLine {
  index: number; start_ms: number; end_ms: number; text: string
  /** Its first and last word (indexes into the words it was built from) */
  first_word?: number; last_word?: number
}

/**
 * A short piece of a line, for exact clip edges (refineEdges): it ends at a sentence end, at a pause
 * after a few words, or at its line's end. `line`: the line it is part of.
 */
export interface Phrase { index: number; line: number; start_ms: number; end_ms: number; text: string }

export interface Subscores { hook: number; emotion: number; story: number; value: number; clarity: number }

export interface FoundClip {
  start_line: number
  end_line: number
  start_ms: number
  end_ms: number
  title: string
  summary: string
  reason: string
  score: number
  subscores: Subscores
  /** Which window found it */
  window: number
}

/** 'best' = the strongest moments in general; 'search' = only moments matching the user's request */
export type FinderMode = { kind: 'best' } | { kind: 'search'; criteria: string }

/** Sends one prompt to the model and returns its raw text answer (`opts.temperature`: 0 for the same answer every time) */
export type AskModel = (system: string, user: string, opts?: { temperature?: number }) => Promise<string>

export const MIN_CLIP_MS = 15_000
/** What the model is asked for; a clip may run on to HARD_MAX_CLIP_MS to finish its thought */
export const MAX_CLIP_MS = 90_000
/** Past this a clip is shortened (back to a sentence end): Reels and Shorts both take 2 minutes */
export const HARD_MAX_CLIP_MS = 120_000
const LINE_PAUSE_MS = 2_000
/** From this many words a line ends at the next breath (a gap of BREATH_MS) … */
const LINE_SOFT_WORDS = 25
/** … and at this many in any case (speech with no punctuation and no breath) */
const LINE_HARD_WORDS = 45
const BREATH_MS = 250
/** Room kept before a clip's first word and after its last (taken only from silence) */
const LEAD_MS = 350
const TAIL_MS = 600
/** Silence left between the room and a neighbouring word, so none of it is heard */
const EDGE_KEEP_MS = 60
/** A phrase (refineEdges' pieces) ends at a pause this long once it has PHRASE_MIN_WORDS words, and at PHRASE_MAX_WORDS in any case */
const PHRASE_PAUSE_MS = 350
const PHRASE_MIN_WORDS = 3
const PHRASE_MAX_WORDS = 25
/** Lines shown around a clip when its edges are checked (lib: refineEdges) */
const REFINE_BEFORE = 8
const REFINE_AFTER = 12
/** Search results checked at most (the best by score; the rest keep the edges they were found with) */
const REFINE_MAX = 12
const WINDOW_MS = 10 * 60_000
const WINDOW_OVERLAP_MS = 60_000
const MAX_PARALLEL = 4
const BEST_PER_WINDOW = 4
const BEST_TOP_N = 8
const SEARCH_MIN_SCORE = 40
// Sentence-ending punctuation in English and Indian scripts (danda, double danda, pipe)
const SENTENCE_END = /[.?!।॥|]["'”’)]*$/

// ── Lines ───────────────────────────────────────────────────────────────────────

/**
 * A new line after a pause over 2 s, after sentence-ending punctuation, or — speech with little
 * punctuation (Telugu, Hindi…) — at the first breath after 25 words (45 at most). A line used to
 * end at exactly 25 words, mid-sentence, and clips could only start and end on line ends (user,
 * 2026-10-31: clips cut off mid-sentence).
 */
export function buildLines(words: FinderWord[]): TranscriptLine[] {
  const lines: TranscriptLine[] = []
  let cur: FinderWord[] = []
  let first = 0
  const flush = (last: number) => {
    if (cur.length === 0) return
    lines.push({
      index: lines.length,
      start_ms: cur[0].start_ms,
      end_ms: cur[cur.length - 1].end_ms,
      text: cur.map(w => w.word).join(' '),
      first_word: first, last_word: last,
    })
    cur = []
    first = last + 1
  }
  words.forEach((w, i) => {
    if (cur.length > 0 && w.start_ms - cur[cur.length - 1].end_ms > LINE_PAUSE_MS) flush(i - 1)
    cur.push(w)
    const next = words[i + 1]
    const breath = !next || next.start_ms - w.end_ms >= BREATH_MS
    if (SENTENCE_END.test(w.word.trim()) || (cur.length >= LINE_SOFT_WORDS && breath) || cur.length >= LINE_HARD_WORDS) flush(i)
  })
  flush(words.length - 1)
  return lines
}

/**
 * The lines from..to as phrases (user, 2026-10-31: a clip ended mid-sentence because the end of
 * the thought was in the middle of a line — Telugu lines run 25–45 words, to a breath, and a clip
 * could only end on a line's end). A phrase ends at a sentence end, at a pause of PHRASE_PAUSE_MS
 * once it has PHRASE_MIN_WORDS words, at PHRASE_MAX_WORDS (a run of words with no pause at all — a forced cut — as long as a line's own soft limit), and always at its line's end — so the
 * edges a clip had before are phrase edges too. Numbered from 0 in the order given.
 */
export function buildPhrases(words: FinderWord[], lines: TranscriptLine[], from: number, to: number): Phrase[] {
  const out: Phrase[] = []
  for (let li = from; li <= to; li++) {
    const l = lines[li]
    if (l.first_word === undefined || l.last_word === undefined) {
      out.push({ index: out.length, line: li, start_ms: l.start_ms, end_ms: l.end_ms, text: l.text })
      continue
    }
    let cur: FinderWord[] = []
    for (let k = l.first_word; k <= l.last_word; k++) {
      const w = words[k]
      cur.push(w)
      const next = k < l.last_word ? words[k + 1] : null
      const pause = next ? next.start_ms - w.end_ms : Infinity
      if (!next || (SENTENCE_END.test(w.word.trim()) && cur.length >= 2) || (cur.length >= PHRASE_MIN_WORDS && pause >= PHRASE_PAUSE_MS) || cur.length >= PHRASE_MAX_WORDS) {
        out.push({ index: out.length, line: li, start_ms: cur[0].start_ms, end_ms: cur[cur.length - 1].end_ms, text: cur.map(x => x.word).join(' ') })
        cur = []
      }
    }
  }
  return out
}

function mmss(ms: number) {
  const s = Math.floor(ms / 1000)
  return `${Math.floor(s / 60).toString().padStart(2, '0')}:${(s % 60).toString().padStart(2, '0')}`
}

// ── Windows ─────────────────────────────────────────────────────────────────────

/** ~10-minute windows of whole lines, each starting ~1 minute before the previous one ends */
export function makeWindows(lines: TranscriptLine[]): TranscriptLine[][] {
  if (lines.length === 0) return []
  const windows: TranscriptLine[][] = []
  let start = 0
  while (start < lines.length) {
    const windowStartMs = lines[start].start_ms
    let end = start
    while (end + 1 < lines.length && lines[end + 1].end_ms - windowStartMs <= WINDOW_MS) end++
    windows.push(lines.slice(start, end + 1))
    if (end === lines.length - 1) break
    // Next window starts at the first line that begins within the last minute of this one
    const overlapFromMs = lines[end].end_ms - WINDOW_OVERLAP_MS
    let next = end
    while (next > start + 1 && lines[next - 1].start_ms >= overlapFromMs) next--
    start = Math.max(next, start + 1)
  }
  return windows
}

// ── Prompts ─────────────────────────────────────────────────────────────────────

/**
 * Titles and other text for the user are always in Roman letters: the speaker's own language as
 * people type it (Tenglish, Hinglish, Tanglish…), never Telugu, Devanagari or other scripts.
 */
export const ROMAN_RULE = `Write the text in Roman (English) letters, in the speaker's own language the way people type it in chats: Telugu as Tenglish (e.g. "Ee cinema ki audience pichi ekkaru!"), Hindi as Hinglish (e.g. "Yeh scene dekh ke sab hil gaye!"), Tamil as Tanglish, and so on; keep English words as they are, and if the speaker talks in English write English. Never use Telugu, Devanagari, Tamil or any other non-Latin script.`

const LANGUAGE_RULES = `The transcript can be in Telugu, Hindi, Tamil or another Indian language, often mixed with English. Judge meaning, humour and emotion in the original language; do not translate it in your head first and lose the nuance.
${ROMAN_RULE.replace('the text', 'each "title"')} Write "summary" and "reason" in simple English. Keep all JSON keys in English.`

const SCORING_RULES = `Score every clip with five whole numbers from 0 to 10:
- "hook": do the first 3 seconds grab attention on their own?
- "emotion": how strongly will viewers feel something (laugh, surprise, anger, inspiration)?
- "story": is it self-contained and understandable without the rest of the video?
- "value": is it useful, insightful or funny enough to share?
- "clarity": is the speech clear and on one point, without rambling?
Be honest and spread the scores: an average moment is around 5, not 8.`

const LINE_RULES = `Each transcript line is written as "L<number> [mm:ss] text". Choose clips by line numbers only: "start_line" is the first line of the clip and "end_line" the last, from the lines shown. A clip should be 15–90 seconds long (use the mm:ss times to judge); a little longer is fine when the thought needs it. Start on the line that opens the thought (not mid-way through it) and end on the line where the thought, punchline or reaction is complete — ending too early is worse than a few extra seconds. Each clip must stand on its own: if it answers a question, start at the question and end after the whole answer; if it tells a story or makes a claim, end after the payoff or the point. Never end on a question, on "but…" or "because…", or in the middle of a list.`

const ITEM_SHAPE = `{"start_line":number,"end_line":number,"title":"catchy 3-7 word title","summary":"one sentence","reason":"why it works, max 15 words","hook":0-10,"emotion":0-10,"story":0-10,"value":0-10,"clarity":0-10}`

export function buildPrompt(mode: FinderMode, window: TranscriptLine[], durationMs: number, perWindow = BEST_PER_WINDOW, exclude: Array<[number, number]> = []) {
  const transcript = window.map(l => `L${l.index} [${mmss(l.start_ms)}] ${l.text}`).join('\n')
  const from = window[0].start_ms, to = window[window.length - 1].end_ms
  const taken = exclude.filter(([a, b]) => b > from && a < to)
  const part = `Video duration: ${mmss(durationMs)}. This part covers ${mmss(from)}–${mmss(to)}.`
    + (taken.length ? `\nAlready made into clips (pick other moments, not these): ${taken.map(([a, b]) => `${mmss(a)}–${mmss(b)}`).join(', ')}.` : '')

  if (mode.kind === 'best') {
    const system = `You are a viral short-clip detector for Reels and Shorts. From this part of a video transcript, pick up to ${perWindow} moments that would work best as standalone vertical clips. Fewer is fine if this part is weak.

${LINE_RULES}

${SCORING_RULES}

${LANGUAGE_RULES}

Return ONLY a JSON array, no other text. Each element: ${ITEM_SHAPE}`
    return { system, user: `${part}\n\nTranscript:\n${transcript}` }
  }

  const system = `You are a strict, skeptical video clip finder. Given part of a video transcript and a request describing what to look for, find moments that genuinely match.

Be conservative. Most videos do NOT contain what any given request is looking for — that is the normal case, not an edge case. Only include a moment if a viewer watching just that clip, with no explanation from you, would immediately agree it matches. Do not stretch, do not include a moment just because it is loosely related or you can construct a justification for it — if you find yourself explaining why something "counts", it doesn't.

If nothing in this part is a genuine, confident match, you MUST return an empty JSON array: []. Returning [] is the correct and expected answer for most (request, video) combinations — never force a result to avoid returning nothing.

${LINE_RULES} Do not stop at the line that merely contains the key moment — include the next 2-3 lines after it (the reaction, the response, the rest of the thought) in end_line.

${SCORING_RULES}

${LANGUAGE_RULES} In "summary", say why this moment matches the request.

Return ONLY a JSON array, no other text. Each element: ${ITEM_SHAPE}`
  return { system, user: `What to look for: ${mode.criteria}\n\n${part}\n\nTranscript:\n${transcript}` }
}

// ── Parsing ─────────────────────────────────────────────────────────────────────

const sub = (v: unknown) => {
  const n = Math.round(Number(v))
  return Number.isFinite(n) ? Math.min(10, Math.max(0, n)) : 0
}

/** Turns the model's answer into clips, dropping anything with bad or out-of-window line numbers */
export function parseCandidates(text: string, window: TranscriptLine[], lines: TranscriptLine[], windowIndex = 0): FoundClip[] {
  const match = text.match(/\[[\s\S]*\]/)
  if (!match) throw new Error(`No JSON array in model response: ${text.slice(0, 200)}`)
  const parsed: unknown = JSON.parse(match[0])
  if (!Array.isArray(parsed)) throw new Error('Model response is not an array')
  const first = window[0].index, last = window[window.length - 1].index
  const out: FoundClip[] = []
  for (const raw of parsed) {
    if (!raw || typeof raw !== 'object') continue
    const r = raw as Record<string, unknown>
    const startLine = Number(r.start_line), endLine = Number(r.end_line)
    if (!Number.isInteger(startLine) || !Number.isInteger(endLine)) continue
    if (startLine < first || endLine > last || endLine < startLine) continue
    const subscores: Subscores = {
      hook: sub(r.hook), emotion: sub(r.emotion), story: sub(r.story), value: sub(r.value), clarity: sub(r.clarity),
    }
    const total = subscores.hook + subscores.emotion + subscores.story + subscores.value + subscores.clarity
    out.push({
      start_line: startLine,
      end_line: endLine,
      start_ms: lines[startLine].start_ms,
      end_ms: lines[endLine].end_ms,
      title: typeof r.title === 'string' ? r.title.trim() : '',
      summary: typeof r.summary === 'string' ? r.summary.trim() : '',
      reason: typeof r.reason === 'string' ? r.reason.trim().split(/\s+/).slice(0, 15).join(' ') : '',
      score: Math.min(99, Math.round(total * 2)),
      subscores,
      window: windowIndex,
    })
  }
  return out
}

// ── Merge ───────────────────────────────────────────────────────────────────────

/** A line that ends a sentence or is followed by a pause: a clean place for a clip to stop */
const endsCleanly = (lines: TranscriptLine[], i: number) =>
  SENTENCE_END.test(lines[i].text.trim()) || !lines[i + 1] || lines[i + 1].start_ms - lines[i].end_ms >= 2 * BREATH_MS

/**
 * Holds a clip to 15 s – 2 min by moving its end to another line end; null if it can't be done.
 * The model is asked for 15–90 s, but a clip it makes longer to finish its thought keeps its end
 * (it used to be cut back to 90 s — the ending, the part that mattered, went). Only past 2 minutes
 * is it shortened, back to the latest line that ends a sentence or a breath.
 */
export function fitLength(clip: FoundClip, lines: TranscriptLine[]): FoundClip | null {
  let endLine = clip.end_line
  // Too short: extend line by line (never past the cap)
  while (lines[endLine].end_ms - clip.start_ms < MIN_CLIP_MS
    && endLine + 1 < lines.length
    && lines[endLine + 1].end_ms - clip.start_ms <= HARD_MAX_CLIP_MS) endLine++
  // Too long: back to the latest clean line end under the cap (else the latest line end under it)
  if (lines[endLine].end_ms - clip.start_ms > HARD_MAX_CLIP_MS) {
    let fit = endLine
    while (fit > clip.start_line && lines[fit].end_ms - clip.start_ms > HARD_MAX_CLIP_MS) fit--
    let clean = fit
    while (clean > clip.start_line && !endsCleanly(lines, clean) && lines[clean].end_ms - clip.start_ms > MIN_CLIP_MS) clean--
    endLine = endsCleanly(lines, clean) && lines[clean].end_ms - clip.start_ms >= MIN_CLIP_MS ? clean : fit
  }
  const length = lines[endLine].end_ms - clip.start_ms
  if (length < MIN_CLIP_MS || length > HARD_MAX_CLIP_MS) return null
  return { ...clip, end_line: endLine, end_ms: lines[endLine].end_ms }
}

/**
 * A second look at a picked clip's edges (user, 2026-10-31: clips stopped before the answer or
 * the payoff, or started without the setup). The model sees the clip marked, with REFINE_BEFORE
 * lines before and REFINE_AFTER after, as short phrases (buildPhrases), and may move the start
 * earlier for the setup it needs (or later, off the last topic) and the end later for the whole
 * answer / punchline / reaction (or earlier, off a new topic) — to the phrase, so a clip can end
 * where a sentence ends inside a long line. Asked at temperature 0 (the same clip got different
 * edges from one run to the next). Its answer is used only when it's inside the phrases shown and
 * the clip stays 15 s – 2 min; otherwise, or if the call fails, the clip is kept as it was.
 */
export async function refineEdges(clip: FoundClip, lines: TranscriptLine[], words: FinderWord[], ask: AskModel): Promise<{ clip: FoundClip; why: string | null }> {
  const first = Math.max(0, clip.start_line - REFINE_BEFORE)
  const last = Math.min(lines.length - 1, clip.end_line + REFINE_AFTER)
  const phrases = buildPhrases(words, lines, first, last)
  // The clip as it is: its first and last phrase (its edges are phrase edges: phrases never cross a line)
  const inClip = (ph: Phrase) => ph.start_ms >= clip.start_ms && ph.end_ms <= clip.end_ms
  const was = phrases.filter(inClip)
  if (!was.length) return { clip, why: null }
  const shown = phrases.map(ph => `${inClip(ph) ? '>' : ' '} P${ph.index} [${mmss(ph.start_ms)}] ${ph.text}`).join('\n')
  const system = `You check where a short clip cut from a video transcript should start and end, so it works on its own as a Reel or Short. The transcript is written in short phrases, "P<number> [mm:ss] text"; the clip's current phrases are marked with ">". A clip starts at the beginning of a phrase and ends at the end of one, so choose the exact phrase where the thought begins and the one where it is complete — not merely a nearby one.
- Start: on the line that opens this thought. Move it earlier only for what the clip needs to make sense — the question being answered, the setup of a story or joke, who or what "he", "she", "this" or "that" refers to — not for small talk or the end of the previous topic. Move it later if the first marked lines belong to the previous topic.
- End: after the thought is complete — the whole answer, the punchline, the payoff of the story, the conclusion, or the reaction to it. Never end on "but…" or "because…", or in the middle of a sentence or a list. Move it earlier if the last marked lines start a new topic.
- Questions: in interviews and panels someone often asks a question near the end. If the clip would end on a question (in any language, with or without a question mark — e.g. "enduku…?", "kya…?", "cheppandi", "tell me"), either go on to the end of its answer, or end before the question begins. Never end on the question itself.
- Keep the clip between 15 and 120 seconds (use the mm:ss times).
- If the edges are already right, return them unchanged.
${LANGUAGE_RULES.split('\n')[0]}
Return ONLY a JSON object, no other text: {"start_phrase":number,"end_phrase":number,"why":"what changed and why, max 12 words, in English"}`
  const user = `Clip: "${clip.title}" — ${clip.summary}\n\nTranscript:\n${shown}`
  try {
    const text = await ask(system, user, { temperature: 0 })
    const m = text.match(/\{[\s\S]*\}/)
    if (!m) return { clip, why: null }
    const r = JSON.parse(m[0]) as Record<string, unknown>
    const s = Number(r.start_phrase), e = Number(r.end_phrase)
    if (!Number.isInteger(s) || !Number.isInteger(e) || s < 0 || e >= phrases.length || e < s) return { clip, why: null }
    const a = phrases[s], b = phrases[e]
    if (a === was[0] && b === was[was.length - 1]) return { clip, why: null }
    const length = b.end_ms - a.start_ms
    if (length < MIN_CLIP_MS || length > HARD_MAX_CLIP_MS) return { clip, why: null }
    return {
      clip: { ...clip, start_line: a.line, end_line: b.line, start_ms: a.start_ms, end_ms: b.end_ms },
      why: typeof r.why === 'string' ? r.why.slice(0, 120) : '',
    }
  } catch {
    return { clip, why: null }
  }
}

/**
 * Room around the speech (user, 2026-10-31: clips began on the first syllable and stopped on the
 * last one): up to LEAD_MS before the first word and TAIL_MS after the last, taken only from the
 * silence there — never reaching the neighbouring words — and within the video. The neighbours are
 * the words right before and after the clip's own edges (an edge can be inside a line).
 */
export function withRoom(clip: FoundClip, words: FinderWord[], durationMs: number): FoundClip {
  const startMs = clip.start_ms, endMs = clip.end_ms
  const prev = words.filter(w => w.end_ms <= startMs).reduce((m, w) => Math.max(m, w.end_ms), -Infinity)
  const next = words.filter(w => w.start_ms >= endMs).reduce((m, w) => Math.min(m, w.start_ms), Infinity)
  const before = Number.isFinite(prev) ? startMs - prev : startMs
  const after = Number.isFinite(next) ? next - endMs : durationMs - endMs
  // (never past the 2-minute cap: what's left under it is shared, the lead first)
  const spare = Math.max(0, HARD_MAX_CLIP_MS - (endMs - startMs))
  const lead = Math.max(0, Math.min(LEAD_MS, before - EDGE_KEEP_MS, spare))
  const tail = Math.max(0, Math.min(TAIL_MS, after - EDGE_KEEP_MS, spare - lead))
  return {
    ...clip,
    start_ms: Math.max(0, startMs - lead),
    end_ms: Math.min(Math.max(durationMs, endMs), endMs + tail),
  }
}

/** Keeps the higher-scoring clip when two overlap by more than half of the shorter one */
export function dedupe(clips: FoundClip[]): FoundClip[] {
  const kept: FoundClip[] = []
  for (const c of [...clips].sort((a, b) => b.score - a.score)) {
    const clash = kept.some(k => {
      const overlap = Math.min(k.end_ms, c.end_ms) - Math.max(k.start_ms, c.start_ms)
      const shorter = Math.min(k.end_ms - k.start_ms, c.end_ms - c.start_ms)
      return overlap > 0.5 * shorter
    })
    if (!clash) kept.push(c)
  }
  return kept
}

async function mapLimit<T, R>(items: T[], limit: number, fn: (item: T, i: number) => Promise<R>): Promise<R[]> {
  const results = new Array<R>(items.length)
  let next = 0
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const i = next++
      results[i] = await fn(items[i], i)
    }
  })
  await Promise.all(workers)
  return results
}

/**
 * Top `limit` clips by score, taking at most `perWindow` from any one window first so the
 * picks come from across the video; leftover places go to the best of the rest.
 */
export function pickSpread(clips: FoundClip[], limit: number, windowCount: number): FoundClip[] {
  const cap = Math.ceil(limit / Math.max(1, windowCount)) + 1
  const taken = new Map<number, number>()
  const picked: FoundClip[] = []
  const rest: FoundClip[] = []
  for (const c of clips) {
    const n = taken.get(c.window) ?? 0
    if (picked.length < limit && n < cap) { picked.push(c); taken.set(c.window, n + 1) } else rest.push(c)
  }
  for (const c of rest) if (picked.length < limit) picked.push(c)
  return picked.sort((a, b) => b.score - a.score)
}

/**
 * Finds clips over the whole transcript. Windows that fail are logged and skipped; throws only
 * when every window fails, so the caller can fall back.
 */
export async function findClips(opts: {
  words: FinderWord[]
  durationMs: number
  mode: FinderMode
  ask: AskModel
  log?: (msg: string) => void
  /** Best moments: how many clips to return (default 8) */
  limit?: number
  /** Best moments: candidates to ask for per window (default 4) */
  perWindow?: number
  /** Source-time ranges already made into clips: new clips must not repeat them */
  exclude?: Array<[number, number]>
  /** Best moments: spread the picks across the video instead of pure top-by-score */
  spread?: boolean
  /** Check each picked clip's start and end with the model (refineEdges; default on) */
  refine?: boolean
}): Promise<FoundClip[]> {
  const log = opts.log ?? (() => {})
  const lines = buildLines(opts.words)
  const windows = makeWindows(lines)
  if (windows.length === 0) return []

  let failures = 0
  const perWindow = await mapLimit(windows, MAX_PARALLEL, async (window, i) => {
    try {
      const { system, user } = buildPrompt(opts.mode, window, opts.durationMs, opts.perWindow, opts.exclude)
      return parseCandidates(await opts.ask(system, user), window, lines, i)
    } catch (e) {
      failures++
      log(`window ${i + 1}/${windows.length} failed: ${e instanceof Error ? e.message : String(e)}`)
      return []
    }
  })
  if (failures === windows.length) throw new Error(`All ${windows.length} clip-finder windows failed`)

  // Anything overlapping an existing clip by more than half (of the shorter one) is a repeat
  const repeats = (c: FoundClip) => (opts.exclude ?? []).some(([a, b]) => {
    const overlap = Math.min(b, c.end_ms) - Math.max(a, c.start_ms)
    return overlap > 0.5 * Math.min(b - a, c.end_ms - c.start_ms)
  })
  const fitted = perWindow.flat()
    .map(c => fitLength(c, lines))
    .filter((c): c is FoundClip => c !== null && !repeats(c))
  const merged = dedupe(fitted) // sorted by score, highest first
  const limit = opts.limit ?? BEST_TOP_N
  const search = opts.mode.kind === 'search'
  // The picks, then two spares (checked too): a pick whose new edges overlap a better one is
  // dropped, and a spare takes its place
  const primary = search ? merged.filter(c => c.score >= SEARCH_MIN_SCORE)
    : opts.spread ? pickSpread(merged, limit, windows.length) : merged.slice(0, limit)
  const picked = search ? primary : [...primary, ...merged.filter(c => !primary.includes(c)).slice(0, 2)]
  const toCheck = opts.refine === false ? [] : picked.slice(0, search ? REFINE_MAX : picked.length)
  const checked = await mapLimit(toCheck, MAX_PARALLEL, async c => {
    const r = await refineEdges(c, lines, opts.words, opts.ask)
    if (r.why !== null) log(`edges ${mmss(c.start_ms)}–${mmss(c.end_ms)} → ${mmss(r.clip.start_ms)}–${mmss(r.clip.end_ms)}: ${r.why}`)
    return r.clip
  })
  const all = [...checked, ...picked.slice(toCheck.length)]
  const kept = new Set(dedupe(all.filter(c => !repeats(c))))
  const out = all.filter(c => kept.has(c)).slice(0, search ? undefined : limit)
  return out.map(c => withRoom(c, opts.words, opts.durationMs)).sort((a, b) => b.score - a.score)
}
