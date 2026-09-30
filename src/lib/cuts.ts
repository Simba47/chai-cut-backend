/**
 * "Remove pauses and filler words": which parts of a clip to cut, and how times move once they
 * are gone.
 *
 * Copied as-is between the repos (chai-cut-backend/src/lib/cuts.ts, chai-cut-frontend
 * src/lib/cuts.ts): the editor shows "Removes about X s" with it, the worker cuts with it at
 * export. Keep the two identical.
 *
 * All times are milliseconds in the SOURCE video.
 */

export type CutWord = { word: string; word_roman?: string | null; start_ms: number; end_ms: number }
export type Range = [number, number]

// ── Filler words (edit here) ────────────────────────────────────────────────────
// Matched on the word itself and on its Roman spelling, lower case, punctuation stripped.

/** Hesitation sounds: always cut */
const HESITATIONS = new Set([
  'um', 'umm', 'uh', 'uhh', 'uhm', 'hmm', 'hm', 'erm', 'er', 'ah',
  'उम', 'उम्म', 'अं', 'हम्म', 'ఉమ్', 'అమ్', 'హ్మ్',
])
/** Cut only when a pause sits next to them: otherwise they are part of the sentence */
const STANDALONE = new Set([
  // Hindi / Urdu
  'matlab', 'मतलब', 'woh', 'वो',
  // Telugu
  'ante', 'అంటే', 'adi', 'అది', 'aa', 'ఆ',
])
/** Two-word fillers, cut only next to a pause */
const PHRASES: string[][] = [['you', 'know']]
// Any word said twice in a row loses the first one ("the the", "like like", "haan haan",
// "accha accha", "emo emo"); like / haan / accha / emo are never cut on their own.

// ── Timing rules ────────────────────────────────────────────────────────────────
const LONG_PAUSE_MS = 600      // a gap longer than this is shortened…
const KEEP_EDGE_MS = 75        // …to this much after the last word and before the next (150 ms)
const ALONE_GAP_MS = 250       // a pause this long next to a word makes it "stand alone"
const REPEAT_GAP_MS = 500      // "the the": the same word again within this long
const MIN_CUT_MS = 120         // shorter cuts are not worth a jump
const MERGE_GAP_MS = 100       // cuts closer than this become one

const norm = (s?: string | null) => (s ?? '').toLowerCase().replace(/[\p{P}\p{S}]/gu, '').trim()
const forms = (w: CutWord) => [norm(w.word), norm(w.word_roman)].filter(Boolean)
const isIn = (set: Set<string>, w: CutWord) => forms(w).some(f => set.has(f))
const same = (a: CutWord, b: CutWord) => forms(a).some(f => forms(b).includes(f))

/** Source-time ranges to remove from the clip [clipStart, clipEnd), sorted and non-overlapping */
export function computeCutRanges(allWords: CutWord[], clipStart: number, clipEnd: number): Range[] {
  const words = allWords
    .filter(w => w.start_ms >= clipStart && w.end_ms <= clipEnd && w.end_ms >= w.start_ms)
    .sort((a, b) => a.start_ms - b.start_ms)
  if (words.length === 0) return []
  const gapBefore = (i: number) => (i > 0 ? words[i].start_ms - words[i - 1].end_ms : Infinity)
  const gapAfter = (i: number) => (i < words.length - 1 ? words[i + 1].start_ms - words[i].end_ms : Infinity)
  const alone = (i: number, j = i) => gapBefore(i) >= ALONE_GAP_MS || gapAfter(j) >= ALONE_GAP_MS

  const removed = new Array<boolean>(words.length).fill(false)
  for (let i = 0; i < words.length; i++) {
    const w = words[i]
    const next = words[i + 1]
    const repeated = !!next && same(w, next) && next.start_ms - w.end_ms <= REPEAT_GAP_MS
    if (isIn(HESITATIONS, w)) removed[i] = true
    else if (isIn(STANDALONE, w) && alone(i)) removed[i] = true
    else if (repeated) removed[i] = true
    for (const phrase of PHRASES) {
      const n = phrase.length
      if (i + n <= words.length && phrase.every((p, k) => forms(words[i + k]).includes(p)) && alone(i, i + n - 1)) {
        for (let k = 0; k < n; k++) removed[i + k] = true
      }
    }
  }
  // Never leave the clip with no words at all
  if (removed.every(Boolean)) return []

  const cuts: Range[] = []
  const kept = words.map((w, i) => ({ w, i })).filter(x => !removed[x.i])
  // Removed words before the first kept word / after the last one
  if (kept[0].i > 0) cuts.push([words[0].start_ms, kept[0].w.start_ms - KEEP_EDGE_MS])
  const last = kept[kept.length - 1]
  if (last.i < words.length - 1) cuts.push([last.w.end_ms + KEEP_EDGE_MS, words[words.length - 1].end_ms])
  // Between kept words: removed words in between, or a long pause, leave 150 ms of air (half of
  // a tighter gap next to a removed word, so a quick "the the" still loses its first word)
  for (let k = 1; k < kept.length; k++) {
    const a = kept[k - 1].w, b = kept[k].w
    const firstGone = words[kept[k - 1].i + 1], lastGone = words[kept[k].i - 1]
    if (kept[k].i - kept[k - 1].i > 1) {
      cuts.push([
        a.end_ms + Math.min(KEEP_EDGE_MS, (firstGone.start_ms - a.end_ms) / 2),
        b.start_ms - Math.min(KEEP_EDGE_MS, (b.start_ms - lastGone.end_ms) / 2),
      ])
    } else if (b.start_ms - a.end_ms > LONG_PAUSE_MS) {
      cuts.push([a.end_ms + KEEP_EDGE_MS, b.start_ms - KEEP_EDGE_MS])
    }
  }

  return cleanRanges(cuts, clipStart, clipEnd)
}

/** Clamped to the clip, merged when close, and without cuts too short to matter */
export function cleanRanges(ranges: Range[], clipStart: number, clipEnd: number): Range[] {
  const sorted = ranges
    .map(([a, b]): Range => [Math.max(clipStart, Math.round(a)), Math.min(clipEnd, Math.round(b))])
    .filter(([a, b]) => b > a)
    .sort((x, y) => x[0] - y[0])
  const merged: Range[] = []
  for (const r of sorted) {
    const prev = merged[merged.length - 1]
    if (prev && r[0] - prev[1] < MERGE_GAP_MS) prev[1] = Math.max(prev[1], r[1])
    else merged.push([r[0], r[1]])
  }
  return merged.filter(([a, b]) => b - a >= MIN_CUT_MS)
}

/** How much the cuts take out, in ms */
export const removedMs = (cuts: Range[]) => cuts.reduce((s, [a, b]) => s + (b - a), 0)

/** The parts of [clipStart, clipEnd) that stay, in order */
export function keptRanges(cuts: Range[], clipStart: number, clipEnd: number): Range[] {
  const kept: Range[] = []
  let cursor = clipStart
  for (const [a, b] of cuts) {
    if (a > cursor) kept.push([cursor, a])
    cursor = Math.max(cursor, b)
  }
  if (clipEnd > cursor) kept.push([cursor, clipEnd])
  return kept
}

/**
 * Source time → time in the shortened clip (0 = its first frame). A time inside a cut lands on
 * the join.
 */
export function makeTimeMap(kept: Range[]): (sourceMs: number) => number {
  const starts: number[] = []
  let total = 0
  for (const [a, b] of kept) { starts.push(total); total += b - a }
  return (t: number) => {
    for (let i = 0; i < kept.length; i++) {
      const [a, b] = kept[i]
      if (t < a) return starts[i]
      if (t <= b) return starts[i] + (t - a)
    }
    return total
  }
}
