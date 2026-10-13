/**
 * Runaway readings: now and then the speech-to-text model gets stuck and writes the same phrase
 * over and over (seen in production: 8,132 words for 5 minutes of audio, about 27 a second).
 * Those words then fill the transcript, and clip finding reads the loop instead of the video.
 *
 * A chunk's reading is suspect when a phrase repeats many times in a row, or when it has more
 * words than anyone can say in that time. The caller reads a suspect chunk once more and, if it
 * is still wrong, keeps the reading with each loop cut back to its first copies.
 *
 * No imports: the checks are plain functions over the words.
 */

/** A word as the reader returns it: seconds from the start of the chunk */
export interface GuardWord { word: string; start: number; end: number }

/** Faster than this over a whole chunk is not speech (fast Telugu or Hindi is about 3.5 words a second) */
export const MAX_WORDS_PER_S = 4.5
/** Phrases up to this many words are checked for repeating */
const LOOP_MAX_WORDS = 8
/** Copies in a row that make a loop: a phrase of 2+ words, and a single word ("no no no…" is real speech up to a point) */
const LOOP_MIN_COPIES = 8
const LOOP_MIN_COPIES_ONE_WORD = 12
/** …and at least this many words in all: a chant or chorus sung over and over is real, a stuck reading runs to hundreds */
const LOOP_MIN_TOTAL_WORDS = 40
/** Copies of a loop that are kept: the speaker may really have said it twice */
const LOOP_KEEP_COPIES = 2

export interface Loop { start: number; words: number; copies: number }

const norm = (w: string) => w.toLowerCase().replace(/[^\p{L}\p{N}]/gu, '')

/** The phrases that repeat many times in a row: where each starts, its length in words and how many copies */
export function findLoops(words: readonly { word: string }[]): Loop[] {
  const t = words.map(w => norm(w.word))
  const same = (a: number, b: number, n: number) => {
    for (let k = 0; k < n; k++) if (t[a + k] !== t[b + k]) return false
    return true
  }
  const loops: Loop[] = []
  let i = 0
  while (i < t.length) {
    let best: Loop | null = null
    for (let n = 1; n <= LOOP_MAX_WORDS && i + 2 * n <= t.length; n++) {
      // A run of punctuation only is not a phrase
      if (t.slice(i, i + n).every(x => !x)) continue
      let copies = 1
      while (i + (copies + 1) * n <= t.length && same(i, i + copies * n, n)) copies++
      const need = n === 1 ? LOOP_MIN_COPIES_ONE_WORD : LOOP_MIN_COPIES
      // The phrase length that explains the most words wins (a looping 2-word phrase is not a 1-word loop)
      if (copies >= need && copies * n >= LOOP_MIN_TOTAL_WORDS && (!best || copies * n > best.copies * best.words)) best = { start: i, words: n, copies }
    }
    if (best) { loops.push(best); i += best.copies * best.words } else i++
  }
  return loops
}

/** Why a chunk's reading looks wrong, or null when it looks like speech. `seconds`: the audio's length */
export function runawayReason(words: readonly GuardWord[], seconds: number): string | null {
  const loops = findLoops(words)
  if (loops.length) {
    const worst = loops.reduce((a, b) => (b.copies * b.words > a.copies * a.words ? b : a))
    return `a ${worst.words}-word phrase repeated ${worst.copies} times${loops.length > 1 ? ` (${loops.length} loops)` : ''}`
  }
  const rate = words.length / Math.max(1, seconds)
  if (rate > MAX_WORDS_PER_S) return `${words.length} words in ${Math.round(seconds)} s (${rate.toFixed(1)} a second)`
  return null
}

/**
 * The reading with every loop cut back to its first copies, and without words timed outside the
 * audio (a stuck reading also loses its timing). Words keep their order.
 */
export function trimRunaway<W extends GuardWord>(words: readonly W[], seconds: number): W[] {
  const drop = new Set<number>()
  for (const loop of findLoops(words)) {
    for (let k = loop.start + LOOP_KEEP_COPIES * loop.words; k < loop.start + loop.copies * loop.words; k++) drop.add(k)
  }
  return words.filter((w, i) => !drop.has(i) && Number.isFinite(w.start) && w.start >= -0.5 && w.start <= seconds + 1)
}
