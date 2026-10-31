/**
 * Speech the transcription skipped (user, 2026-10-31: Make my clips cut topics wrong; the transcript of
 * a 92-min Telugu interview had 7% of the video in gaps of 5 s+ that were all speech, and captions
 * there were missing too). Gemini's transcription skips stretches of speech inside a piece — even in
 * the middle of one speaker's turn — at any piece length and with any setting (tested 2026-10-31 on
 * one 5-minute piece: 5 min / 2.5 min / 1 min pieces, speaker labels off, Telugu + English: 112–167 s
 * of it left out every time). Read again ON ITS OWN, each gap came back (12 of 12); stitched into one
 * request with silence between, only 4 of 13 did. So each gap is read again as its own short stretch
 * of the real audio, gaps close together in one stretch.
 *
 * This file only plans it (which gaps, which stretches, which words to keep); jobs/transcribe.ts
 * reads the audio.
 */

/** A gap without words this long or longer is read again (seconds) */
export const REREAD_MIN_SEC = 3
/** …when it has at least this much sound in it (the rest is a pause) */
export const REREAD_MIN_SOUND_SEC = 1
/** A stretch read again is at most this long: short ones are read whole */
export const REREAD_WINDOW_SEC = 40
/** Gaps this close together are read in one stretch (the speech between them comes along) */
export const REREAD_JOIN_SEC = 6
/** Room either side of a gap, so its first and last words are whole */
export const REREAD_PAD_SEC = 0.3
/** At most this many stretches per piece (the ones with the most gap in them) */
export const REREAD_MAX = 16
/**
 * Gemini skips inside a stretch read again too (an 11-minute interview: the first round brought back
 * 516 words, a second 260 more), so what is still empty is read again — at most this many rounds,
 * stopping when a round brings back fewer than REREAD_ROUND_MIN_WORDS
 */
export const REREAD_ROUNDS = 3
export const REREAD_ROUND_MIN_WORDS = 5

type Timed = { start: number; end: number }
export type Gap = [number, number]
export interface RereadWindow { from: number; to: number; gaps: Gap[] }

/** The gaps of REREAD_MIN_SEC or more between the words of a piece `pieceSec` long (piece-relative seconds) */
export function skippedGaps(words: readonly Timed[], pieceSec: number, minSec = REREAD_MIN_SEC): Gap[] {
  const gaps: Gap[] = []
  let cursor = 0
  for (const w of [...words].sort((a, b) => a.start - b.start)) {
    if (w.start - cursor >= minSec) gaps.push([cursor, w.start])
    cursor = Math.max(cursor, w.end)
  }
  if (pieceSec - cursor >= minSec) gaps.push([cursor, pieceSec])
  return gaps
}

/**
 * The stretches to read again: a gap longer than REREAD_WINDOW_SEC is read in parts; gaps no more than
 * REREAD_JOIN_SEC apart share a stretch while it stays within REREAD_WINDOW_SEC. Each stretch has
 * REREAD_PAD_SEC either side, within the piece. More than REREAD_MAX: the ones with the most gap in
 * them, in time order.
 */
export function rereadWindows(gaps: readonly Gap[], pieceSec: number): RereadWindow[] {
  // Long gaps in parts
  const parts: Gap[] = []
  for (const [a, b] of [...gaps].sort((p, q) => p[0] - q[0])) {
    const n = Math.ceil((b - a) / REREAD_WINDOW_SEC)
    for (let k = 0; k < n; k++) parts.push([a + ((b - a) * k) / n, a + ((b - a) * (k + 1)) / n])
  }
  const out: RereadWindow[] = []
  for (const g of parts) {
    const last = out[out.length - 1]
    const lastEnd = last ? last.gaps[last.gaps.length - 1][1] : -Infinity
    if (last && g[0] - lastEnd <= REREAD_JOIN_SEC && g[1] - last.gaps[0][0] <= REREAD_WINDOW_SEC) last.gaps.push(g)
    else out.push({ from: 0, to: 0, gaps: [g] })
  }
  for (const w of out) {
    w.from = Math.max(0, w.gaps[0][0] - REREAD_PAD_SEC)
    w.to = Math.min(pieceSec, w.gaps[w.gaps.length - 1][1] + REREAD_PAD_SEC)
  }
  if (out.length <= REREAD_MAX) return out
  const size = (w: RereadWindow) => w.gaps.reduce((s, [a, b]) => s + b - a, 0)
  const keep = new Set([...out].sort((p, q) => size(q) - size(p)).slice(0, REREAD_MAX))
  return out.filter(w => keep.has(w))
}

/**
 * The words a stretch read again brings back: those starting inside one of its gaps (the speech
 * around the gaps was already read), placed at piece time (`w.start` is from the stretch's start)
 */
export function wordsInGaps<W extends Timed>(window: RereadWindow, words: readonly W[]): W[] {
  const out: W[] = []
  for (const w of words) {
    const start = window.from + w.start, end = window.from + w.end
    if (!window.gaps.some(([a, b]) => start >= a - 0.15 && start < b)) continue
    out.push({ ...w, start, end: Math.max(end, start + 0.08) })
  }
  return out
}
