/**
 * Parts of a clip the user removed in the editor ("delete this part of the video"), and how
 * times move between the SOURCE video and the editor's TIMELINE once they are gone.
 *
 * Copied as-is between the repos (chai-cut-backend/src/lib/trims.ts, chai-cut-frontend
 * src/lib/trims.ts). Keep the two identical.
 *
 * What is saved: clips.trim_ranges — [[start, end], …] in milliseconds of the SOURCE video.
 * Everything else a clip saves (sections, text, photos, music…) keeps its usual times: clip time
 * as if nothing were removed. The editor closes the gaps on screen (timeline time), and the export
 * cuts them out the way "Remove pauses" does (jobs/render.ts condenseClip).
 */

import { keptRanges, type Range } from './cuts.js'

/** The longest a clip can play once its start or end is dragged (most short-form apps allow 5 minutes) */
export const MAX_CLIP_MS = 5 * 60 * 1000
/** The shortest a clip can be made */
export const MIN_CLIP_MS = 1000

/** A kept sliver shorter than this between two removed parts goes with them */
const MIN_KEEP_MS = 100

/** Saved removed parts, made safe to use: inside the clip, in order, merged. [] when unusable. */
export function cleanTrims(raw: unknown, clipStart: number, clipEnd: number): Range[] {
  if (!Array.isArray(raw)) return []
  const ranges = raw
    .filter((r): r is [number, number] => Array.isArray(r) && r.length === 2 && Number.isFinite(Number(r[0])) && Number.isFinite(Number(r[1])))
    .map(([a, b]): Range => [Math.max(clipStart, Math.round(Number(a))), Math.min(clipEnd, Math.round(Number(b)))])
    .filter(([a, b]) => b > a)
    .sort((x, y) => x[0] - y[0])
  const merged: Range[] = []
  for (const r of ranges) {
    const prev = merged[merged.length - 1]
    if (prev && r[0] - prev[1] < MIN_KEEP_MS) prev[1] = Math.max(prev[1], r[1])
    else merged.push([r[0], r[1]])
  }
  // A sliver left at either end of the clip goes too
  if (merged.length && merged[0][0] - clipStart < MIN_KEEP_MS) merged[0][0] = clipStart
  if (merged.length && clipEnd - merged[merged.length - 1][1] < MIN_KEEP_MS) merged[merged.length - 1][1] = clipEnd
  // Something must stay
  if (merged.reduce((s, [a, b]) => s + (b - a), 0) >= clipEnd - clipStart) return []
  return merged
}

export interface TrimMap {
  /** The parts of the source video that stay, in order (source ms) */
  kept: Range[]
  /** Timeline time where each kept part starts */
  starts: number[]
  /** How long the clip plays */
  lengthMs: number
  /** Source ms → timeline ms (0 = the clip's first frame). A time inside a removed part lands on the join. */
  toTimeline: (sourceMs: number) => number
  /** Timeline ms → source ms. At a join: the first frame AFTER it (where something starting there starts). */
  toSource: (t: number) => number
  /** Timeline ms → source ms. At a join: the last moment BEFORE it (where something ending there ends). */
  toSourceEnd: (t: number) => number
}

/** The time map of a clip [clipStart, clipEnd) with `trims` (already clean) removed */
export function trimMap(trims: Range[], clipStart: number, clipEnd: number): TrimMap {
  const kept = keptRanges(trims, clipStart, clipEnd)
  const starts: number[] = []
  let total = 0
  for (const [a, b] of kept) { starts.push(total); total += b - a }
  const last = kept.length - 1
  return {
    kept, starts, lengthMs: total,
    toTimeline: (v: number) => {
      // Outside the clip times keep running, so nothing that sticks out is squashed
      if (v < clipStart) return v - clipStart
      if (v > clipEnd) return total + (v - clipEnd)
      for (let i = 0; i < kept.length; i++) {
        const [a, b] = kept[i]
        if (v < a) return starts[i]
        if (v <= b) return starts[i] + (v - a)
      }
      return total
    },
    toSource: (t: number) => {
      if (t < 0) return clipStart + t
      if (t >= total) return clipEnd + (t - total)
      for (let i = 0; i < kept.length; i++) {
        if (t < starts[i] + (kept[i][1] - kept[i][0])) return kept[i][0] + (t - starts[i])
      }
      return clipEnd
    },
    toSourceEnd: (t: number) => {
      if (t <= 0) return clipStart + t
      if (t > total) return clipEnd + (t - total)
      for (let i = 0; i < kept.length; i++) {
        if (t <= starts[i] + (kept[i][1] - kept[i][0])) return kept[i][0] + (t - starts[i])
      }
      return last >= 0 ? kept[last][1] : clipEnd
    },
  }
}

/**
 * Transcript words (source ms) where they are heard in the clip as it plays: words inside a
 * removed part are left out, and times become clipStart + timeline ms — so everything that reads
 * "a word's time minus the clip's start = clip time" needs no change. Words outside the clip keep
 * their distance from it.
 */
export function wordsOnTimeline<W extends { start_ms: number; end_ms: number }>(words: W[], map: TrimMap, clipStart: number, clipEnd: number): W[] {
  return words
    .filter(w => {
      const mid = (w.start_ms + w.end_ms) / 2
      return mid < clipStart || mid >= clipEnd || map.kept.some(([a, b]) => mid >= a && mid < b)
    })
    .map(w => ({ ...w, start_ms: clipStart + map.toTimeline(w.start_ms), end_ms: clipStart + map.toTimeline(w.end_ms) }))
}

/**
 * The removed parts after also removing timeline [a, b) — the stretch the user picked, in the
 * timeline as it is now (`map`). Parts already removed inside it are swallowed.
 */
export function addTrim(trims: Range[], map: TrimMap, a: number, b: number, clipStart: number, clipEnd: number): Range[] {
  if (!(b > a)) return trims
  return cleanTrims([...trims, [map.toSource(Math.max(0, a)), map.toSourceEnd(Math.min(map.lengthMs, b))]], clipStart, clipEnd)
}
