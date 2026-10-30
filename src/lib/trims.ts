/**
 * Parts of a clip the user removed in the editor ("delete this part of the video"), and how
 * times move between the SOURCE video and the editor's TIMELINE once they are gone.
 *
 * The worker only needs cleanTrims; the editor's copy (chai-cut-frontend src/lib/trims.ts) also
 * maps times between the source and the timeline. Keep cleanTrims identical in the two.
 *
 * What is saved: clips.trim_ranges — [[start, end], …] in milliseconds of the SOURCE video.
 * Everything else a clip saves (sections, text, photos, music…) keeps its usual times: clip time
 * as if nothing were removed. The editor closes the gaps on screen (timeline time), and the export
 * cuts them out the way "Remove pauses" does (jobs/render.ts condenseClip).
 */

import type { Range } from './cuts.js'

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
