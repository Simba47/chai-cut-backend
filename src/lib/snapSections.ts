/**
 * Sub-edit: where a section (a stretch with its own layout) ends and the next begins should be where
 * the camera cuts. A boundary that lies a little before or after a cut leaves the old layout on the
 * new picture (or the new layout on the old one) for those seconds. This moves such a boundary onto
 * the nearest camera cut.
 *
 * All times are milliseconds in clip time (0 = the clip's first frame).
 */

/** A boundary only moves to a cut this close; further away it is left where it is */
export const SNAP_MAX_MS = 4000
/** A section never gets shorter than this by a move */
export const SNAP_MIN_SECTION_MS = 500
/** A boundary this close to a cut is already on it */
const ON_CUT_MS = 150
/** Two sections that follow each other: the first's end and the next's start differ by at most this */
const JOINED_MS = 50

export interface SnapSection {
  id: string
  start_ms: number
  end_ms: number
  /** false: left alone (locked, horizontal, or showing a photo or another video) */
  movable: boolean
}

/**
 * The sections whose start or end moved, with their new times. Sections must be in time order.
 * `cuts`: camera cuts in clip time.
 */
export function snapSectionBoundaries(sections: readonly SnapSection[], cuts: readonly number[]): Array<{ id: string; start_ms: number; end_ms: number }> {
  const secs = sections.map(s => ({ ...s }))
  const used = new Set<number>()
  for (let i = 0; i < secs.length - 1; i++) {
    const a = secs[i], b = secs[i + 1]
    if (!a.movable || !b.movable) continue
    if (Math.abs(a.end_ms - b.start_ms) > JOINED_MS) continue
    const boundary = a.end_ms
    if (cuts.some(c => Math.abs(c - boundary) <= ON_CUT_MS)) continue
    const lo = a.start_ms + SNAP_MIN_SECTION_MS, hi = b.end_ms - SNAP_MIN_SECTION_MS
    const cut = cuts
      .filter(c => !used.has(c) && Math.abs(c - boundary) <= SNAP_MAX_MS && c >= lo && c <= hi)
      .sort((x, y) => Math.abs(x - boundary) - Math.abs(y - boundary))[0]
    if (cut === undefined) continue
    used.add(cut)
    a.end_ms = cut
    b.start_ms = cut
  }
  return secs
    .filter((s, i) => s.start_ms !== sections[i].start_ms || s.end_ms !== sections[i].end_ms)
    .map(({ id, start_ms, end_ms }) => ({ id, start_ms, end_ms }))
}
