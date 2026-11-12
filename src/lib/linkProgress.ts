// A video imported from a link (Google Drive, Dropbox, yt-dlp): where each step sits on the dashboard
// card's bar (videos.download_progress). The website's dashboard (app/dashboard/page.tsx) names the
// steps by these same numbers — keep the two in step.
//
// (user, 2026-11-12) The bar used to stop at 47% for the whole save to storage — the longest part for
// a big file — then jump to ready: it now moves through the download AND the save, in order.

export const LINK_BAR = {
  /** The job has started */
  start: 5,
  /** Downloaded: 5 → 50 while the file comes in */
  fetched: 50,
  /** Checked that it's a video we can read (or, from yt-dlp, made into an MP4) */
  checking: 51,
  /** Saving to storage: 52 → 95 while the parts go up */
  saveFrom: 52,
  saved: 95,
  /** Name and length read; ready right after (the job then writes 100 with status 'ready') */
  almost: 96,
} as const

const between = (from: number, to: number, done: number, total: number) =>
  total > 0 ? Math.round(from + Math.min(1, Math.max(0, done / total)) * (to - from)) : from

/** The bar while the file downloads (`total` 0 = size unknown: it stays at the start) */
export const fetchPct = (done: number, total: number) => between(LINK_BAR.start, LINK_BAR.fetched, done, total)

/** The bar while the file is saved to storage */
export const savePct = (done: number, total: number) => between(LINK_BAR.saveFrom, LINK_BAR.saved, done, total)

/**
 * Writes the bar's value one write at a time, never lower than before: a value asked for while a write
 * is running waits for it, and only the latest of those is written. (Writes used to be sent all at
 * once, so a slow earlier one could land after a later one and move the bar back.) `done()` waits for
 * the last write — before the job marks the video ready, so no late write comes after that.
 */
export function progressFeed(write: (pct: number) => Promise<unknown>) {
  let asked = -1
  let written = -1
  let running: Promise<void> | null = null
  const pump = async () => {
    while (asked > written) {
      const pct = asked
      written = pct
      try { await write(pct) } catch { /* only a progress note: the next value tries again */ }
    }
    running = null
  }
  return {
    set(pct: number) {
      if (pct <= asked) return
      asked = pct
      if (!running) running = pump()
    },
    async done() { while (running) await running },
  }
}
