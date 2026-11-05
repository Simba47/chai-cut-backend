/**
 * The picture inside a video's black bars (user, 2026-10-31: on letterboxed videos — a YouTube upload
 * with bars top and bottom — views kept head room above the faces and reached into the bar, a black
 * strip at the top of the view, in every format). A video's bars are found once (the worker: jobs/
 * picture.ts, saved as videos.picture_box) and every framing keeps inside the picture:
 *   - formats decided from the faces (the Rulebook) are decided in the picture's own space — the faces
 *     moved into it (toPicture), the picture's own shape — and the views moved back (fromPicture);
 *   - any other view (the old framing, Follow, a format picked by hand) is kept inside it (insidePicture).
 *
 * Copied as-is between the repos (chai-cut-backend src/lib/pictureBox.ts, chai-cut-frontend
 * src/modules/editor/pictureBox.ts). Keep the two identical. No imports.
 */

/** A box in fractions of the video frame */
export interface PictureBox { x: number; y: number; w: number; h: number }
/** The whole frame: no bars */
export const WHOLE_FRAME: PictureBox = { x: 0, y: 0, w: 1, h: 1 }
/** Bars thinner than this (a share of the frame) aren't bars: a dark edge, a frame's border */
export const MIN_BAR = 0.02

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))

/** A picture box, or the whole frame when it is missing, broken or has no bars to speak of */
export function pictureOrWhole(p: Partial<PictureBox> | null | undefined): PictureBox {
  const ok = p && [p.x, p.y, p.w, p.h].every(v => typeof v === 'number' && isFinite(v))
  if (!ok) return WHOLE_FRAME
  const x = clamp(p.x!, 0, 1), y = clamp(p.y!, 0, 1), w = clamp(p.w!, 0, 1 - x), h = clamp(p.h!, 0, 1 - y)
  // (a picture of under a third of the frame either way: a misreading, not bars)
  if (w < 0.33 || h < 0.33) return WHOLE_FRAME
  const bars = x >= MIN_BAR || y >= MIN_BAR || 1 - (x + w) >= MIN_BAR || 1 - (y + h) >= MIN_BAR
  return bars ? { x, y, w, h } : WHOLE_FRAME
}

export const hasBars = (p: PictureBox) => p.x > 0 || p.y > 0 || p.w < 1 || p.h < 1

/** A box from the frame into the picture's own space (0–1 across the picture) */
export function toPicture<B extends PictureBox>(b: B, p: PictureBox): B {
  return { ...b, x: (b.x - p.x) / p.w, y: (b.y - p.y) / p.h, w: b.w / p.w, h: b.h / p.h }
}
/** And back from the picture's space into the frame */
export function fromPicture<B extends PictureBox>(b: B, p: PictureBox): B {
  return { ...b, x: p.x + b.x * p.w, y: p.y + b.y * p.h, w: b.w * p.w, h: b.h * p.h }
}
/** The picture's shape (width / height in pixels), for a video of shape `videoAR` */
export const pictureAR = (videoAR: number, p: PictureBox) => (videoAR * p.w) / p.h

/**
 * A view kept inside the picture: the same shape (its slot's), as big as it was unless the picture is
 * smaller (then as big as fits), around the same middle, slid in from any bar. A view of the whole
 * frame (a slot showing the video whole: Blur fill, Horizontal, a visual shown whole) becomes the whole
 * picture.
 */
export function insidePicture<B extends PictureBox>(v: B, p: PictureBox): B {
  if (!hasBars(p) || !(v.w > 0 && v.h > 0)) return v
  if (v.x <= 1e-6 && v.y <= 1e-6 && v.w >= 1 - 1e-6 && v.h >= 1 - 1e-6) return { ...v, x: p.x, y: p.y, w: p.w, h: p.h }
  const k = Math.min(1, p.w / v.w, p.h / v.h)
  const w = v.w * k, h = v.h * k
  const cx = v.x + v.w / 2, cy = v.y + v.h / 2
  return { ...v, w, h, x: clamp(cx - w / 2, p.x, p.x + p.w - w), y: clamp(cy - h / 2, p.y, p.y + p.h - h) }
}

/**
 * The picture from the crops FFmpeg's cropdetect read on frames spread over the video (pixels: w, h,
 * x, y each): the widest of them (a dark frame reads narrower), as a share of the frame
 */
export function pictureFromCrops(crops: ReadonlyArray<{ w: number; h: number; x: number; y: number }>, frameW: number, frameH: number): PictureBox {
  if (!crops.length || !(frameW > 0 && frameH > 0)) return WHOLE_FRAME
  const left = Math.min(...crops.map(c => c.x)), right = Math.max(...crops.map(c => c.x + c.w))
  const top = Math.min(...crops.map(c => c.y)), bottom = Math.max(...crops.map(c => c.y + c.h))
  return pictureOrWhole({ x: left / frameW, y: top / frameH, w: (right - left) / frameW, h: (bottom - top) / frameH })
}
