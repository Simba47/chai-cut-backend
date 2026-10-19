// ── Room for a crop box next to other people (Make my clips' split and trio boxes) ──
// A split or trio box shows its own people. On a close shot people sit near each other and are
// big, so a box sized from its own faces alone reached over the next person (user, 2026-10-19:
// a 4-person close shot's trio boxes overlapped). Now each box first gets its own faces with a
// little air; the space between two boxes is shared out by how much more each wanted; and a
// small gap is kept between them (user's choice). A box is narrower (zoomed in) when it has to be.
// All values are shares of the source frame's width (as detection and crop boxes are).

export interface FaceSpan { x: number; w: number }

/** Air a box keeps beside its own faces at least, as a share of a face's width */
export const MIN_AIR_FACE = 0.6
/** The gap kept between two boxes (and before another person's own air), share of the frame's width */
export const BOX_GAP = 0.012

const spanOf = (faces: FaceSpan[]) => ({ left: Math.min(...faces.map(f => f.x)), right: Math.max(...faces.map(f => f.x + f.w)) })
const airOf = (faces: FaceSpan[]) => MIN_AIR_FACE * faces.reduce((t, f) => t + f.w, 0) / faces.length

/** The narrowest a box for `own` faces may be: its faces with a little air either side */
export function minWidthFor(own: FaceSpan[]): number {
  const { left, right } = spanOf(own)
  return right - left + 2 * airOf(own)
}

/**
 * The stretch [L, R] a box for `own` faces may cover without reaching `others` — people who
 * aren't in a box beside it (a reaction's other people): up to a gap before each one's face and
 * its air. Its own faces always stay inside.
 */
export function roomFor(own: FaceSpan[], others: FaceSpan[]): [number, number] {
  const { left, right } = spanOf(own)
  const mid = (left + right) / 2
  let L = 0, R = 1
  for (const o of others) {
    const air = MIN_AIR_FACE * o.w + BOX_GAP
    if (o.x + o.w / 2 < mid) L = Math.max(L, o.x + o.w + air)
    else R = Math.min(R, o.x - air)
  }
  return [Math.min(L, left), Math.max(R, right)]
}

/**
 * Rooms for boxes side by side, left to right (a trio of groups): each group's faces and the
 * width it wants. Between two neighbours, each first gets its own air; the space left over goes
 * first to the box with fewer people (one face needs the air — a box of two is wide anyway, as
 * the user framed it by hand), else is shared by how much more each wanted; BOX_GAP stays between.
 */
export function sideBySideRooms(groups: Array<{ faces: FaceSpan[]; want: number }>): Array<[number, number]> {
  const g = groups.map(x => {
    const { left, right } = spanOf(x.faces), air = airOf(x.faces), cx = (left + right) / 2
    // (where the wanted box would sit: moved inward at the frame's edge)
    const wantL = Math.max(0, Math.min(1 - x.want, cx - x.want / 2))
    return { n: x.faces.length, minL: Math.max(0, left - air), minR: Math.min(1, right + air), wantL, wantR: wantL + x.want }
  })
  const rooms: Array<[number, number]> = g.map(() => [0, 1])
  for (let i = 0; i + 1 < g.length; i++) {
    const a = g[i], b = g[i + 1]
    let cut: number
    if (a.wantR + BOX_GAP <= b.wantL) cut = (a.wantR + b.wantL) / 2            // both fit as wanted
    else if (b.minL - a.minR <= BOX_GAP) cut = (a.minR + b.minL) / 2          // not even their air: halfway
    else {
      const space = b.minL - a.minR - BOX_GAP
      const moreA = Math.max(0, a.wantR - a.minR), moreB = Math.max(0, b.minL - b.wantL)
      const toA = a.n < b.n ? Math.min(moreA, space)
        : b.n < a.n ? space - Math.min(moreB, space)
          : moreA + moreB > 0 ? (space * moreA) / (moreA + moreB) : space / 2
      cut = a.minR + toA + BOX_GAP / 2
    }
    rooms[i][1] = cut - BOX_GAP / 2
    rooms[i + 1][0] = cut + BOX_GAP / 2
  }
  return rooms
}

/**
 * A box `w` wide (as wanted) centred on `cx`, fitted into `room`: narrower when the room is
 * smaller (never below `minW`), moved within the room rather than past it, and inside the frame.
 */
export function fitInRoom(cx: number, w: number, room: [number, number], minW: number): { x: number; w: number } {
  const [L, R] = room
  const fitted = Math.min(1, Math.max(Math.min(w, R - L), Math.min(w, minW)))
  let x = cx - fitted / 2
  if (fitted <= R - L) x = Math.min(Math.max(x, L), R - fitted)
  return { x: Math.max(0, Math.min(1 - fitted, x)), w: fitted }
}

const median = (v: number[]) => { const s = [...v].sort((a, b) => a - b); return s.length ? s[Math.floor(s.length / 2)] : NaN }

/**
 * One steady width and shift for a person's box over a stretch of frames (so it doesn't pulse):
 * each frame gives the room on either side of the person (`own` their face, `others` everyone
 * else there); the box is `want` wide unless the room (median over the frames) is smaller, and
 * shifted the same way in every frame when one side has less room. Null: no frame to go by.
 */
export function steadyRoom(frames: Array<{ cx: number; own: FaceSpan; others: FaceSpan[] }>, want: number): { w: number; shift: number } | null {
  const rooms = frames.map(f => { const [L, R] = roomFor([f.own], f.others); return { left: f.cx - L, right: R - f.cx, min: minWidthFor([f.own]) } })
  if (!rooms.length) return null
  const left = median(rooms.map(r => r.left)), right = median(rooms.map(r => r.right)), minW = median(rooms.map(r => r.min))
  const w = Math.min(1, Math.max(Math.min(want, left + right), Math.min(want, minW)))
  // (both sides have room for half of it: centred; else moved toward the side with more room)
  const shift = w <= left + right ? Math.min(Math.max(0, w / 2 - left), right - w / 2) : 0
  return { w, shift }
}
