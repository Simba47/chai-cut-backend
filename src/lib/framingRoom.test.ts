import { test } from 'node:test'
import assert from 'node:assert/strict'
import { BOX_GAP, fitInRoom, minWidthFor, sideBySideRooms, steadyRoom, type FaceSpan } from './framingRoom.js'

// Faces (shares of the frame's width) measured from the user's screenshots of one podcast
// (2026-10-19): four people in a row — a close shot, where the AI's trio boxes overlapped, and a
// wide shot of the same four around the table, where they were right.
const close: FaceSpan[] = [{ x: 0.063, w: 0.09 }, { x: 0.322, w: 0.088 }, { x: 0.515, w: 0.093 }, { x: 0.818, w: 0.09 }]
const wide: FaceSpan[] = [{ x: 0.034, w: 0.056 }, { x: 0.316, w: 0.053 }, { x: 0.613, w: 0.05 }, { x: 0.879, w: 0.053 }]

/** Make my clips' trio of a 4-person shot (ai_edit.ts groupSlots): outer two alone, the middle two together */
function trio(faces: FaceSpan[], kept = true) {
  const groups = [[faces[0]], [faces[1], faces[2]], [faces[3]]].map(g => {
    const left = Math.min(...g.map(x => x.x)), right = Math.max(...g.map(x => x.x + x.w))
    const faceW = g.reduce((t, x) => t + x.w, 0) / g.length
    return { faces: g, cx: (left + right) / 2, want: Math.min(0.6, Math.max(0.18, right - left + 3 * faceW)) }
  })
  if (!kept) return groups.map(g => { const w = g.want; return { x: Math.max(0, Math.min(1 - w, g.cx - w / 2)), w } })
  const rooms = sideBySideRooms(groups)
  return groups.map((g, i) => fitInRoom(g.cx, g.want, rooms[i], minWidthFor(g.faces)))
}
const inside = (f: FaceSpan, b: { x: number; w: number }) => f.x >= b.x - 1e-9 && f.x + f.w <= b.x + b.w + 1e-9
const touches = (f: FaceSpan, b: { x: number; w: number }) => f.x + f.w > b.x && f.x < b.x + b.w

test('close shot: the old sizing overlapped (the bug as the user saw it)', () => {
  const [top, middle] = trio(close, false)
  assert.ok(top.x + top.w > middle.x, 'Top reached into Middle')
  assert.ok(touches(close[1], top), "Top showed part of the man in glasses' face")
})

test('close shot: boxes no longer overlap, with a small gap between them', () => {
  const [top, middle, bottom] = trio(close)
  assert.ok(middle.x - (top.x + top.w) >= BOX_GAP - 1e-9, `Top ${top.x + top.w} / Middle ${middle.x}`)
  assert.ok(bottom.x - (middle.x + middle.w) >= BOX_GAP - 1e-9, `Middle ${middle.x + middle.w} / Bottom ${bottom.x}`)
})

test('close shot: each box shows its own people and nobody else', () => {
  const [top, middle, bottom] = trio(close)
  assert.ok(inside(close[0], top) && inside(close[1], middle) && inside(close[2], middle) && inside(close[3], bottom))
  assert.ok(!touches(close[1], top) && !touches(close[0], middle) && !touches(close[3], middle) && !touches(close[2], bottom))
})

test("close shot: about as the user framed it by hand (Top ≈ 27–33% of the frame, not cropped tight on the face)", () => {
  const [top, middle, bottom] = trio(close)
  assert.ok(top.w >= 0.24 && top.w <= 0.36, `Top ${top.w}`)
  assert.ok(bottom.w >= 0.24, `Bottom ${bottom.w}`)
  assert.ok(middle.w >= minWidthFor([close[1], close[2]]) - 1e-9, `Middle ${middle.w}`)
})

test('wide shot: unchanged — every box was already clear of the others', () => {
  assert.deepEqual(trio(wide), trio(wide, false))
})

test('a reaction box (one person) beside a close neighbour narrows and steps away from them, steadily', () => {
  // The man in glasses in a split, with the next person close on his right
  const frames = [0, 1, 2].map(i => ({ cx: 0.366 + i * 0.002, own: { x: 0.322 + i * 0.002, w: 0.088 }, others: [close[0], close[2], close[3]] }))
  const r = steadyRoom(frames, 0.34)!
  const box = { x: 0.366 + r.shift - r.w / 2, w: r.w }
  assert.ok(r.w < 0.34, `narrower than the 0.34 default: ${r.w}`)
  assert.ok(!touches(close[0], box) && !touches(close[2], box), JSON.stringify(box))
  assert.ok(inside(frames[0].own, box))
})

test('a reaction box with nobody close keeps its usual width, centred', () => {
  const r = steadyRoom([{ cx: 0.5, own: { x: 0.47, w: 0.06 }, others: [{ x: 0.05, w: 0.06 }] }], 0.4)!
  assert.equal(r.w, 0.4)
  assert.equal(r.shift, 0)
})
