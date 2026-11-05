import { test } from 'node:test'
import assert from 'node:assert/strict'
import { pictureOrWhole, pictureFromCrops, toPicture, fromPicture, insidePicture, pictureAR, WHOLE_FRAME } from './pictureBox.js'

// "Telugu test": 1920×1080 with 132 px bars top and bottom (cropdetect: 1920x816+0+132)
const LETTERBOX = pictureFromCrops([{ w: 1920, h: 816, x: 0, y: 132 }], 1920, 1080)
const near = (a: number, b: number) => Math.abs(a - b) < 1e-9

test('the picture from cropdetect: the widest crop of all frames; no bars is the whole frame', () => {
  assert.ok(near(LETTERBOX.y, 132 / 1080) && near(LETTERBOX.h, 816 / 1080) && LETTERBOX.x === 0 && LETTERBOX.w === 1)
  // one dark frame read narrow (the iPhone video): the widest wins → whole frame
  assert.deepEqual(pictureFromCrops([{ w: 996, h: 1772, x: 1422, y: 0 }, { w: 3840, h: 1772, x: 0, y: 0 }], 3840, 1772), WHOLE_FRAME)
  assert.deepEqual(pictureFromCrops([], 1920, 1080), WHOLE_FRAME)
})

test('thin edges are not bars; a broken or tiny picture is the whole frame', () => {
  assert.deepEqual(pictureOrWhole({ x: 0, y: 0.01, w: 1, h: 0.98 }), WHOLE_FRAME)
  assert.deepEqual(pictureOrWhole({ x: 0, y: 0.4, w: 1, h: 0.2 }), WHOLE_FRAME)
  assert.deepEqual(pictureOrWhole(null), WHOLE_FRAME)
  assert.deepEqual(pictureOrWhole({ x: 0, y: NaN, w: 1, h: 1 }), WHOLE_FRAME)
})

test('into the picture and back gives the same box; the picture is wider than the frame', () => {
  const face = { x: 0.4, y: 0.3, w: 0.1, h: 0.18 }
  const back = fromPicture(toPicture(face, LETTERBOX), LETTERBOX)
  for (const k of ['x', 'y', 'w', 'h'] as const) assert.ok(near(back[k], face[k]))
  assert.ok(near(pictureAR(16 / 9, LETTERBOX), (16 / 9) / (816 / 1080)))
})

test('a view reaching into the top bar slides down into the picture, same size and shape', () => {
  const v = { x: 0.3, y: 0.05, w: 0.3, h: 0.5 }
  const r = insidePicture(v, LETTERBOX)
  assert.ok(near(r.y, LETTERBOX.y) && near(r.w, v.w) && near(r.h, v.h) && near(r.x, v.x))
})

test('a full-height Vertical shrinks to the height of the picture, same shape, same middle', () => {
  const v = { x: 0.34, y: 0, w: 0.316, h: 1 }
  const r = insidePicture(v, LETTERBOX)
  assert.ok(near(r.h, LETTERBOX.h) && near(r.y, LETTERBOX.y))
  assert.ok(near(r.w / r.h, v.w / v.h))
  assert.ok(near(r.x + r.w / 2, v.x + v.w / 2))
})

test('the whole frame (Blur fill, Horizontal, a visual shown whole) becomes the whole picture', () => {
  const r = insidePicture({ x: 0, y: 0, w: 1, h: 1, t_ms: 0 }, LETTERBOX)
  assert.deepEqual([r.x, r.y, r.w, r.h, r.t_ms], [LETTERBOX.x, LETTERBOX.y, LETTERBOX.w, LETTERBOX.h, 0])
})

test('no bars: views are left exactly as they are', () => {
  const v = { x: 0.3, y: 0, w: 0.3, h: 1, t_ms: 5 }
  assert.equal(insidePicture(v, WHOLE_FRAME), v)
})
