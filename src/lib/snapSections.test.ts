import { test } from 'node:test'
import assert from 'node:assert/strict'
import { snapSectionBoundaries, type SnapSection } from './snapSections.js'

const sec = (id: string, start_ms: number, end_ms: number, movable = true): SnapSection => ({ id, start_ms, end_ms, movable })

// A real clip (naresh, 0:00–0:59): its sections and the camera cuts found in it
const sections = [
  sec('V1', 0, 9500), sec('S1', 9500, 20200), sec('T1', 20200, 28400), sec('S2', 28400, 38200),
  sec('V2', 38200, 43800), sec('S3', 43800, 48800), sec('T2', 48800, 59600),
]
const cuts = [3100, 19200, 19900, 25200, 26200, 33000, 42200, 42800, 44400, 45200, 45800, 46900, 47500, 48200, 49300, 49800, 51500]

test('a boundary moves to the nearest camera cut within 4 s', () => {
  const moved = Object.fromEntries(snapSectionBoundaries(sections, cuts).map(s => [s.id, s]))
  // Split / Trio at 20.2: the cut at 19.9 is 0.3 s before it
  assert.equal(moved.S1.end_ms, 19900)
  assert.equal(moved.T1.start_ms, 19900)
  // Trio / Split at 28.4: the cut at 26.2 is the nearest (2.2 s before)
  assert.equal(moved.T1.end_ms, 26200)
  assert.equal(moved.S2.start_ms, 26200)
  // Split / Vertical at 38.2: the cut at 42.2 is 4.0 s after
  assert.equal(moved.S2.end_ms, 42200)
  assert.equal(moved.V2.start_ms, 42200)
})

test('a boundary with no cut near it stays', () => {
  // 9.5: the nearest cuts are 3.1 and 19.2, both more than 4 s away
  const moved = snapSectionBoundaries(sections, cuts).map(s => s.id)
  assert.ok(!moved.includes('V1'))
})

test('a boundary already on a cut does not move', () => {
  const out = snapSectionBoundaries([sec('a', 0, 5000), sec('b', 5000, 12000)], [5100])
  assert.deepEqual(out, [])
})

test('locked or special sections are left alone', () => {
  const out = snapSectionBoundaries([sec('a', 0, 5000), sec('b', 5000, 12000, false)], [3500])
  assert.deepEqual(out, [])
})

test('a section never becomes shorter than half a second', () => {
  const out = snapSectionBoundaries([sec('a', 0, 5000), sec('b', 5000, 12000)], [200])
  assert.deepEqual(out, [])
})

test('two boundaries never take the same cut', () => {
  const out = snapSectionBoundaries([sec('a', 0, 4000), sec('b', 4000, 6000), sec('c', 6000, 12000)], [5000])
  const cutsUsed = out.flatMap(s => [s.start_ms, s.end_ms]).filter(t => t === 5000)
  assert.ok(cutsUsed.length <= 2)
  const byId = Object.fromEntries(out.map(s => [s.id, s]))
  // the first boundary takes it; the second has no cut left and stays, so b is not squeezed to nothing
  assert.equal(byId.a?.end_ms, 5000)
  assert.equal(byId.c, undefined)
})

test('sections that do not follow each other are not joined', () => {
  const out = snapSectionBoundaries([sec('a', 0, 5000), sec('b', 7000, 12000)], [5500])
  assert.deepEqual(out, [])
})

test('no cuts: nothing changes', () => {
  assert.deepEqual(snapSectionBoundaries(sections, []), [])
})
