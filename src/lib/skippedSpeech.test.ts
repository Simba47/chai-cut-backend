import { test } from 'node:test'
import assert from 'node:assert/strict'
import { skippedGaps, rereadWindows, wordsInGaps, REREAD_MAX, REREAD_WINDOW_SEC } from './skippedSpeech.js'

const w = (start: number, end: number) => ({ word: 'x', start, end })

test('gaps of 3 s or more between words, at the start and the end of the piece too', () => {
  const words = [w(4, 5), w(5.2, 6), w(8.5, 9), w(20, 21)]
  assert.deepEqual(skippedGaps(words, 30), [[0, 4], [9, 20], [21, 30]])
  // (2.5 s between 6 and 8.5: a pause, not a gap)
})

test('gaps close together share one stretch; far apart, each its own; padded, inside the piece', () => {
  const ws = rereadWindows([[0, 4], [9, 20], [60, 64]], 300)
  assert.deepEqual(ws.map(x => [x.from, x.to, x.gaps.length]), [[0, 20.3, 2], [59.7, 64.3, 1]])
})

test('a stretch never grows past 40 s', () => {
  const ws = rereadWindows([[0, 15], [17, 30], [33, 45]], 300)
  assert.ok(ws.every(x => x.to - x.from <= REREAD_WINDOW_SEC + 0.6 + 1e-9), JSON.stringify(ws))
  assert.equal(ws.length, 2)
})

test('a long gap (85 s at a piece end) is read in parts of at most 40 s', () => {
  const ws = rereadWindows([[215, 300]], 300)
  assert.equal(ws.length, 3)
  assert.ok(ws.every(x => x.gaps.every(([a, b]) => b - a <= REREAD_WINDOW_SEC)))
  assert.equal(ws[0].from, 214.7)
  assert.equal(ws[ws.length - 1].to, 300)
})

test(`more than ${REREAD_MAX} stretches: the ones with the most gap, in time order`, () => {
  const gaps = Array.from({ length: 20 }, (_, i) => [i * 15, i * 15 + (i === 3 ? 3 : 5)] as [number, number])
  const ws = rereadWindows(gaps, 300)
  assert.equal(ws.length, REREAD_MAX)
  assert.ok(!ws.some(x => x.gaps[0][0] === 45), 'the smallest gap is left out')
  assert.deepEqual(ws.map(x => x.from), [...ws.map(x => x.from)].sort((a, b) => a - b))
})

test('only words starting inside a gap come back, at piece time', () => {
  const win = { from: 9.7, to: 20.3, gaps: [[10, 20]] as [number, number][] }
  // read from the stretch's start: 0.1 s (before the gap, already read), 1 s and 9 s (inside)
  const got = wordsInGaps(win, [w(0.1, 0.25), w(1, 1.4), w(9, 9.5), w(10.5, 10.6)])
  assert.deepEqual(got.map(x => [Math.round(x.start * 10) / 10, Math.round(x.end * 10) / 10]), [[10.7, 11.1], [18.7, 19.2]])
})
