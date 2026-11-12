import { test } from 'node:test'
import assert from 'node:assert/strict'
import { LINK_BAR, fetchPct, savePct, progressFeed } from './linkProgress.js'

test('the steps fill the bar in order, with no gap the bar waits in', () => {
  assert.equal(fetchPct(0, 100), LINK_BAR.start)
  assert.equal(fetchPct(50, 100), 28)
  assert.equal(fetchPct(100, 100), LINK_BAR.fetched)
  assert.equal(fetchPct(500, 100), LINK_BAR.fetched)          // more than said: never past its step
  assert.equal(fetchPct(10, 0), LINK_BAR.start)               // size unknown
  assert.equal(savePct(0, 1000), LINK_BAR.saveFrom)
  assert.equal(savePct(1000, 1000), LINK_BAR.saved)
  assert.ok(LINK_BAR.start < LINK_BAR.fetched && LINK_BAR.fetched < LINK_BAR.checking
    && LINK_BAR.checking < LINK_BAR.saveFrom && LINK_BAR.saveFrom < LINK_BAR.saved && LINK_BAR.saved < LINK_BAR.almost
    && LINK_BAR.almost < 100)
})

test('progress is written one at a time, in order, never lower, and only the latest of a queue', async () => {
  const written: number[] = []
  let inFlight = 0, most = 0
  const feed = progressFeed(async pct => {
    inFlight++; most = Math.max(most, inFlight)
    // the first write is slow: the old way, a later one landed first and the bar went back
    await new Promise(r => setTimeout(r, pct === 10 ? 30 : 1))
    written.push(pct)
    inFlight--
  })
  feed.set(10)
  for (let p = 11; p <= 40; p++) feed.set(p)   // asked while 10 is still being written
  feed.set(20)                                   // lower: ignored
  await feed.done()
  assert.deepEqual(written, [10, 40])
  assert.equal(most, 1)
  feed.set(41); feed.set(42)
  await feed.done()
  assert.deepEqual(written, [10, 40, 41, 42])
})

test('a failed write does not stop later ones', async () => {
  const written: number[] = []
  const feed = progressFeed(async pct => { if (pct === 1) throw new Error('db hiccup'); written.push(pct) })
  feed.set(1)
  await feed.done()
  feed.set(2)
  await feed.done()
  assert.deepEqual(written, [2])
})
