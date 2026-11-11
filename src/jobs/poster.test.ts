import { test } from 'node:test'
import assert from 'node:assert/strict'
import { posterAt, posterKeyFor } from './poster.js'

test('the picture is kept next to the video, as its editing copy', () => {
  assert.equal(posterKeyFor('videos/u1/abc.mp4'), 'videos/u1/abc_poster.jpg')
  assert.equal(posterKeyFor('videos/u1/my.clip.mov'), 'videos/u1/my.clip_poster.jpg')
})

test('the picture comes from a tenth of the way in, between 2 and 30 seconds', () => {
  assert.equal(posterAt(60_000), 6)
  assert.equal(posterAt(10_000), 2)
  assert.equal(posterAt(11 * 60_000), 30)
})

test('a very short video: its middle; no length known: 2 seconds', () => {
  assert.equal(posterAt(3000), 1.5)
  assert.equal(posterAt(null), 2)
  assert.equal(posterAt(0), 2)
})
