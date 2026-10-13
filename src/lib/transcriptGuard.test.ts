import { test } from 'node:test'
import assert from 'node:assert/strict'
import { findLoops, runawayReason, trimRunaway, type GuardWord } from './transcriptGuard.js'

/** Words spread evenly over `seconds` */
const spoken = (text: string, seconds: number): GuardWord[] => {
  const parts = text.split(' ')
  return parts.map((word, i) => ({ word, start: (i * seconds) / parts.length, end: ((i + 0.9) * seconds) / parts.length }))
}

const NORMAL = 'ee cinema chala bagundi kani second half konchem slow ga undi ani naku anipinchindi meeru emantaru'

test('normal speech passes', () => {
  const words = spoken(Array(40).fill(NORMAL).join(' ').split(' ').slice(0, 600).join(' '), 300)
  // 600 words in 300 s (2 a second), and the sentence repeats only as a whole, every 16 words
  assert.equal(runawayReason(words, 300), null)
  assert.deepEqual(trimRunaway(words, 300), words)
})

test('a phrase looping hundreds of times is caught and cut back to two copies', () => {
  const loop = Array(400).fill('mari adi kaadu').join(' ')
  const words = spoken(`${NORMAL} ${loop} ${NORMAL}`, 300)
  assert.match(runawayReason(words, 300) ?? '', /3-word phrase repeated 400 times/)
  const kept = trimRunaway(words, 300)
  const n = NORMAL.split(' ').length
  assert.equal(kept.length, n + 6 + n)
  assert.deepEqual(kept.slice(n, n + 6).map(w => w.word), ['mari', 'adi', 'kaadu', 'mari', 'adi', 'kaadu'])
})

test('the production case: 8,132 words in 300 s is caught by its rate even without a clean loop', () => {
  const words = Array.from({ length: 8132 }, (_, i) => ({ word: `w${i % 997}x${i % 13}`, start: (i * 300) / 8132, end: (i * 300) / 8132 + 0.02 }))
  assert.match(runawayReason(words, 300) ?? '', /8132 words in 300 s/)
})

test('a word said a few times in a row is speech, not a loop', () => {
  const words = spoken('no no no no no I said no', 4)
  assert.deepEqual(findLoops(words), [])
  assert.equal(runawayReason(words, 4), null)
})

test('a single word repeated dozens of times is a loop', () => {
  const words = spoken(Array(50).fill('okay').join(' '), 60)
  assert.deepEqual(findLoops(words), [{ start: 0, words: 1, copies: 50 }])
})

test('a short chant or chorus repeated a few times is not a loop', () => {
  const words = spoken(Array(10).fill('jai ho').join(' '), 20)
  assert.deepEqual(findLoops(words), [])
})

test('punctuation and case do not hide a loop', () => {
  const words = spoken(Array(25).fill('Avunu, avunu.').join(' '), 60)
  assert.equal(findLoops(words).length, 1)
})

test('words timed outside the audio are dropped', () => {
  const words: GuardWord[] = [{ word: 'a', start: 1, end: 1.2 }, { word: 'b', start: 302, end: 302.3 }, { word: 'c', start: Number.NaN, end: 0 }]
  assert.deepEqual(trimRunaway(words, 300).map(w => w.word), ['a'])
})
