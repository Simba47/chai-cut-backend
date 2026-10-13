import { test } from 'node:test'
import assert from 'node:assert/strict'
import { convertWordForWord } from './wordConvert.js'

const words = Array.from({ length: 250 }, (_, i) => `w${i}`)
const upper = (batch: string[]) => batch.map(w => w.toUpperCase())

test('every word converted, in order, in batches of 80', async () => {
  const sizes: number[] = []
  const out = await convertWordForWord(words, async batch => { sizes.push(batch.length); return JSON.stringify(upper(batch)) }, '[t]')
  assert.deepEqual(out, words.map(w => w.toUpperCase()))
  assert.deepEqual(sizes.sort((a, b) => b - a), [80, 80, 80, 10])
})

test('a batch that comes back with one word too many is split until it comes back right', async () => {
  // The model adds a stray word whenever it gets more than 40 words (the production case: 80 in, 81 out)
  const out = await convertWordForWord(words, async batch => JSON.stringify(batch.length > 40 ? [...upper(batch), 'extra'] : upper(batch)), '[t]')
  assert.deepEqual(out, words.map(w => w.toUpperCase()))
})

test('one stubborn word costs at most a 10-word piece, not the whole batch', async () => {
  // Any batch holding w5 always comes back one short
  const out = await convertWordForWord(words, async batch => JSON.stringify(batch.includes('w5') ? upper(batch).slice(1) : upper(batch)), '[t]')
  const lost = out.filter(v => v === undefined).length
  assert.ok(lost > 0 && lost <= 10, `lost ${lost}`)
  assert.equal(out[200], 'W200')
})

test('a request that fails outright is tried twice, then its words are left for the fallback', async () => {
  let calls = 0
  const out = await convertWordForWord(['a', 'b'], async () => { calls++; throw new Error('Gemini 503') }, '[t]')
  assert.deepEqual(out, [undefined, undefined])
  assert.equal(calls, 2)
})

test('a second try that comes back right is used', async () => {
  let calls = 0
  const out = await convertWordForWord(['a', 'b', 'c'], async batch => JSON.stringify(++calls === 1 ? ['A'] : upper(batch)), '[t]')
  assert.deepEqual(out, ['A', 'B', 'C'])
})

test('no more than three batches are sent at once', async () => {
  let open = 0, most = 0
  await convertWordForWord(Array.from({ length: 800 }, (_, i) => `w${i}`), async batch => {
    most = Math.max(most, ++open)
    await new Promise(r => setTimeout(r, 5))
    open--
    return JSON.stringify(batch)
  }, '[t]')
  assert.equal(most, 3)
})
