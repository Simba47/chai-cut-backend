import { test } from 'node:test'
import assert from 'node:assert/strict'

process.env.DATABASE_URL ??= 'postgres://test:test@127.0.0.1:1/test'
const { fixSpellings } = await import('./glossary.js')

const say = (...ws: string[]) => ws.map((w, i) => ({ word: w, word_roman: null as string | null, start_ms: i * 400, end_ms: i * 400 + 350 }))
const words = (r: { word: string }[]) => r.map(x => x.word).join(' ')

test('a name said split, or in the wrong case, is spelled as the glossary has it', () => {
  const { rows, fixed } = fixSpellings(say('welcome', 'to', 'short', 'cut', 'today'), ['Shortcut'])
  assert.equal(words(rows), 'welcome to Shortcut today')
  assert.equal(fixed, 1)
  const joined = rows[2]
  assert.deepEqual([joined.start_ms, joined.end_ms], [800, 1550], 'the joined word spans both')
  assert.equal(words(fixSpellings(say('SHORTCUT', 'rocks'), ['Shortcut']).rows), 'Shortcut rocks')
})

test('punctuation around the word stays', () => {
  assert.equal(words(fixSpellings(say('try', 'shortcut,', 'now'), ['Shortcut']).rows), 'try Shortcut, now')
})

test('a long name off by a letter is corrected; short words are never guessed at', () => {
  assert.equal(words(fixSpellings(say('I', 'use', 'Shortcat', 'daily'), ['Shortcut']).rows), 'I use Shortcut daily')
  assert.equal(words(fixSpellings(say('the', 'Mutinie', 'team'), ['Mutinee']).rows), 'the Mutinee team', '6+ letters: one off')
  assert.equal(words(fixSpellings(say('the', 'Mutiny', 'team'), ['Mutinee']).rows), 'the Mutiny team', 'two off a 7-letter name: left alone')
  // "cat" is not "Cut": short names only match exactly
  assert.equal(words(fixSpellings(say('my', 'cat', 'sat'), ['Cut']).rows), 'my cat sat')
  assert.equal(words(fixSpellings(say('he', 'shouted', 'loudly'), ['Shortcut']).rows), 'he shouted loudly', 'two letters off a 8-letter name: left alone')
})

test('a two-word name is joined from its words, longer names win', () => {
  const { rows } = fixSpellings(say('open', 'shortcut', 'pro', 'here'), ['Shortcut', 'Shortcut Pro'])
  assert.equal(words(rows), 'open Shortcut Pro here')
})

test('Telugu or Hindi letters are left as they are; the English-letter spelling is corrected', () => {
  const rows = [{ word: 'షార్ట్‌కట్', word_roman: 'shortkat', start_ms: 0, end_ms: 400 }, { word: 'బాగుంది', word_roman: 'baagundi', start_ms: 400, end_ms: 800 }]
  const out = fixSpellings(rows, ['Shortkut']).rows
  assert.equal(out[0].word, 'షార్ట్‌కట్')
  assert.equal(out[0].word_roman, 'Shortkut')
  assert.equal(out[1].word_roman, 'baagundi')
})

test('no glossary: nothing changes', () => {
  const r = say('hello', 'there')
  assert.equal(fixSpellings(r, []).rows, r)
})
