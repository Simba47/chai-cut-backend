import { test } from 'node:test'
import assert from 'node:assert/strict'
import { keyWordsPrompt, keyWordTarget, pickKeyWords, readKeyWords } from './keyWords.js'

const said = 'No first of all I think with Pushpa Two we also distributed it in Tamil Nadu. I think Sukumar sir as a director and Allu Arjun as a hero was a magical combination'
const words = said.split(' ').map((word, i) => ({ word, start_ms: i * 300, end_ms: i * 300 + 250 }))
const at = (w: string) => words.findIndex(x => x.word.replace(/\W/g, '') === w)
const marks = (r: Record<string, true> | null | undefined) => Object.keys(r ?? {}).map(k => words.find(w => String(w.start_ms) === k)!.word)
const picked = (r: { emphasis: Record<string, true> } | Record<string, true> | null) => marks(r && 'emphasis' in r ? (r.emphasis as Record<string, true>) : (r as Record<string, true> | null))

test('a name of several words marks every word of it', () => {
  const r = readKeyWords(JSON.stringify({ keys: [{ i: at('Allu'), text: 'Allu Arjun' }, { i: at('Tamil'), text: 'Tamil Nadu' }] }), words)
  assert.deepEqual(picked(r), ['Tamil', 'Nadu.', 'Allu', 'Arjun'])
})

test('a pick whose number is a word or two off goes to the word it names; too far off, it is left out', () => {
  const r = readKeyWords(JSON.stringify({ keys: [{ i: at('magical') - 1, text: 'magical' }, { i: at('Sukumar') + 2, text: 'Sukumar' }, { i: 0, text: 'Pushpa' }] }), words)
  assert.deepEqual(picked(r), ['Sukumar', 'magical'])
})

test('filler and forms of address are never key words on their own; a word not in the clip is left out', () => {
  const r = readKeyWords(JSON.stringify({ keys: [{ i: at('sir'), text: 'sir' }, { i: at('a'), text: 'a' }, { i: at('in'), text: 'in' }, { i: 3, text: 'Mahesh' }, { i: at('Pushpa'), text: 'Pushpa' }] }), words)
  assert.deepEqual(picked(r), ['Pushpa'])
})

test('a phrase loses the little words at its ends; the same word or name is picked once', () => {
  const r = readKeyWords(JSON.stringify({ keys: [{ i: at('magical') - 1, text: 'a magical combination' }, { i: at('Pushpa'), text: 'Pushpa' }, { i: at('Pushpa'), text: 'Pushpa' }] }), words)
  assert.deepEqual(picked(r), ['Pushpa', 'magical', 'combination'])
  // (longer than a name or a number with its unit: left out)
  const long = readKeyWords(JSON.stringify({ keys: [{ i: 0, text: 'No first of all' }] }), words)
  assert.deepEqual(long!.emphasis, {})
})

test('a pick that starts at the same moment as a word outside it is left out (it would mark that word too)', () => {
  const same = [{ word: 'a', start_ms: 0, end_ms: 100 }, { word: 'magical', start_ms: 0, end_ms: 300 }, { word: 'Allu', start_ms: 400, end_ms: 500 }, { word: 'Arjun', start_ms: 400, end_ms: 700 }, { word: 'Pushpa', start_ms: 800, end_ms: 900 }]
  const r = readKeyWords(JSON.stringify({ keys: [{ i: 1, text: 'magical' }, { i: 2, text: 'Allu Arjun' }, { i: 4, text: 'Pushpa' }] }), same)
  // ("Allu Arjun" start together but are both the pick: kept)
  assert.deepEqual(Object.keys(r!.emphasis).map(Number).sort((a, b) => a - b), [400, 800])
})

test('no more than the clip’s share of key words (about one every three caption lines)', () => {
  const many = Array.from({ length: 120 }, (_, i) => ({ word: `word${i}`, start_ms: i * 300, end_ms: i * 300 + 250 }))
  const r = readKeyWords(JSON.stringify({ keys: many.map((w, i) => ({ i, text: w.word })) }), many)
  assert.equal(keyWordTarget(120), 10)
  assert.equal(Object.keys(r!.emphasis).length, Math.ceil(10 * 1.25))
})

test('an answer that can’t be read is null; one with no picks is none', () => {
  assert.equal(readKeyWords('sorry', words), null)
  assert.equal(readKeyWords('{"keys": oops}', words), null)
  assert.deepEqual(readKeyWords('{"keys": []}', words), { emphasis: {}, line_keys: {} })
})

test('AI reads the clip as sentences with the video title, and is asked for at most its share', async () => {
  let prompt = '', opts: { thinkingBudget: number } | null = null
  const r = await pickKeyWords(words, 'Pushpa 2 interview', async (p, o) => { prompt = p; opts = o; return JSON.stringify({ keys: [{ i: at('Pushpa'), text: 'Pushpa' }], line: [] }) })
  assert.deepEqual(picked(r), ['Pushpa'])
  assert.match(prompt, /titled "Pushpa 2 interview"/)
  assert.match(prompt, new RegExp(`Pick at most ${keyWordTarget(words.length)},`))
  // (a sentence ends at its full stop: "Nadu." closes the first line)
  assert.match(prompt, /\d+=Nadu\.\n\d+=I /)
  assert.ok(opts!.thinkingBudget > 0)
  assert.equal(keyWordsPrompt(words, null).includes('titled'), false)
})

test('line words: one a phrase, checked like key words (filler, a word not there, one sharing its start: left out); the same word may come again', () => {
  const r = readKeyWords(JSON.stringify({ keys: [], line: [
    { i: at('distributed'), text: 'distributed' }, { i: at('director'), text: 'director' }, { i: at('a'), text: 'a' },
    { i: at('hero'), text: 'hero' }, { i: 2, text: 'Mahesh' }, { i: at('Pushpa'), text: 'Pushpa' }, { i: at('Pushpa'), text: 'Pushpa' },
  ] }), words)
  assert.deepEqual(marks(r!.line_keys), ['Pushpa', 'distributed', 'director', 'hero'])
  assert.deepEqual(r!.emphasis, {})
})

test('an answer from before line words (keys only) still gives its key words', () => {
  const r = readKeyWords(JSON.stringify({ keys: [{ i: at('Pushpa'), text: 'Pushpa' }] }), words)
  assert.deepEqual(picked(r), ['Pushpa'])
  assert.deepEqual(r!.line_keys, {})
})
