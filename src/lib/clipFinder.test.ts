import { test } from 'node:test'
import assert from 'node:assert/strict'
import { buildLines, buildPhrases, refineEdges, withRoom, findClips, type FinderWord, type FoundClip } from './clipFinder.js'

// Speech with no punctuation (as Telugu comes back): words 300 ms apart (a 100 ms gap), a pause of
// `pause` ms after the words listed, so lines end only at a breath after 25 words
function speech(n: number, pauses: Record<number, number>, from = 0): FinderWord[] {
  const out: FinderWord[] = []
  let t = from
  for (let i = 0; i < n; i++) {
    out.push({ word: `w${i}`, start_ms: t, end_ms: t + 200 })
    t += 300 + (pauses[i] ?? 0)
  }
  return out
}

test('every line knows its words, end to end', () => {
  const words = speech(80, { 29: 400, 59: 400 })
  const lines = buildLines(words)
  assert.equal(lines[0].first_word, 0)
  lines.slice(1).forEach((l, i) => assert.equal(l.first_word, lines[i].last_word! + 1))
  assert.equal(lines[lines.length - 1].last_word, words.length - 1)
})

test('phrases end at a pause after a few words and never cross a line: a sentence end inside a long line is an edge', () => {
  // A 30-word line (it ends at the breath after word 29); inside it a 500 ms pause after word 11
  const words = speech(60, { 11: 500, 29: 400 })
  const lines = buildLines(words)
  assert.equal(lines[0].last_word, 29)
  const ph = buildPhrases(words, lines, 0, lines.length - 1)
  assert.deepEqual(ph.filter(p => p.line === 0).map(p => p.text.split(' ').length), [12, 18])
  // the line's own end is a phrase end
  assert.ok(ph.some(p => p.end_ms === lines[0].end_ms))
  assert.ok(ph.every(p => lines[p.line].start_ms <= p.start_ms && p.end_ms <= lines[p.line].end_ms))
})

test('a phrase is at most 25 words even with no pause', () => {
  const words = speech(45, {})
  const lines = buildLines(words)
  const ph = buildPhrases(words, lines, 0, 0)
  assert.ok(ph.every(p => p.text.split(' ').length <= 25))
})

// A clip on lines L1–L3 of 6 lines, each line 30 words (9 s), each with a pause inside after word 12
const words = speech(180, Object.fromEntries([0, 1, 2, 3, 4, 5].flatMap(l => [[l * 30 + 11, 500], [l * 30 + 29, 400]])))
const lines = buildLines(words)
const clip = (s: number, e: number): FoundClip => ({
  start_line: s, end_line: e, start_ms: lines[s].start_ms, end_ms: lines[e].end_ms,
  title: 't', summary: 's', reason: 'r', score: 50, subscores: { hook: 5, emotion: 5, story: 5, value: 5, clarity: 5 }, window: 0,
})

test('the edge check works in phrases: the clip can end where a sentence ends inside a line, asked at temperature 0', async () => {
  let asked: { system: string; user: string; temperature?: number } | null = null
  // phrases: 2 per line → line 3's first phrase is P6, its second P7
  const ask = async (system: string, user: string, o?: { temperature?: number }) => {
    asked = { system, user, temperature: o?.temperature }
    return '{"start_phrase": 2, "end_phrase": 6, "why": "ends on the conclusion"}'
  }
  const { clip: c, why } = await refineEdges(clip(1, 3), lines, words, ask)
  assert.equal(asked!.temperature, 0)
  assert.match(asked!.user, /> P2 \[/)
  assert.match(asked!.user, /> P7 \[/)
  assert.match(asked!.user, /\n {2}P8 \[/)
  assert.equal(why, 'ends on the conclusion')
  assert.equal(c.start_ms, lines[1].start_ms)
  // the end of line 3's first phrase (its word 11), not the line's end
  assert.equal(c.end_ms, words[3 * 30 + 11].end_ms)
  assert.equal(c.end_line, 3)
})

test('an answer outside the phrases shown, or too short, keeps the clip as it was', async () => {
  for (const answer of ['{"start_phrase": 2, "end_phrase": 99}', '{"start_phrase": 2, "end_phrase": 2}', 'no json']) {
    const r = await refineEdges(clip(1, 3), lines, words, async () => answer)
    assert.equal(r.why, null)
    assert.equal(r.clip.end_ms, lines[3].end_ms)
  }
})

test('room at the edges comes from the words right next to them, inside a line too', () => {
  const c = { ...clip(1, 3), end_ms: words[3 * 30 + 11].end_ms }
  const r = withRoom(c, words, 10 * 60_000)
  // before: the 400 ms breath after line 0 (+100 ms gap) → 350 ms lead; after: the 500 ms pause (+100) → 540 ms of tail
  assert.equal(c.start_ms - r.start_ms, 350)
  assert.equal(r.end_ms - c.end_ms, 600 - 60)
})

test('findClips: picked by lines, edges to the phrase, room added', async () => {
  const ask = async (system: string, user: string) => (/^You check where/.test(system)
    ? '{"start_phrase": 2, "end_phrase": 6, "why": "w"}'
    : '[{"start_line":1,"end_line":3,"title":"T","summary":"S","reason":"R","hook":8,"emotion":8,"story":8,"value":8,"clarity":8}]')
  const out = await findClips({ words, durationMs: 10 * 60_000, mode: { kind: 'best' }, ask, limit: 1 })
  assert.equal(out.length, 1)
  assert.equal(out[0].end_ms, words[3 * 30 + 11].end_ms + 540)
  assert.equal(out[0].start_ms, lines[1].start_ms - 350)
})
