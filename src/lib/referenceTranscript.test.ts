import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  referenceMode, parseTranscriberReply, momentAgreement, parseResultCheck,
  referencePercent, inRollout, wordsSource, captionRanges, geminiLanguage,
} from './referenceTranscript.js'

test('checking a submitted transcript: still working, done, or failed', () => {
  assert.deepEqual(parseResultCheck(202, { status: 'running' }), { status: 'pending' })
  const done = parseResultCheck(200, { words: [{ word: 'ఒకటి', start_ms: 0, end_ms: 400 }], model: 'm' })
  assert.equal(done.status, 'done')
  assert.equal(done.status === 'done' && done.reply.words.length, 1)
  assert.deepEqual(parseResultCheck(500, { error: 'out of memory' }), { status: 'failed', error: 'out of memory' })
  assert.deepEqual(parseResultCheck(404, null), { status: 'failed', error: 'the transcriber answered 404' })
  // a 200 without words is a failure, not an empty transcript
  assert.equal(parseResultCheck(200, { model: 'm' }).status, 'failed')
})

test('the switch is off unless set to shadow or on', () => {
  assert.equal(referenceMode(undefined), 'off')
  assert.equal(referenceMode(''), 'off')
  assert.equal(referenceMode('shadow'), 'shadow')
  assert.equal(referenceMode(' ON '), 'on')
  assert.equal(referenceMode('yes'), 'off')
})

test("the transcriber's reply: words kept in time order, broken ones dropped", () => {
  const reply = parseTranscriberReply({
    model: 'indic-transcribe-core', language: 'te', audio_seconds: 12.5, gpu_seconds: 1.2,
    words: [
      { word: 'రెండు', start_ms: 900, end_ms: 1300 },
      { word: 'ఒకటి', start_ms: 100, end_ms: 600 },
      { word: '', start_ms: 1400, end_ms: 1500 },             // empty
      { word: 'x', start_ms: 2000, end_ms: 1900 },             // ends before it starts
      { word: 'y', start_ms: 'soon', end_ms: 2100 },           // not a number
    ],
  })
  assert.deepEqual(reply.words.map(w => w.word), ['ఒకటి', 'రెండు'])
  assert.equal(reply.model, 'indic-transcribe-core')
  assert.equal(reply.gpuSeconds, 1.2)
})

test('a reply without a word list is refused', () => {
  assert.throws(() => parseTranscriberReply({ model: 'm' }), /no words/)
  assert.throws(() => parseTranscriberReply(null), /no words/)
})

test('moment agreement: how many of the baseline picks the reference transcript also found', () => {
  const m = (s: number, e: number) => ({ start_ms: s * 1000, end_ms: e * 1000 })
  const base = [m(0, 30), m(100, 140), m(300, 330), m(500, 560)]
  const other = [m(5, 32), m(310, 340), m(800, 830)]
  const a = momentAgreement(base, other)
  // 0-30 and 300-330 overlap more than half of the shorter clip; the other two do not
  assert.equal(a.baselineFound, 0.5)
  // of the reference picks, 2 of 3 are in the baseline
  assert.equal(a.referenceInBaseline, 2 / 3)
  assert.equal(a.baselineCount, 4)
  assert.equal(a.referenceCount, 3)
})

test('moment agreement with nothing to compare is zero, not NaN', () => {
  assert.deepEqual(momentAgreement([], []), { baselineFound: 0, referenceInBaseline: 0, baselineCount: 0, referenceCount: 0 })
})

// ── Phase 2 (REFERENCE_TRANSCRIBER=on) ──────────────────────────────────────

test('the share of videos on the reference transcript: 0-100, all of them when not set', () => {
  assert.equal(referencePercent(undefined), 100)
  assert.equal(referencePercent(''), 100)
  assert.equal(referencePercent('10'), 10)
  assert.equal(referencePercent(' 50 '), 50)
  assert.equal(referencePercent('0'), 0)
  assert.equal(referencePercent('250'), 100)
  assert.equal(referencePercent('-5'), 0)
  assert.equal(referencePercent('ten'), 100)
})

test('a video is in or out of the share for good, and the share holds about that many', () => {
  const ids = Array.from({ length: 2000 }, (_, i) => `${i.toString(16).padStart(8, '0')}-0a80-4b02-a0d5-1bd31638cc32`)
  const id = '57c01f3b-0a80-4b02-a0d5-1bd31638cc32'
  assert.equal(inRollout(id, 37), inRollout(id, 37))
  assert.equal(inRollout(id, 0), false)
  assert.equal(inRollout(id, 100), true)
  // a video in at 10% stays in at 50%
  for (const v of ids) if (inRollout(v, 10)) assert.equal(inRollout(v, 50), true)
  const share = ids.filter(v => inRollout(v, 10)).length / ids.length
  assert.ok(share > 0.07 && share < 0.13, `about 10%, got ${share}`)
})

test("where Make my clips' words come from", () => {
  const s = (o: Partial<Parameters<typeof wordsSource>[0]>) => wordsSource({
    savedComplete: false, useReference: true, reference: 'pending', fullReadingRunning: false, timedOut: false, ...o,
  })
  // a finished whole-video reading is always used: nothing more is read or paid for
  assert.equal(s({ savedComplete: true, reference: 'done' }), 'saved')
  // otherwise a finished reference transcript, at once (no waiting for the whole-video reading)
  assert.equal(s({ reference: 'done', fullReadingRunning: true }), 'reference')
  // one still being made is waited for, then today's way
  assert.equal(s({ reference: 'pending' }), 'wait')
  assert.equal(s({ reference: 'pending', timedOut: true }), 'read')
  // none, or it failed: today's way (wait for the upload's reading if it runs, else read now)
  assert.equal(s({ reference: 'none', fullReadingRunning: true }), 'wait')
  assert.equal(s({ reference: 'failed' }), 'read')
  // outside the share: exactly today's way, a reference transcript is not used
  assert.equal(s({ useReference: false, reference: 'done', fullReadingRunning: true }), 'wait')
  assert.equal(s({ useReference: false, reference: 'done' }), 'read')
  assert.equal(s({ useReference: false, fullReadingRunning: true, timedOut: true }), 'read')
})

test('the parts Gemini reads for captions: each clip and a second either side, overlaps joined', () => {
  const m = (s: number, e: number) => ({ start_ms: s * 1000, end_ms: e * 1000 })
  assert.deepEqual(captionRanges([m(100, 130), m(10, 40), m(39, 70)], 600_000), [[9000, 71000], [99000, 131000]])
  // never before the start or past the end of the video
  assert.deepEqual(captionRanges([m(0, 20), m(590, 600)], 600_000), [[0, 21000], [589000, 600000]])
  assert.deepEqual(captionRanges([], 600_000), [])
})

test("the reference transcript's language, as Gemini is told it", () => {
  assert.equal(geminiLanguage('te'), 'te-IN')
  assert.equal(geminiLanguage('kn'), 'kn-IN')
  assert.equal(geminiLanguage('hi'), 'hi-IN')
  // the app's own code for Odia (transcribe.ts)
  assert.equal(geminiLanguage('or'), 'od-IN')
  // English, or one the app has no captions for: Gemini detects it itself
  assert.equal(geminiLanguage('en'), undefined)
  assert.equal(geminiLanguage('ur'), undefined)
  assert.equal(geminiLanguage(null), undefined)
})
