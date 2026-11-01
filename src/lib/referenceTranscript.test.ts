import { test } from 'node:test'
import assert from 'node:assert/strict'
import { referenceMode, parseTranscriberReply, momentAgreement, parseResultCheck } from './referenceTranscript.js'

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
