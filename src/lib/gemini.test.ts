import { test } from 'node:test'
import assert from 'node:assert/strict'
import { retryAfterMs, retryDelayMs } from './gemini.js'

test('retry waits grow, stay under 30 s, and are spread out', () => {
  for (let attempt = 1; attempt <= 8; attempt++) {
    const cap = Math.min(30_000, 2000 * 2 ** (attempt - 1))
    const waits = Array.from({ length: 200 }, () => retryDelayMs(attempt))
    assert.ok(waits.every(w => w >= cap / 2 && w <= cap), `attempt ${attempt}`)
    // Not all the same: workers refused together come back apart
    assert.ok(new Set(waits.map(w => Math.round(w))).size > 50, `attempt ${attempt} spread`)
  }
})

test("Google's hint is never cut short", () => {
  assert.ok(retryDelayMs(1, 20_000) >= 20_000)
})

test('Retry-After: seconds, HTTP dates, junk', () => {
  assert.equal(retryAfterMs(null), 0)
  assert.equal(retryAfterMs('7'), 7000)
  assert.equal(retryAfterMs('600'), 60_000)
  assert.equal(retryAfterMs('soon'), 0)
  const inTen = new Date(Date.now() + 10_000).toUTCString()
  assert.ok(Math.abs(retryAfterMs(inTen) - 10_000) < 1500)
})
