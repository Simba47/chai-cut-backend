import { test } from 'node:test'
import assert from 'node:assert/strict'

// A fake Gemini API (no network): the API-key route, with at most 3 requests in flight
process.env.AI_MAX_CONCURRENT = '3'
process.env.GEMINI_API_KEY = 'test-key'
delete process.env.GOOGLE_CLOUD_PROJECT
const { askGemini } = await import('./gemini.js')

const answer = (text: string) => new Response(JSON.stringify({ candidates: [{ content: { parts: [{ text }] } }] }), { status: 200 })

test('one process never has more requests in flight than AI_MAX_CONCURRENT, and every request is answered', async () => {
  let open = 0, most = 0
  globalThis.fetch = (async (_url: string, init: { body: string }) => {
    most = Math.max(most, ++open)
    await new Promise(r => setTimeout(r, 10))
    open--
    return answer(JSON.parse(init.body).contents[0].parts[0].text)
  }) as typeof fetch
  const asks = Array.from({ length: 25 }, (_, i) => askGemini(`q${i}`))
  assert.deepEqual(await Promise.all(asks), Array.from({ length: 25 }, (_, i) => `q${i}`))
  assert.equal(most, 3)
})

test('a busy answer (429) is tried again, with Retry-After respected', async () => {
  let calls = 0
  const at: number[] = []
  globalThis.fetch = (async () => {
    at.push(Date.now())
    return ++calls === 1 ? new Response('busy', { status: 429, headers: { 'retry-after': '2' } }) : answer('ok')
  }) as typeof fetch
  assert.equal(await askGemini('q'), 'ok')
  assert.equal(calls, 2)
  assert.ok(at[1] - at[0] >= 1900, `waited ${at[1] - at[0]} ms`)
})

test('a refused request (400) is not retried', async () => {
  let calls = 0
  globalThis.fetch = (async () => { calls++; return new Response('bad', { status: 400 }) }) as typeof fetch
  await assert.rejects(askGemini('q'), /Gemini 400/)
  assert.equal(calls, 1)
})
