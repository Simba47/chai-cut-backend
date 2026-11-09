// Video names from Drive/Dropbox links (user, 2026-11-10: emojis and ｜ came out as "ð..." and "ï½.")
// Run: npx tsx --test src/lib/fileName.test.ts
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { filenameFromDisposition, titleFromFilename } from './fileName.js'

const asFetchReadsIt = (s: string) => Buffer.from(s, 'utf8').toString('latin1')
const NAME = 'Anchor Pradeep Interview With Nani 😂🔥 ｜ Kayadu Lohar ｜ The Paradise.mp4'

test('a UTF-8 name read as Latin-1 comes back whole', () => {
  assert.equal(filenameFromDisposition(asFetchReadsIt(`attachment; filename="${NAME}"`)), NAME)
})

test('a name that really is Latin-1 stays as it came', () => {
  assert.equal(filenameFromDisposition('attachment; filename="caf\u00e9 night.mp4"'), 'café night.mp4')
})

test('plain and filename* names read as before', () => {
  assert.equal(filenameFromDisposition('attachment; filename="talk.mp4"'), 'talk.mp4')
  assert.equal(filenameFromDisposition('attachment; filename=talk.mp4'), 'talk.mp4')
  assert.equal(filenameFromDisposition(`attachment; filename="x.mp4"; filename*=UTF-8''%F0%9F%98%82%20talk.mp4`), '😂 talk.mp4')
  assert.equal(filenameFromDisposition('attachment'), null)
  assert.equal(filenameFromDisposition(null), null)
})

test('the title drops the extension and counts an emoji as one character', () => {
  assert.equal(titleFromFilename(NAME), 'Anchor Pradeep Interview With Nani 😂🔥 ｜ Kayadu Lohar ｜ The Paradise')
  const long = `${'a'.repeat(119)}😂😂.mp4`
  assert.equal(titleFromFilename(long), `${'a'.repeat(119)}😂`)
  assert.equal(titleFromFilename(null), undefined)
})
