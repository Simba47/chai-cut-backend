// One-off: captions saved before English words were also written in the video's own script.
// For every non-English transcript, words in English letters get their native-script spelling in
// `word` (what "Auto language" shows); `word_roman` keeps the English spelling ("English").
//
//   npx tsx src/scripts/nativize-captions.ts            dry run: only reports what would change
//   npx tsx src/scripts/nativize-captions.ts --apply    writes the changes
import 'dotenv/config'
import db from '../db.js'
import { toNativeScript } from '../jobs/transcribe.js'

const apply = process.argv.includes('--apply')
const LATIN_ONLY = /^[^ऀ-ൿ]*[A-Za-z][^ऀ-ൿ]*$/

const transcripts = await db`
  SELECT t.id, t.language, t.video_id
  FROM transcripts t
  WHERE t.language IS NOT NULL AND t.language NOT LIKE 'en%' AND t.language <> 'unknown'
`
let total = 0, changed = 0
for (const t of transcripts) {
  const words = await db`SELECT id, word, word_roman, start_ms, end_ms FROM transcript_words WHERE transcript_id = ${t.id} ORDER BY start_ms`
  const latin = words.filter(w => LATIN_ONLY.test(String(w.word).trim()))
  if (!latin.length) continue
  total += latin.length
  if (!apply) { console.log(`transcript ${t.id} (${t.language}): ${latin.length} of ${words.length} words in English letters`); continue }
  const native = await toNativeScript(latin.map(w => ({ word: String(w.word), start: 0, end: 0 })), String(t.language))
  for (let i = 0; i < latin.length; i++) {
    if (!native[i]) continue
    const w = latin[i]
    await db`UPDATE transcript_words SET word = ${native[i]!}, word_roman = COALESCE(word_roman, ${String(w.word).trim().replace(/[.,!?।]+$/, '')}) WHERE id = ${w.id}`
    changed++
  }
  console.log(`transcript ${t.id} (${t.language}): ${latin.length} English-letter words, ${native.filter(Boolean).length} converted`)
}
console.log(apply ? `Done: ${changed} of ${total} words converted.` : `Dry run: ${total} words in English letters across ${transcripts.length} non-English transcripts. Run with --apply to convert.`)
await db.end()
process.exit(0)
