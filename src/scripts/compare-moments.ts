// Does Gemini pick the same best moments from a rough transcript as from its own?
// Runs the app's own moment finder (findClips, with selectHighlights' settings) on each transcript
// of the same video and compares the picks. Local test only; nothing is saved to the database.
//
//   set -a; . ~/shortcut-rig/rig.env; set +a
//   npx tsx src/scripts/compare-moments.ts <out.json> gemini=<words.json> gemini2=<words.json> indic=<words.json> ...
// Each transcript is a JSON array of {word, start_ms, end_ms} (or {words: [...]}).
// The first one is the baseline; list it twice under two names to see how much picks vary by chance.
import { readFileSync, writeFileSync } from 'node:fs'
import { askGemini } from '../lib/gemini.js'
import { findClips, pickSpread, buildLines, makeWindows, type FoundClip, type FinderWord } from '../lib/clipFinder.js'

const [outPath, ...specs] = process.argv.slice(2)
if (!outPath || specs.length < 2) {
  console.error('usage: compare-moments.ts <out.json> name=<words.json> name2=<words.json> ...')
  process.exit(1)
}

function load(path: string): FinderWord[] {
  const raw = JSON.parse(readFileSync(path, 'utf8'))
  const words = (Array.isArray(raw) ? raw : raw.words) as FinderWord[]
  return words.filter(w => w.word && Number.isFinite(w.start_ms) && Number.isFinite(w.end_ms))
}

/** Same moment: they share more than half of the shorter one */
const same = (a: FoundClip, b: FoundClip) =>
  Math.min(a.end_ms, b.end_ms) - Math.max(a.start_ms, b.start_ms) > 0.5 * Math.min(a.end_ms - a.start_ms, b.end_ms - b.start_ms)

/** Share of `base` moments that `other` also found */
const found = (base: FoundClip[], other: FoundClip[]) => base.filter(b => other.some(o => same(b, o))).length / Math.max(1, base.length)

const mmss = (ms: number) => `${Math.floor(ms / 60000)}:${String(Math.floor((ms % 60000) / 1000)).padStart(2, '0')}`

const results: Record<string, { all: FoundClip[]; top10: FoundClip[]; seconds: number }> = {}
for (const spec of specs) {
  const [name, path] = spec.split('=')
  const words = load(path)
  const durationMs = Math.max(...words.map(w => w.end_ms))
  const t0 = Date.now()
  // The same settings as selectHighlights in jobs/ai_edit.ts
  const all = await findClips({
    words, durationMs, mode: { kind: 'best' }, ask: (system, user) => askGemini(user, { system }),
    limit: 60, perWindow: 12, spread: true, log: m => console.warn(`[${name}]`, m),
  })
  const top10 = pickSpread(all, 10, Math.max(1, makeWindows(buildLines(words)).length))
  results[name] = { all, top10, seconds: Math.round((Date.now() - t0) / 1000) }
  console.log(`${name}: ${words.length} words -> ${all.length} moments in ${results[name].seconds}s`)
}

const names = Object.keys(results)
const base = names[0]
console.log(`\nShare of the baseline (${base}) moments that each transcript also found:`)
console.log('transcript'.padEnd(14) + 'top 10'.padStart(8) + 'all moments'.padStart(14))
for (const n of names.slice(1)) {
  console.log(n.padEnd(14) + `${Math.round(found(results[base].top10, results[n].top10) * 100)}%`.padStart(8)
    + `${Math.round(found(results[base].all, results[n].all) * 100)}%`.padStart(14))
}
console.log(`\nIs each top-10 pick somewhere in the baseline's full list (${results[base].all.length} moments)?`)
for (const n of names.slice(1)) console.log(`${n.padEnd(14)} ${Math.round(found(results[n].top10, results[base].all) * 100)}%`)

for (const n of names) {
  console.log(`\n── ${n}: top 10 ──`)
  for (const c of results[n].top10) {
    const inBase = n === base ? '' : results[base].all.some(b => same(c, b)) ? '  ✓ baseline has it' : '  ✗ new'
    console.log(`  ${mmss(c.start_ms)}-${mmss(c.end_ms)}  score ${c.score}  ${c.title}${inBase}`)
  }
}
writeFileSync(outPath, JSON.stringify(results, null, 1))
console.log(`\nsaved ${outPath}`)
