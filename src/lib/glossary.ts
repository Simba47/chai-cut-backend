import db from '../db.js'

/**
 * A brand kit's glossary (the editor's Brand kit tool: names and words captions must spell exactly
 * — the brand, its products, people). Transcription can't be told them (the speech-to-text model
 * takes no text hints), so the words it wrote are corrected after it:
 *  - words that spell a name once their spaces, case and punctuation are ignored become the name
 *    ("short cut" / "SHORTCUT" → "Shortcut"), joined into one word over their combined time;
 *  - one long word off by a letter or two becomes the name ("Shortcat" → "Shortcut"); short words
 *    are never guessed at, so everyday words stay as they are.
 * Both the word shown and its English-letter spelling (word_roman) are checked.
 */

type Row = { word: string; word_roman?: string | null; start_ms: number; end_ms: number }

const norm = (s: string) => s.toLowerCase().normalize('NFKD').replace(/[^\p{L}\p{N}]/gu, '')
const isLatin = (s: string) => /^[\p{Script=Latin}\p{N}\p{P}\p{S}\s]+$/u.test(s)
/** Leading and trailing punctuation of a word, kept around the name ("shortcut," → "Shortcut,") */
const edges = (s: string) => { const m = /^([^\p{L}\p{N}]*)(.*?)([^\p{L}\p{N}]*)$/u.exec(s)!; return [m[1], m[3]] as const }

function distance(a: string, b: string): number {
  const d = Array.from({ length: b.length + 1 }, (_, j) => j)
  for (let i = 1; i <= a.length; i++) {
    let prev = d[0]; d[0] = i
    for (let j = 1; j <= b.length; j++) {
      const t = d[j]
      d[j] = Math.min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] === b[j - 1] ? 0 : 1))
      prev = t
    }
  }
  return d[b.length]
}

/** How far a word may be from a name to be taken for it (letters off): none for short names */
const reach = (len: number) => (len >= 9 ? 2 : len >= 6 ? 1 : 0)

/** The transcript's words with the glossary's spellings (a new list; joined words take the first's place) */
export function fixSpellings<T extends Row>(rows: T[], terms: string[]): { rows: T[]; fixed: number } {
  const names = [...new Set(terms.map(t => t.trim()).filter(t => norm(t).length >= 2))]
    .map(t => ({ text: t, key: norm(t), parts: t.split(/\s+/).length }))
    // Longer names first: "Shortcut Pro" before "Shortcut"
    .sort((a, b) => b.key.length - a.key.length)
  if (!names.length || !rows.length) return { rows, fixed: 0 }
  const out: T[] = []
  let fixed = 0
  for (let i = 0; i < rows.length; i++) {
    let done = false
    for (const field of ['word', 'word_roman'] as const) {
      const at = (k: number) => String(rows[k]?.[field] ?? '')
      if (!at(i) || !isLatin(at(i))) continue
      for (const n of names) {
        // Up to as many words as the name has, plus one (a name said split: "short cut")
        let joined = ''
        for (let k = i; k < Math.min(rows.length, i + n.parts + 1); k++) {
          if (!isLatin(at(k))) break
          joined += norm(at(k))
          if (joined.length > n.key.length + 2) break
          const exact = joined === n.key
          const near = k === i && n.parts === 1 && joined.length >= 5 && reach(n.key.length) > 0 && distance(joined, n.key) <= reach(n.key.length)
          if (!exact && !near) continue
          const [pre] = edges(at(i)), [, post] = edges(at(k))
          const spelled = `${pre}${n.text}${post}`
          if (k === i && spelled === at(i)) break   // already right
          const row = { ...rows[i], end_ms: rows[k].end_ms, [field]: spelled } as T
          // A name in English letters reads the same in the shown word when that is English letters too
          if (field === 'word_roman' && isLatin(String(rows[i].word))) (row as Row).word = spelled
          if (field === 'word' && rows[i].word_roman != null && k === i) (row as Row).word_roman = spelled
          out.push(row)
          fixed++
          i = k
          done = true
          break
        }
        if (done) break
      }
      if (done) break
    }
    if (!done) out.push(rows[i])
  }
  return { rows: out, fixed }
}

/** The names in all of a video owner's brand kits' glossaries ([] when none, or no table yet) */
export async function ownerGlossary(videoId: string): Promise<string[]> {
  try {
    const rows = await db<{ terms: unknown }[]>`
      SELECT bk.data->'glossary' AS terms FROM brand_kits bk JOIN videos v ON v.user_id = bk.user_id WHERE v.id = ${videoId}`
    return [...new Set(rows.flatMap(r => (Array.isArray(r.terms) ? r.terms : []).filter((t): t is string => typeof t === 'string')))].slice(0, 200)
  } catch {
    return []
  }
}
