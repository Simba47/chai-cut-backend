/**
 * A brand kit's words (brand kit rebuild, Phase 5, 2026-10-28): how the brand writes, kept with the
 * kit (BrandKitData.words) —
 *  - spellings: names captions must spell the brand's way ("short cut" / "SHORTCUT" → "ShortCut"),
 *    each with what it's often heard as ("chai cut" → "ChaiCut")
 *  - hidden words: never shown in captions or post text as they are ("damn" → "d***")
 *  - lines: the brand's calls to action ("Follow for more"), the first one ending its post captions when it asks
 *  - hashtags every post gets, and the brand's tone (ready lines to start from; how post text sounds)
 *
 * Nothing here is paid for: plain rules over the words we already have. Captions aren't changed where
 * they're saved — the words are spelled as they're read (the editor, a clip's preview, the export), so
 * a kit's new spelling reaches every clip at once, old ones too.
 *
 * Copied as-is between the repos (chai-cut-backend/src/lib/brandWords.ts, chai-cut-frontend
 * src/lib/brandWords.ts): the worker spells the captions it exports and the post text it writes, the
 * app the ones it shows. Keep the two identical. No imports.
 */

export interface BrandSpelling {
  /** The name as the brand writes it */
  text: string
  /** What it's often heard as (other spellings or words), each made the name too */
  heard: string[]
}

export type BrandTone = 'friendly' | 'bold' | 'expert' | 'fun' | 'calm'

export interface BrandWords {
  spell: BrandSpelling[]
  /** Words and phrases never shown as they are */
  hide: string[]
  /** Calls to action, the main one first */
  lines: string[]
  /** Whether the main line ends every post caption written for a clip */
  end_posts: boolean
  /** Without the # */
  hashtags: string[]
  tone: BrandTone | null
}

export const MAX_SPELLINGS = 50
export const MAX_HEARD = 5
export const MAX_HIDDEN = 50
export const MAX_LINES = 6
export const MAX_HASHTAGS = 10
/** How long a name, a hidden word, a line and a hashtag can be */
export const WORD_MAX = 40
export const LINE_MAX = 80

export const NO_BRAND_WORDS: BrandWords = { spell: [], hide: [], lines: [], end_posts: false, hashtags: [], tone: null }

/** The tones a brand can pick: how its post text sounds (`prompt`), and ready lines to start from */
export const TONES: Array<{ id: BrandTone; name: string; prompt: string; lines: string[] }> = [
  { id: 'friendly', name: 'Friendly', prompt: 'warm, friendly and simple, like talking to a friend',
    lines: ['Follow for more 🙌', 'Save this for later', 'Tag a friend who needs this', 'Share this with someone you love'] },
  { id: 'bold', name: 'Bold', prompt: 'bold, punchy and confident, in short sentences',
    lines: ['Follow now — don’t miss the next one', 'Hit follow for more', 'Comment if you agree', 'Send this to your crew'] },
  { id: 'expert', name: 'Expert', prompt: 'clear, expert and professional, with no slang',
    lines: ['Follow for more insights', 'Save this for reference', 'Link in bio to learn more', 'Share this with your team'] },
  { id: 'fun', name: 'Fun', prompt: 'fun, playful and full of energy',
    lines: ['Follow for more fun 😄', 'Tag your bestie 👀', 'Drop a 🔥 if you agree', 'Part 2? Comment below'] },
  { id: 'calm', name: 'Calm', prompt: 'calm, kind and encouraging',
    lines: ['Follow for a little calm every day', 'Save this for a hard day', 'Share it with someone who needs it', 'Breathe. Follow for more.'] },
]

/** A word with its case, spaces, accents and punctuation left out: what spellings are matched on */
export const wordKey = (s: string) =>
  // (accents on English letters go; the vowel signs of Telugu, Hindi… are part of the word)
  s.toLowerCase().normalize('NFKD').replace(/(\p{Script=Latin})\p{M}+/gu, '$1').replace(/[^\p{L}\p{N}\p{M}]/gu, '')
const isLatin = (s: string) => /^[\p{Script=Latin}\p{N}\p{P}\p{S}\s]+$/u.test(s)
const oneLine = (v: unknown, max: number) => (typeof v === 'string' ? v.replace(/\s+/g, ' ').trim().slice(0, max).trim() : '')
const hashtagOf = (v: unknown) => (typeof v === 'string' ? v.replace(/^#+/, '').replace(/[^\p{L}\p{N}_]/gu, '').slice(0, WORD_MAX) : '')

/** A kit's saved words made safe: anything missing or wrong left out */
export function cleanBrandWords(raw: unknown): BrandWords {
  const d = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
  const list = (v: unknown) => (Array.isArray(v) ? v : [])
  const seen = new Set<string>()
  const spell: BrandSpelling[] = []
  for (const s of list(d.spell)) {
    const e = (s && typeof s === 'object' ? s : {}) as Record<string, unknown>
    const text = oneLine(e.text, WORD_MAX), key = wordKey(text)
    if (key.length < 2 || seen.has(key)) continue
    seen.add(key)
    const heardSeen = new Set([key])
    const heard = list(e.heard).map(h => oneLine(h, WORD_MAX))
      .filter(h => { const k = wordKey(h); if (k.length < 2 || heardSeen.has(k)) return false; heardSeen.add(k); return true })
      .slice(0, MAX_HEARD)
    spell.push({ text, heard })
    if (spell.length >= MAX_SPELLINGS) break
  }
  const unique = (items: string[], max: number, key: (s: string) => string = wordKey) => {
    const got = new Set<string>()
    return items.filter(x => { const k = key(x); if (!k || got.has(k)) return false; got.add(k); return true }).slice(0, max)
  }
  return {
    spell,
    hide: unique(list(d.hide).map(h => oneLine(h, WORD_MAX)).filter(h => wordKey(h).length >= 2), MAX_HIDDEN),
    lines: unique(list(d.lines).map(l => oneLine(l, LINE_MAX)), MAX_LINES, l => l.toLowerCase()),
    end_posts: d.end_posts === true,
    hashtags: unique(list(d.hashtags).map(hashtagOf), MAX_HASHTAGS, h => h.toLowerCase()),
    tone: TONES.some(t => t.id === d.tone) ? d.tone as BrandTone : null,
  }
}

/** Whether a kit's words change anything */
export const hasBrandWords = (w: BrandWords | null | undefined): w is BrandWords =>
  !!w && (w.spell.length > 0 || w.hide.length > 0 || w.lines.length > 0 || w.hashtags.length > 0 || !!w.tone)

// ── Captions ─────────────────────────────────────────────────────────────────

type Row = { word: string; word_roman?: string | null; start_ms: number; end_ms: number }

/** Leading and trailing punctuation of a word, kept around what replaces it ("shortcut," → "ShortCut,") */
const edges = (s: string) => { const m = /^([^\p{L}\p{N}]*)([\s\S]*?)([^\p{L}\p{N}\p{M}]*)$/u.exec(s)!; return [m[1], m[2], m[3]] as const }

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

/** Words said further apart than this are never one name */
const JOIN_GAP_MS = 600
/** Endings a name keeps ("ShortCut" + "s", "'s", "es") */
const SUFFIXES = new Set(['s', 'es'])
/** How many letters a long name may be off by and still be taken for it (none under 7: everyday words stay) */
const reach = (len: number) => (len >= 11 ? 2 : len >= 7 ? 1 : 0)

interface Key { text: string; key: string; parts: number; own: boolean }
function keysOf(spell: BrandSpelling[]): Key[] {
  return spell.flatMap(s => [
    { text: s.text, key: wordKey(s.text), parts: s.text.split(' ').length, own: true },
    ...s.heard.map(h => ({ text: s.text, key: wordKey(h), parts: h.split(' ').length, own: false })),
  ]).filter(k => k.key.length >= 2)
    // Longer first: "ShortCut Pro" before "ShortCut"
    .sort((a, b) => b.key.length - a.key.length)
}

/**
 * The words with the brand's spellings (a new list): words that spell a name once case, spaces and
 * punctuation are left out become it ("short cut" → "ShortCut", joined into one word over their time);
 * what it's heard as becomes it too; with an ending kept ("shortcuts" → "ShortCuts"); and one long word
 * a letter off becomes it ("Shortcat" → "ShortCut"). The word shown and its English-letter spelling
 * (word_roman) are both checked; only words in English letters are.
 */
export function spellWords<T extends Row>(rows: T[], spell: BrandSpelling[]): { rows: T[]; fixed: number } {
  const keys = keysOf(spell)
  if (!keys.length || !rows.length) return { rows, fixed: 0 }
  const out: T[] = []
  let fixed = 0
  for (let i = 0; i < rows.length; i++) {
    let done = false
    for (const field of ['word', 'word_roman'] as const) {
      const at = (k: number) => String(rows[k]?.[field] ?? '')
      if (!at(i) || !isLatin(at(i))) continue
      for (const n of keys) {
        let joined = ''
        // Up to as many words as the name has, plus one (a name said split: "short cut")
        for (let k = i; k < Math.min(rows.length, i + n.parts + 1); k++) {
          if (k > i && (!isLatin(at(k)) || rows[k].start_ms - rows[k - 1].end_ms > JOIN_GAP_MS)) break
          joined += wordKey(at(k))
          if (joined.length > n.key.length + 2) break
          const rest = joined.startsWith(n.key) ? joined.slice(n.key.length) : null
          const exact = rest === ''
          const ending = !!rest && n.key.length >= 3 && SUFFIXES.has(rest)
          const near = !exact && !ending && k === i && n.parts === 1 && n.own && joined.length >= 5 && joined[0] === n.key[0]
            && reach(n.key.length) > 0 && distance(joined, n.key) <= reach(n.key.length)
          if (!exact && !ending && !near) continue
          const [pre] = edges(at(i)), [, core, post] = edges(at(k))
          // (the ending as it was written: "'s", "s", "es")
          const kept = ending ? new RegExp(`['’]?\\p{L}{${rest!.length}}$`, 'u').exec(core)?.[0] ?? rest! : ''
          const spelled = `${pre}${n.text}${kept}${post}`
          if (k === i && spelled === at(i)) { done = true; out.push(rows[i]); break }   // already right
          const row = { ...rows[i], end_ms: rows[k].end_ms, [field]: spelled } as T
          if (field === 'word_roman') {
            // (the word shown: the name too when it's in English letters; else the words said, together)
            ;(row as Row).word = isLatin(String(rows[i].word)) ? spelled : rows.slice(i, k + 1).map(r => r.word).join(' ')
          } else if (rows[i].word_roman != null) (row as Row).word_roman = spelled
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

/** A word hidden: its first letter, then stars ("damn" → "d***"); a single letter, or other letters: all stars */
export function maskWord(s: string): string {
  const [pre, core, post] = edges(s)
  if (!core) return s
  const chars = Array.from(core)
  const masked = !isLatin(core) ? '***' : chars.length === 1 ? '*' : chars[0] + '*'.repeat(Math.max(2, Math.min(5, chars.length - 1)))
  return `${pre}${masked}${post}`
}

/** The words with the brand's hidden words and phrases masked (each word of a phrase; both spellings of a word) */
export function hideWords<T extends Row>(rows: T[], hide: string[]): { rows: T[]; hidden: number } {
  const keys = hide.map(h => ({ key: wordKey(h), parts: h.split(' ').length })).filter(k => k.key.length >= 2)
    .sort((a, b) => b.key.length - a.key.length)
  if (!keys.length || !rows.length) return { rows, hidden: 0 }
  const out = [...rows]
  let hidden = 0
  for (let i = 0; i < rows.length; i++) {
    let to = -1
    for (const field of ['word', 'word_roman'] as const) {
      const at = (k: number) => wordKey(String(rows[k]?.[field] ?? ''))
      if (!at(i)) continue
      for (const h of keys) {
        let joined = ''
        for (let k = i; k < Math.min(rows.length, i + h.parts); k++) {
          joined += at(k)
          if (joined === h.key) { to = k; break }
          if (!h.key.startsWith(joined)) break
        }
        if (to >= 0) break
      }
      if (to >= 0) break
    }
    if (to < 0) continue
    for (let k = i; k <= to; k++) {
      const r = rows[k]
      out[k] = { ...r, word: maskWord(String(r.word)), ...(r.word_roman != null ? { word_roman: maskWord(String(r.word_roman)) } : {}) }
    }
    hidden++
    i = to
  }
  return { rows: out, hidden }
}

/** Caption words as the brand writes them: its spellings, then its hidden words masked (unchanged without any) */
export function brandCaptionWords<T extends Row>(rows: T[], w: BrandWords | null | undefined): T[] {
  if (!w || (!w.spell.length && !w.hide.length)) return rows
  return hideWords(spellWords(rows, w.spell).rows, w.hide).rows
}

// ── Written text (a post caption, a hook, a title) ──────────────────────────

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
const NOT_WORD = '(?<![\\p{L}\\p{N}])'
const NOT_WORD_AFTER = '(?![\\p{L}\\p{N}])'

/** Text as the brand writes it: its spellings (an ending kept), its hidden words masked */
export function brandText(text: string, w: BrandWords | null | undefined): string {
  if (!w || !text) return text
  let out = text
  for (const k of keysOf(w.spell)) {
    // (the name's letters, any space, dash, dot or apostrophe between them: "short-cut", "Short Cut")
    const body = Array.from(k.key).map(escape).join('[\\s\\-_.\'’]?')
    out = out.replace(new RegExp(`${NOT_WORD}${body}((?:['’]?s|es)?)${NOT_WORD_AFTER}`, 'giu'), (_m, ending: string) => `${k.text}${ending}`)
  }
  for (const h of w.hide) {
    const parts = h.split(' ').map(p => wordKey(p)).filter(Boolean)
    if (!parts.length) continue
    const body = parts.map(p => Array.from(p).map(escape).join('[\'’]?')).join('[\\s\\-_.]+')
    out = out.replace(new RegExp(`${NOT_WORD}${body}${NOT_WORD_AFTER}`, 'giu'), m => m.split(/([\s\-_.]+)/).map((p, i) => (i % 2 ? p : maskWord(p))).join(''))
  }
  return out
}

/** Hashtags as the brand has them: its own first, then the others — none hidden, none twice, up to `max` */
export function brandHashtags(tags: string[], w: BrandWords | null | undefined, max = 10): string[] {
  const hidden = new Set((w?.hide ?? []).map(wordKey))
  const seen = new Set<string>()
  return [...(w?.hashtags ?? []), ...tags.map(hashtagOf)]
    .filter(t => { const k = t.toLowerCase(); if (!t || seen.has(k) || hidden.has(wordKey(t))) return false; seen.add(k); return true })
    .slice(0, max)
}

/** A clip's written text (lib/clipText.ts ClipText) as the brand writes it */
export function brandClipText<T extends { hook: string; title: string; post_caption: string; hashtags: string[] }>(t: T, w: BrandWords | null | undefined): T {
  if (!hasBrandWords(w)) return t
  return { ...t, hook: brandText(t.hook, w), title: brandText(t.title, w), post_caption: brandText(t.post_caption, w), hashtags: brandHashtags(t.hashtags, w) }
}

/** The brand's voice for the post text the app writes anyway (lib/clipText.ts BrandVoice); null: nothing to say */
export function brandVoice(w: BrandWords | null | undefined): { tone: string; avoid: string; hashtags: string[]; cta: string } | null {
  if (!hasBrandWords(w)) return null
  return {
    tone: TONES.find(t => t.id === w.tone)?.prompt ?? '',
    avoid: w.hide.join(', '),
    hashtags: w.hashtags,
    cta: w.end_posts ? w.lines[0] ?? '' : '',
  }
}
