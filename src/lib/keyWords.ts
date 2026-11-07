/**
 * A clip's key words: the few words its captions make stand out (caption_styles.emphasis,
 * { "<word start_ms>": true }), and its line words (caption_styles.line_keys, the same shape): the
 * word a Creator style makes big on a line with no key word (creator.ts creatorKeyIndex). Asked of
 * AI in one go, once per clip (or again from the Key words tab).
 *
 * The SAME file in both repos (frontend src/lib/keyWords.ts for the editor, backend
 * src/lib/keyWords.ts for the export's first render): keep the two identical.
 *
 * Until 2026-11-06 AI was given the clip as 5-word pieces and told to pick 1–2 words in each, by
 * number, with no thinking: it marked over half the lines, often with filler ("a", "in", "sir",
 * "film") or a word next to the one it meant (user: key words should be what matters in context).
 * Now it reads the clip's sentences (and the video's title), picks only what matters — names,
 * numbers, references, the main idea, the punchline — about one for every three caption lines,
 * and each pick names its word, so a pick whose number points elsewhere is moved to its word or
 * left out. Line words (2026-11-07): one stressed word in every short phrase, so a Creator line
 * with no key word shows a word that means something, not just its longest one.
 */

export interface KeyWordSource { word: string; start_ms: number; end_ms: number }
export type AskForKeyWords = (prompt: string, opts: { schema: object; temperature: number; thinkingBudget: number }) => Promise<string>
/** Words marked, by the word's start_ms */
export type WordMarks = Record<string, true>
export interface KeyWordPicks { emphasis: WordMarks; line_keys: WordMarks }

const PICKS = { type: 'ARRAY', items: { type: 'OBJECT', properties: { i: { type: 'INTEGER' }, text: { type: 'STRING' } }, required: ['i', 'text'] } }
export const KEY_WORDS_SCHEMA = {
  type: 'OBJECT',
  properties: { keys: PICKS, line: PICKS },
  required: ['keys', 'line'],
}

/** About one key word for every three caption lines (a line: ~4 words) */
export const WORDS_PER_KEY = 12
const MAX_KEYS = 40
/** About one line word for every short phrase */
export const WORDS_PER_LINE_WORD = 2.5
/** A pick is a word, or a name or a number with its unit of at most this many words */
const MAX_PICK_WORDS = 3
/** A pick's number may be this many words off its word */
const NEAR = 3
/** A sentence ends at its punctuation, a pause this long (ms), or this many words */
const SENTENCE_PAUSE_MS = 700
const SENTENCE_MAX_WORDS = 25
/** Never a key word or line word on its own, whatever AI says: little words and forms of address */
const FILLER = new Set([
  'a', 'an', 'the', 'in', 'on', 'of', 'to', 'at', 'by', 'for', 'and', 'or', 'but', 'so', 'is', 'are', 'was', 'were', 'be', 'it', 'its',
  'this', 'that', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'my', 'your', 'very', 'just', 'like', 'okay', 'ok', 'yeah', 'right',
  'um', 'uh', 'sir', 'madam', 'garu', 'gaaru', 'ji', 'anna', 'andi', 'ante', 'ani', 'kuda', 'inka',
])

/** A word as it is compared: small letters, no punctuation */
const plain = (s: string) => s.normalize('NFC').toLowerCase().replace(/[^\p{L}\p{N}\p{M}%]+/gu, '')

/** How many key words a clip of `n` words gets at most */
export const keyWordTarget = (n: number) => Math.max(1, Math.min(MAX_KEYS, Math.round(n / WORDS_PER_KEY)))
/** How many line words a clip of `n` words gets at most */
export const lineWordTarget = (n: number) => Math.max(1, Math.ceil(n / WORDS_PER_LINE_WORD))

/** The clip's words as sentences, each word written as index=word */
function sentences(words: KeyWordSource[]): string {
  const out: number[][] = []
  words.forEach((w, i) => {
    const cur = out[out.length - 1]
    const prev = words[i - 1]
    if (!cur || cur.length >= SENTENCE_MAX_WORDS || w.start_ms - prev.end_ms > SENTENCE_PAUSE_MS || /[.!?।]$/.test(prev.word.trim())) out.push([i])
    else cur.push(i)
  })
  return out.map(s => s.map(i => `${i}=${words[i].word}`).join(' ')).join('\n')
}

export function keyWordsPrompt(words: KeyWordSource[], title: string | null): string {
  const n = keyWordTarget(words.length)
  return `These are the captions of a short video clip${title ? ` from a video titled "${title}"` : ''} (any language, often Telugu, Hindi or Tamil mixed with English), one sentence a line, each word written as index=word.

Read the whole clip first and understand what it is about. Then make two lists.

"keys": the words that matter most in the clip's context — the ones a viewer must not miss:
- names of people, films, places, brands and teams
- numbers, amounts, dates and records
- the main idea or subject of the clip, and a reference that carries it
- the punchline or the one strong word of an emotional moment
Never pick filler or plain words: articles, pronouns, "very", "so", "in", "all", forms of address ("sir", "garu", "ji", "anna"), or general words like "film", "movie", "thing", "people" unless the clip is about exactly that.
Each pick is ONE word. Only a name ("Allu Arjun", "Tamil Nadu") or a number with its unit ("10 crores") is two or three words: give its first word's index and the whole of it. Never a phrase of plain words ("a box office", "good films").
Pick at most ${n}, fewer if fewer really stand out. Most lines have none. Never the same name or word twice: once, where it matters most.

"line": go through the whole clip, start to end, in steps of 2 or 3 words, and give the ONE word in each step that carries its meaning — the word a speaker would stress. Every step gets one, also those with a key word, so about one word in every ${WORDS_PER_LINE_WORD} is a line word and no stretch of 3 words is left without one. Never filler or a form of address; a name or a number with its unit may be two or three words, as above.

Return ONLY JSON: {"keys": [{"i": <index of its first word>, "text": "<the word or name exactly as written>"}], "line": [same form]}

${sentences(words)}`
}

/**
 * The picks of one list in AI's answer, each checked against the words it names. `once`: the same
 * word or name only once (key words); `limit`: at most this many picks.
 */
function readPicks(list: unknown, words: KeyWordSource[], limit: number, once: boolean): WordMarks {
  const picked: WordMarks = {}
  if (!Array.isArray(list)) return picked
  const seen = new Set<string>()
  let count = 0
  for (const k of list) {
    if (count >= limit) break
    let i = Number((k as { i?: unknown })?.i)
    const parts = String((k as { text?: unknown })?.text ?? '').trim().split(/\s+/).map(plain).filter(Boolean)
    // (no little word at either end: "a magical combination" is "magical combination")
    while (parts.length && FILLER.has(parts[0])) { parts.shift(); i++ }
    while (parts.length && FILLER.has(parts[parts.length - 1])) parts.pop()
    if (!Number.isInteger(i) || !parts.length || parts.length > MAX_PICK_WORDS) continue
    // (the same word or name once: where AI put it first)
    if (once && seen.has(parts.join(' '))) continue
    // (at its number, else the nearest place within NEAR words where those words are)
    const fits = (at: number) => at >= 0 && at + parts.length <= words.length && parts.every((p, j) => plain(words[at + j].word) === p)
    const at = [0, ...Array.from({ length: NEAR }, (_, d) => [d + 1, -(d + 1)]).flat()].map(d => i + d).find(fits)
    if (at === undefined) continue
    // (marks are saved by start time, and about 1.5% of words start with their neighbour: marking
    // "magical" would mark the "a" that starts with it too, and a Creator style would make "a" the big
    // word. Such a pick is left out, unless the neighbour is part of it)
    const shares = words.some((w, j) => (j < at || j >= at + parts.length)
      && parts.some((_, n) => words[at + n].start_ms === w.start_ms))
    if (shares) continue
    for (let j = 0; j < parts.length; j++) picked[String(words[at + j].start_ms)] = true
    seen.add(parts.join(' '))
    count++
  }
  return picked
}

/** The key words and line words in AI's answer, each checked against the words it names; null when the answer can't be read */
export function readKeyWords(answer: string, words: KeyWordSource[]): KeyWordPicks | null {
  const match = answer.match(/\{[\s\S]*\}/)
  let out: { keys?: unknown; line?: unknown } | null
  try { out = match ? JSON.parse(match[0]) : null } catch { return null }
  if (!out || !Array.isArray(out.keys)) return null
  return {
    emphasis: readPicks(out.keys, words, Math.ceil(keyWordTarget(words.length) * 1.25), true),
    line_keys: readPicks(out.line, words, lineWordTarget(words.length) * 2, false),
  }
}

/** The clip's key words and line words, asked of AI (`ask`: askGemini); null when it fails, so the captions go without */
export async function pickKeyWords(words: KeyWordSource[], title: string | null, ask: AskForKeyWords): Promise<KeyWordPicks | null> {
  if (!words.length) return null
  // (a little thinking: it reads the clip before it picks)
  const answer = await ask(keyWordsPrompt(words, title), { schema: KEY_WORDS_SCHEMA, temperature: 0, thinkingBudget: 1024 })
  return readKeyWords(answer, words)
}
