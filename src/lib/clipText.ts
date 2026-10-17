/**
 * Words to post a clip with: a hook shown over its first 3 seconds, a title, a post caption and
 * hashtags, written by Gemini from what is said in the clip.
 *
 * Copied as-is between the repos (chai-cut-backend/src/lib/clipText.ts, chai-cut-frontend
 * src/lib/clipText.ts): the worker writes them for "Make my clips", the app's Regenerate button
 * rewrites them. Keep the two identical.
 *
 * No framework or AI imports: the model call is passed in (`ask`, see lib/gemini.ts askGemini).
 */

// The same rule as clipFinder.ts ROMAN_RULE: text in Roman letters (Tenglish, Hinglish…)
const ROMAN_RULE = `Write the text in Roman (English) letters, in the speaker's own language the way people type it in chats: Telugu as Tenglish (e.g. "Ee cinema ki audience pichi ekkaru!"), Hindi as Hinglish (e.g. "Yeh scene dekh ke sab hil gaye!"), Tamil as Tanglish, and so on; keep English words as they are, and if the speaker talks in English write English. Never use Telugu, Devanagari, Tamil or any other non-Latin script.`

// The other choices for the hook, title and post caption (Make my clips: "Title in")
const ENGLISH_RULE = `Write the text in English: say what the speaker means in natural, simple English, the way an Indian creator would write it. Keep the names of people, films and places as they are.`
const NATIVE_RULE = `Write the text in the speaker's own language and in that language's own script (Telugu in Telugu letters, Hindi in Devanagari, and so on), the way a native speaker would write it. English words the speaker used can stay in English letters.`

/** What the hook, title and post caption are written in: English letters (Tenglish, Hinglish…), English, or the speaker's own script */
export type ClipTextLanguage = 'roman' | 'english' | 'native'

export interface ClipText { hook: string; title: string; post_caption: string; hashtags: string[] }

/**
 * A brand kit's voice (the editor's Brand kit tool): how the text should sound, words it never
 * uses, hashtags every post gets, and a closing line every post caption ends with.
 */
export interface BrandVoice { tone: string; avoid: string; hashtags: string[]; cta: string }

/** The prompt lines for a brand voice ('' without one) */
function voiceRules(v: BrandVoice | null | undefined): string {
  if (!v) return ''
  const lines = [
    v.tone.trim() && `- Brand voice: write it ${v.tone.trim().replace(/[.\s]+$/, '')}.`,
    v.avoid.trim() && `- Never use these words or phrases: ${v.avoid.trim()}.`,
    v.cta.trim() && `- The post caption gets this closing line added after it, so don't write a call to action of your own: "${v.cta.trim()}".`,
  ].filter(Boolean)
  return lines.length ? `\n${lines.join('\n')}` : ''
}

/** The brand's own hashtags first (always on the post), then the written ones, no repeats */
export function withBrandHashtags(tags: string[], v: BrandVoice | null | undefined): string[] {
  const own = (v?.hashtags ?? []).map(h => h.replace(/^#+/, '').replace(/\s+/g, '')).filter(Boolean)
  const seen = new Set<string>()
  return [...own, ...tags].filter(t => { const k = t.toLowerCase(); if (seen.has(k)) return false; seen.add(k); return true }).slice(0, 10)
}
type TextWord = { word: string; word_roman?: string | null; start_ms: number; end_ms: number }

const SCRIPT_FONTS: Array<[RegExp, string]> = [
  [/[ఀ-౿]/, 'noto-sans-telugu'],
  [/[ऀ-ॿ]/, 'noto-sans-devanagari'],
]

/**
 * The bundled font (render.py _FONT_FILES) for text in this script; bold Montserrat for
 * anything else. Scripts with no bundled font (Tamil, Kannada…) fall back too.
 */
export function fontForText(text: string): string {
  const counts = SCRIPT_FONTS.map(([re, font]) => ({ font, n: [...text].filter(ch => re.test(ch)).length }))
  const best = counts.sort((a, b) => b.n - a.n)[0]
  return best && best.n > 0 ? best.font : 'montserrat-bold'
}

/** The clip's speech as plain lines (a new line after a 700 ms pause or 20 words) */
function clipLines(words: TextWord[]): string {
  const lines: string[][] = []
  words.forEach((w, i) => {
    const cur = lines[lines.length - 1]
    if (!cur || cur.length >= 20 || w.start_ms - words[i - 1].end_ms > 700) lines.push([w.word])
    else cur.push(w.word)
  })
  return lines.map(l => l.join(' ')).join('\n')
}

const clean = (v: unknown, max: number) => (typeof v === 'string' ? v.replace(/\s+/g, ' ').trim().slice(0, max) : '')

/** Sends one prompt to the model and returns its raw text answer */
export type AskText = (prompt: string) => Promise<string>

/** Hook, title, post caption and hashtags for a clip; throws if Gemini fails or answers badly */
export async function generateClipText(words: TextWord[], videoTitle: string | null, ask: AskText, language: ClipTextLanguage = 'roman', voice: BrandVoice | null = null): Promise<ClipText> {
  if (!words.length) throw new Error('This clip has no transcript yet')

  const prompt = `You write the text that goes with a short vertical video clip (Reels / Shorts) for an Indian creator. The speech can be Telugu, Hindi, Tamil or another Indian language, often mixed with English. Judge meaning in the original language.

${videoTitle ? `Full video title: ${videoTitle}\n` : ''}What is said in the clip:
${clipLines(words)}

Return ONLY JSON, keys in English:
{"hook": "...", "title": "...", "post_caption": "...", "hashtags": ["...", ...]}
${language === 'english' ? ENGLISH_RULE : language === 'native' ? NATIVE_RULE : ROMAN_RULE} This applies to the hook, title and post caption.
- hook: 3-7 words shown over the first 3 seconds that make someone stop scrolling. No clickbait lies: it must match what is actually said. No emoji, no hashtags.
- title: at most 60 characters, about this clip's moment (never just the video's title).
- post_caption: 1-2 sentences for the post.
- hashtags: exactly 5, no # sign needed, no spaces inside a tag, a mix of language-specific and English tags.${voiceRules(voice)}`

  const text = await ask(prompt)
  const match = text.match(/\{[\s\S]*\}/)
  if (!match) throw new Error('Gemini did not return JSON')
  const raw = JSON.parse(match[0]) as Record<string, unknown>
  const hook = clean(raw.hook, 80).split(' ').slice(0, 7).join(' ')
  const title = clean(raw.title, 60)
  const written = clean(raw.post_caption, 400)
  // The brand's closing line, added as it is (a prompt alone doesn't keep it word for word)
  const cta = voice?.cta.trim() ?? ''
  const post_caption = cta && !written.includes(cta) ? `${written}${written ? '\n\n' : ''}${cta}`.slice(0, 520) : written
  const hashtags = withBrandHashtags((Array.isArray(raw.hashtags) ? raw.hashtags : [])
    .map(h => clean(h, 40).replace(/^#+/, '').replace(/\s+/g, ''))
    .filter(Boolean)
    .slice(0, 5), voice)
  if (!hook || !title) throw new Error('Gemini left out the hook or title')
  // A title that is only the video's name says nothing about the clip: use the hook instead
  const sameAsVideo = !!videoTitle && title.toLowerCase().replace(/\W/g, '') === videoTitle.toLowerCase().replace(/\W/g, '')
  if (sameAsVideo) return { hook, title: hook.slice(0, 60), post_caption, hashtags }
  return { hook, title, post_caption, hashtags }
}
