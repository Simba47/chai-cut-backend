/**
 * Words to post a clip with: a hook shown over its first 3 seconds, a title, a post caption and
 * hashtags, written by Gemini from what is said in the clip.
 *
 * Copied as-is between the repos (chai-cut-backend/src/lib/clipText.ts, chai-cut-frontend
 * src/lib/clipText.ts): the worker writes them for "Make my clips", the app's Regenerate button
 * rewrites them. Keep the two identical.
 */
import { GoogleGenerativeAI } from '@google/generative-ai'

// The same rule as clipFinder.ts ROMAN_RULE: text in Roman letters (Tenglish, Hinglish…)
const ROMAN_RULE = `Write the text in Roman (English) letters, in the speaker's own language the way people type it in chats: Telugu as Tenglish (e.g. "Ee cinema ki audience pichi ekkaru!"), Hindi as Hinglish (e.g. "Yeh scene dekh ke sab hil gaye!"), Tamil as Tanglish, and so on; keep English words as they are, and if the speaker talks in English write English. Never use Telugu, Devanagari, Tamil or any other non-Latin script.`

export interface ClipText { hook: string; title: string; post_caption: string; hashtags: string[] }
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

/** Hook, title, post caption and hashtags for a clip; throws if Gemini fails or answers badly */
export async function generateClipText(words: TextWord[], videoTitle: string | null, apiKey: string): Promise<ClipText> {
  if (!words.length) throw new Error('This clip has no transcript yet')

  const prompt = `You write the text that goes with a short vertical video clip (Reels / Shorts) for an Indian creator. The speech can be Telugu, Hindi, Tamil or another Indian language, often mixed with English. Judge meaning in the original language.

${videoTitle ? `Full video title: ${videoTitle}\n` : ''}What is said in the clip:
${clipLines(words)}

Return ONLY JSON, keys in English:
{"hook": "...", "title": "...", "post_caption": "...", "hashtags": ["...", ...]}
${ROMAN_RULE} This applies to the hook, title and post caption.
- hook: 3-7 words shown over the first 3 seconds that make someone stop scrolling. No clickbait lies: it must match what is actually said. No emoji, no hashtags.
- title: at most 60 characters, about this clip's moment (never just the video's title).
- post_caption: 1-2 sentences for the post.
- hashtags: exactly 5, no # sign needed, no spaces inside a tag, a mix of language-specific and English tags.`

  const model = new GoogleGenerativeAI(apiKey).getGenerativeModel({ model: 'gemini-2.5-flash' })
  const text = (await model.generateContent(prompt)).response.text()
  const match = text.match(/\{[\s\S]*\}/)
  if (!match) throw new Error('Gemini did not return JSON')
  const raw = JSON.parse(match[0]) as Record<string, unknown>
  const hook = clean(raw.hook, 80).split(' ').slice(0, 7).join(' ')
  const title = clean(raw.title, 60)
  const post_caption = clean(raw.post_caption, 400)
  const hashtags = (Array.isArray(raw.hashtags) ? raw.hashtags : [])
    .map(h => clean(h, 40).replace(/^#+/, '').replace(/\s+/g, ''))
    .filter(Boolean)
    .slice(0, 5)
  if (!hook || !title) throw new Error('Gemini left out the hook or title')
  // A title that is only the video's name says nothing about the clip: use the hook instead
  const sameAsVideo = !!videoTitle && title.toLowerCase().replace(/\W/g, '') === videoTitle.toLowerCase().replace(/\W/g, '')
  if (sameAsVideo) return { hook, title: hook.slice(0, 60), post_caption, hashtags }
  return { hook, title, post_caption, hashtags }
}
