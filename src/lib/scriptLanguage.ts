// Which Indian language a transcript is in, from the script most of its letters are written in.
// (The first script found used to decide: a Kannada sample with a few words Gemini wrote in Tamil
// was read as Tamil, and the whole video's English words were then spelled in Tamil letters.)
const SCRIPTS: Array<[string, RegExp]> = [
  ['te-IN', /[ఀ-౿]/g], ['hi-IN', /[ऀ-ॿ]/g], ['ta-IN', /[஀-௿]/g], ['kn-IN', /[ಀ-೿]/g],
  ['ml-IN', /[ഀ-ൿ]/g], ['bn-IN', /[ঀ-৿]/g], ['gu-IN', /[઀-૿]/g], ['pa-IN', /[਀-੿]/g],
]

/** The language whose script has the most letters in `text`; undefined when there are none (English) */
export function detectSarvamLang(text: string): string | undefined {
  let best: string | undefined
  let most = 0
  for (const [lang, letters] of SCRIPTS) {
    const n = text.match(letters)?.length ?? 0
    if (n > most) { best = lang; most = n }
  }
  return best
}
