/**
 * Word-for-word conversions with the model (Telugu words → Tenglish, English words → Telugu
 * script, …): many words in, exactly one answer per word out, same order.
 *
 * No imports: the model call is passed in, so this is tested without the network.
 */

/** Words per request for the script conversions, and the smallest piece a miscounted batch is split into */
const CONVERT_BATCH = 80
const CONVERT_MIN_SPLIT = 10
/** Batches of one transcript converted at the same time (lib/gemini also caps each process's AI requests) */
const CONVERT_PARALLEL = 3

/**
 * Converts words one for one with the model: batches of 80, three at a time. Returns one raw
 * answer per input word, same order (undefined where a word wasn't converted).
 * The model is told to answer word for word but now and then merges or splits one (80 in, 81
 * out — seen in production). Such a batch is asked once more, then as two halves (smaller batches
 * come back right), down to 10 words: a stray word costs at most 10 words their conversion, not 80.
 * A request that fails outright (after lib/gemini's own retries) leaves its words undefined.
 */
export async function convertWordForWord(words: string[], ask: (batch: string[]) => Promise<string>, label: string): Promise<unknown[]> {
  const out: unknown[] = new Array(words.length).fill(undefined)
  let lost = 0
  const convert = async (from: number, to: number): Promise<void> => {
    const batch = words.slice(from, to)
    let failure = ''
    for (let attempt = 1; attempt <= 2; attempt++) {
      let answer: string
      try {
        answer = await ask(batch)
      } catch (err) {
        failure = err instanceof Error ? err.message : String(err)
        continue
      }
      let arr: unknown
      try { arr = JSON.parse(answer || '[]') } catch { arr = null }
      if (Array.isArray(arr) && arr.length === batch.length) {
        arr.forEach((v, j) => { out[from + j] = v })
        return
      }
      failure = `expected ${batch.length} words, got ${Array.isArray(arr) ? arr.length : 'no list'}`
    }
    // The wrong number of words twice: smaller batches. (A request that failed outright isn't split.)
    if (failure.startsWith('expected') && batch.length > CONVERT_MIN_SPLIT) {
      const mid = from + Math.ceil(batch.length / 2)
      await Promise.all([convert(from, mid), convert(mid, to)])
      return
    }
    lost += batch.length
    console.warn(`${label} words ${from}-${to} not converted: ${failure.slice(0, 200)}`)
  }
  let next = 0
  await Promise.all(Array.from({ length: CONVERT_PARALLEL }, async () => {
    while (next < words.length) {
      const from = next
      next += CONVERT_BATCH
      await convert(from, Math.min(words.length, from + CONVERT_BATCH))
    }
  }))
  if (lost) console.warn(`${label} ${lost} of ${words.length} word(s) left to the fallback`)
  return out
}
