// A download's file name, from its Content-Disposition header (Drive and Dropbox links).

/**
 * Drive sends the name as UTF-8 but fetch reads every header as Latin-1, so 😂 arrives as "ð..."
 * and "｜" as "ï½.". Back to the bytes and read properly; a name that isn't valid UTF-8 really
 * was Latin-1 ("café") and stays as it came.
 */
function fromHeaderBytes(text: string): string {
  if (/[^\u0000-\u00ff]/.test(text)) return text
  try { return new TextDecoder('utf-8', { fatal: true }).decode(Buffer.from(text, 'latin1')) } catch { return text }
}

export function filenameFromDisposition(header: string | null): string | null {
  if (!header) return null
  const star = header.match(/filename\*\s*=\s*(?:UTF-8'')?([^;]+)/i)?.[1]
  if (star) { try { return decodeURIComponent(star.trim().replace(/^"|"$/g, '')) } catch { /* fall through */ } }
  const plain = header.match(/filename\s*=\s*"?([^";]+)"?/i)?.[1]?.trim()
  return plain ? fromHeaderBytes(plain) : null
}

/** The name without its extension, at most 120 characters (counted whole, so an emoji is never cut in half) */
export function titleFromFilename(filename: string | null): string | undefined {
  return filename ? Array.from(filename.replace(/\.[^.]+$/, '').trim()).slice(0, 120).join('').trim() : undefined
}
