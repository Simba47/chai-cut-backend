// One-off: pictures for the dashboard (jobs/poster.ts) for videos that were ready before pictures were made.
// New videos get theirs when they are ready, and a video whose editing copy is made later gets one then; this
// covers the rest. Each is read over a signed link (ffmpeg fetches only the part it needs).
//
//   npx tsx src/scripts/backfill-posters.ts            dry run: only lists the videos without a picture
//   npx tsx src/scripts/backfill-posters.ts --apply    makes their pictures
import 'dotenv/config'
import db from '../db.js'
import { makePoster } from '../jobs/poster.js'

const apply = process.argv.includes('--apply')

// (the column is added when the worker starts: index.ts; here too with --apply — a dry run changes nothing)
if (apply) await db`ALTER TABLE videos ADD COLUMN IF NOT EXISTS poster_path TEXT`
const videos = await db<{ id: string; title: string | null }[]>`
  SELECT id, title FROM videos
  WHERE status = 'ready' AND storage_path IS NOT NULL AND role <> 'asset' AND (to_jsonb(videos)->>'poster_path') IS NULL
  ORDER BY created_at DESC`
console.log(`${videos.length} video(s) without a picture${apply ? '' : ' (dry run: add --apply to make them)'}`)
let made = 0
for (const [i, v] of videos.entries()) {
  if (!apply) { console.log(`  ${v.id}  ${v.title ?? ''}`); continue }
  const ok = await makePoster(v.id)
  if (ok) made++
  console.log(`  ${i + 1}/${videos.length} ${ok ? 'made' : 'skipped'}  ${v.title ?? v.id}`)
}
if (apply) console.log(`Done: ${made} of ${videos.length} pictures made`)
await db.end()
