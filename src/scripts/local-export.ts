/**
 * Export a clip on this PC with the code in this folder — to test changes before they're deployed.
 *
 *   npx tsx --env-file=.env src/scripts/local-export.ts <clip id> [480p|720p|1080p] [output folder]
 *
 * Safe against the live database: it only reads (read-only connection; the export's writes are
 * skipped), and the video is saved in the output folder (default ./local-exports), not uploaded.
 * Needs FFmpeg (and Python with Pillow) on this PC.
 */
const [clipId, quality = '720p', outDir = 'local-exports'] = process.argv.slice(2)
if (!clipId) {
  console.error('Usage: npx tsx --env-file=.env src/scripts/local-export.ts <clip id> [480p|720p|1080p] [output folder]')
  process.exit(1)
}
// Before anything imports the database: local mode (read-only, files kept here)
process.env.LOCAL_EXPORT_DIR = (await import('node:path')).resolve(outDir)

const { default: db } = await import('../db.js')
const { handleRenderJob } = await import('../jobs/render.js')
const [row] = await db`SELECT v.storage_path FROM clips c JOIN videos v ON v.id = c.video_id WHERE c.id = ${clipId}`
if (!row?.storage_path) { console.error('Clip not found (or its video has no file)'); process.exit(1) }
const [{ default_transaction_read_only: ro }] = await db`SHOW default_transaction_read_only`
console.log(`[local export] clip ${clipId}, ${quality}; database read-only: ${ro}`)
if (ro !== 'on') { console.error('Refusing to run: the database connection is not read-only'); process.exit(1) }
const t0 = Date.now()
await handleRenderJob({ id: 'local', type: 'render', status: 'processing', payload: { clip_id: clipId, video_storage_path: row.storage_path, quality } } as never)
console.log(`[local export] done in ${((Date.now() - t0) / 1000).toFixed(0)} s → ${process.env.LOCAL_EXPORT_DIR}`)
await db.end()
