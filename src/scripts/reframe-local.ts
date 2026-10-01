/**
 * Try "Make my clips" framing on a local video, without the database or the job queue: runs the
 * same face / reaction / cut analysis and layout rules as the worker (jobs/ai_edit.ts), prints the
 * layout it chose over time, and renders the 9:16 result with render.py (the exporter), each part
 * labelled with its layout.
 *
 *   npx tsx src/scripts/reframe-local.ts --video <file or URL> [--start 60] [--end 120] [--out out.mp4]
 *                                        [--words-from <video id>]
 *
 * --words-from reads that video's transcript (read-only) for real sentence endings and speakers.
 * Without it, someone is assumed to be talking the whole time (one speaker label).
 */
import { execFile, spawn } from 'node:child_process'
import { promisify } from 'node:util'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

// ai_edit.ts pulls in the database and storage clients; they are never used here, but they
// refuse to load without settings, so give them harmless placeholders if none are set
process.env.DATABASE_URL ||= 'postgres://unused@localhost/unused'
process.env.R2_ENDPOINT ||= 'http://localhost'
process.env.R2_ACCESS_KEY_ID ||= 'unused'
process.env.R2_SECRET_ACCESS_KEY ||= 'unused'
process.env.GEMINI_API_KEY ||= 'unused'
const { analyseClip, buildSegments, probeSize, trackFaces, planSpeakers, reactionStrength, personShots } = await import('../jobs/ai_edit.js')

const exec = promisify(execFile)
const arg = (name: string, fallback?: string) => {
  const i = process.argv.indexOf(`--${name}`)
  return i >= 0 ? process.argv[i + 1] : fallback
}
const video = arg('video')
if (!video) { console.error('Usage: npx tsx src/scripts/reframe-local.ts --video <file or URL> [--start s] [--end s] [--out file.mp4]'); process.exit(1) }
const startS = Number(arg('start', '0'))
const endArg = arg('end')
const out = resolve(arg('out', 'reframe-test.mp4')!)

const tmp = await mkdtemp(join(tmpdir(), 'reframe-'))
try {
  // The part to test, as its own file (keeps analysis and render fast on long videos)
  const clip = join(tmp, 'clip.mp4')
  console.log(`• Cutting ${startS}s–${endArg ?? 'end'}s …`)
  await exec('ffmpeg', ['-v', 'error', '-ss', String(startS), '-i', video, ...(endArg ? ['-t', String(Number(endArg) - startS)] : []),
    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-c:a', 'aac', '-y', clip], { maxBuffer: 64 * 1024 * 1024 })
  const { stdout } = await exec('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'default=nw=1:nk=1', clip])
  const durMs = Math.round(parseFloat(stdout) * 1000)
  const size = await probeSize(clip)
  console.log(`• ${(durMs / 1000).toFixed(1)}s, ${size?.width}×${size?.height}`)

  // The real transcript (read-only), or one continuous speaker with a "word" every 300 ms
  let words: Array<{ start_ms: number; end_ms: number; speaker_id: string | null }>
  const wordsFrom = arg('words-from')
  if (wordsFrom) {
    const { config } = await import('dotenv')
    const env = config({ processEnv: {} }).parsed ?? {}
    const postgres = (await import('postgres')).default
    const sql = postgres(env.DATABASE_URL!, { ssl: 'require', max: 1, onnotice: () => {} })
    const from = startS * 1000, to = from + durMs
    const rows = await sql.begin('read only', tx => tx`
      SELECT tw.start_ms, tw.end_ms, tw.speaker_id FROM transcript_words tw
      WHERE tw.transcript_id = (SELECT t.id FROM transcripts t WHERE t.video_id = ${wordsFrom}
          AND EXISTS (SELECT 1 FROM transcript_words WHERE transcript_id = t.id) ORDER BY t.created_at DESC LIMIT 1)
        AND tw.start_ms >= ${from} AND tw.start_ms < ${to} ORDER BY tw.start_ms`)
    await sql.end()
    words = rows.map(w => ({ start_ms: w.start_ms - from, end_ms: w.end_ms - from, speaker_id: w.speaker_id }))
    console.log(`• Transcript: ${words.length} words`)
  } else {
    words = Array.from({ length: Math.floor(durMs / 300) }, (_, k) => ({ start_ms: k * 300, end_ms: k * 300 + 250, speaker_id: 'A' }))
  }

  console.log('• Finding faces, expressions and camera cuts (4 frames a second) …')
  const t0 = Date.now()
  const analysis = await analyseClip(tmp, clip, 0, durMs, size, words)
  const segments = buildSegments(analysis, durMs)
  const exFrames = analysis.info.frames.filter(f => f.faces.some(face => face.ex)).length
  console.log(`  ${analysis.info.frames.length} frames in ${((Date.now() - t0) / 1000).toFixed(0)}s · up to ${analysis.info.face_count} people · expressions read on ${exFrames} frames · ${analysis.cuts.length} camera cut(s)`)

  // The strongest expressions measured (to tune what counts as a "strong" reaction)
  {
    const frames = [...analysis.info.frames].sort((x, y) => x.frame_index - y.frame_index)
    const tracks = trackFaces(frames, analysis.cuts)
    const speakers = planSpeakers(frames, tracks, analysis.speech, durMs)
    const known = speakers.filter(s => typeof s === 'number').length
    const vals = new Map<number, Array<NonNullable<typeof frames[number]['faces'][number]['ex']>>>()
    frames.forEach((f, fi) => f.faces.forEach((face, k) => { if (face.ex) vals.set(tracks[fi][k], [...(vals.get(tracks[fi][k]) ?? []), face.ex]) }))
    const med = (v: number[]) => [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)] ?? 0
    const top: Array<{ t: number; id: number; s: number; ex: string }> = []
    frames.forEach((f, fi) => f.faces.forEach((face, k) => {
      const id = tracks[fi][k], v = vals.get(id)
      if (!face.ex || !v) return
      const base = { sm: med(v.map(e => e.sm)), jo: med(v.map(e => e.jo)), bu: med(v.map(e => e.bu)), bd: med(v.map(e => e.bd)), ew: med(v.map(e => e.ew)), fr: med(v.map(e => e.fr)) }
      const st = reactionStrength(face.ex, base)
      const e = face.ex
      top.push({ t: f.frame_index / 4, id, s: st, ex: `smile ${e.sm.toFixed(2)} jaw ${e.jo.toFixed(2)} browsUp ${e.bu.toFixed(2)} browsDown ${e.bd.toFixed(2)} eyesWide ${e.ew.toFixed(2)} frown ${e.fr.toFixed(2)}` })
    }))
    top.sort((a, b) => b.s - a.s)
    console.log(`  speaker known on ${known} of ${frames.length} frames`)
    console.log('  strongest expressions (1.00 = strongest; a reaction needs ≥ 0.50 on 2 frames in a second):')
    for (const x of top.slice(0, 8)) console.log(`    ${x.t.toFixed(2).padStart(7)}s person ${x.id}: ${x.s.toFixed(2)}  (${x.ex})`)
  }

  // One-person camera shots: who, talking or listening, reactions
  const shots = personShots(analysis, durMs)
  if (shots.length) {
    console.log('\nOne-person shots:')
    for (const sh of shots) console.log(`  ${(sh.start_ms / 1000).toFixed(1).padStart(6)}s – ${(sh.end_ms / 1000).toFixed(1).padStart(6)}s  person ${sh.person}  ${sh.talking ? 'talking  ' : 'listening'}  reaction ${sh.reaction.toFixed(2)}${sh.peak_ms !== null ? ` at ${(sh.peak_ms / 1000).toFixed(1)}s` : ''}  | lips ${sh.lipMove.toFixed(4)} · max smile ${sh.top.sm.toFixed(2)} jaw ${sh.top.jo.toFixed(2)} browsUp ${sh.top.bu.toFixed(2)} browsDown ${sh.top.bd.toFixed(2)} eyesWide ${sh.top.ew.toFixed(2)} frown ${sh.top.fr.toFixed(2)}`)
  }

  const label = (s: typeof segments[number]) =>
    s.layout === 'trio' && s.slotSources ? 'TRIO — borrowed reactions + speaker (middle)'
      : s.layout === 'trio' ? 'TRIO — reactors + speaker (middle)'
      : s.layout === 'split' && s.slotSources?.[0] === null ? 'SPLIT — reaction shot (top) + speaker (borrowed)'
      : s.layout === 'split' && s.slotSources ? 'SPLIT — borrowed reaction (top) + speaker'
      : s.layout === 'split' && s.slotKfs[0][0].h < 1 ? 'SPLIT — reactor (top) + speaker (bottom)'
      : s.layout === 'split' ? 'SPLIT — two people'
      : 'VERTICAL'
  console.log('\nLayout over time:')
  for (const s of segments) console.log(`  ${(s.start_ms / 1000).toFixed(1).padStart(6)}s – ${(s.end_ms / 1000).toFixed(1).padStart(6)}s  ${label(s)}`)

  // Render it the way an export does, with each part's layout written on it
  const spec = {
    start_ms: 0, end_ms: durMs, main_video_id: 'MAIN',
    segments: segments.map((s, i) => ({
      id: `seg${i}`, start_ms: s.start_ms, end_ms: s.end_ms, layout: s.layout, sort_order: i, video_offset_ms: null,
      crop_boxes: s.slotKfs.map((kfs, slot) => ({
        id: `seg${i}-box${slot}`, slot_index: slot,
        // Borrowed footage: the same video from another moment (the cut file starts at the clip start)
        ...(s.slotSources?.[slot] != null
          ? { source_video_id: 'MAIN', source_offset_ms: s.slotSources[slot], muted: true }
          : { source_video_id: null, source_offset_ms: s.start_ms, muted: false }),
        volume: 1,
        box_keyframes: kfs.map(k => ({ t_ms: k.t_ms, x: k.x, y: k.y, w: k.w, h: k.h })),
      })),
    })),
    text_overlays: segments.map(s => ({
      text: label(s), start_ms: s.start_ms, end_ms: s.end_ms, x: null, y: 0.03, font: 'montserrat-bold', size: 34, color: '#C8FF00',
    })),
    caption_styles: [], words: [],
  }
  const specPath = join(tmp, 'spec.json')
  await writeFile(specPath, JSON.stringify(spec))
  console.log(`\n• Rendering ${out} …`)
  const script = join(dirname(fileURLToPath(import.meta.url)), '../python/render.py')
  await new Promise<void>((ok, fail) => {
    const p = spawn('python3', [script, '--video', clip, '--spec', specPath, '--output', out, '--secondary-videos', JSON.stringify({ MAIN: clip })],
      { stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' } })
    let err = ''
    p.stderr.on('data', d => { err += d })
    p.stdout.on('data', d => { err += d })
    p.on('close', code => (code === 0 ? ok() : fail(new Error(`render.py exited ${code}: ${err.slice(-1500)}`))))
  })
  console.log(`✓ Done: ${out}`)
} finally {
  await rm(tmp, { recursive: true, force: true })
}
process.exit(0)
