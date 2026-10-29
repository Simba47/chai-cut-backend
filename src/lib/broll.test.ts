import { test } from 'node:test'
import assert from 'node:assert/strict'

// broll.ts loads the Gemini helper, which reads its settings when it loads; nothing here calls it
process.env.GOOGLE_CLOUD_PROJECT ??= 'test'
const { brollWindow, withBroll, searchCandidates, BROLL_LAYOUT, CANDIDATES } = await import('./broll.js')
type Kf = { t_ms: number; x: number; y: number; w: number; h: number }
const v = (t_ms: number, x: number): Kf => ({ t_ms, x, y: 0, w: 0.316, h: 1 })

// A clip: 0–10 s one person (Vertical), 10–16 s two people (Split), 16–20 s two close together (one Vertical on both)
const parts = [
  { start_ms: 0, end_ms: 10000, layout: 'vertical' as const, people: 1, slotKfs: [[v(0, 0.3), v(4000, 0.35), v(8000, 0.4)]] },
  { start_ms: 10000, end_ms: 16000, layout: 'split' as const, people: 2, slotKfs: [[{ t_ms: 10000, x: 0, y: 0, w: 0.5, h: 0.9 }], [{ t_ms: 10000, x: 0.5, y: 0, w: 0.5, h: 0.9 }]] },
  { start_ms: 16000, end_ms: 20000, layout: 'vertical' as const, people: 2, slotKfs: [[v(16000, 0.34)]] },
]

test('B-roll goes over one person in a Vertical, never over a Split or a Vertical on two people', () => {
  assert.deepEqual(brollWindow(parts, [], { start_ms: 3000, end_ms: 6000 }), { start_ms: 3000, end_ms: 6000 })
  assert.equal(brollWindow(parts, [], { start_ms: 11000, end_ms: 13000 }), null)
  assert.equal(brollWindow(parts, [], { start_ms: 16500, end_ms: 18500 }), null)
  // (across two parts: the longer side when it is a one-person Vertical and long enough)
  assert.equal(brollWindow(parts, [], { start_ms: 9000, end_ms: 11000 }), null)
  assert.deepEqual(brollWindow(parts, [], { start_ms: 8000, end_ms: 11000 }), { start_ms: 8000, end_ms: 10000 })
})

test('a camera cut inside the moment: the longer side, when long enough', () => {
  assert.deepEqual(brollWindow(parts, [3500], { start_ms: 3000, end_ms: 6000 }), { start_ms: 3500, end_ms: 6000 })
  assert.equal(brollWindow(parts, [4000, 5000], { start_ms: 3000, end_ms: 6000 }), null)
})

test('the old framing (people not known) takes B-roll over its Vertical parts', () => {
  const old = [{ start_ms: 0, end_ms: 10000, layout: 'vertical' as const, slotKfs: [[v(0, 0.3)]] }]
  assert.deepEqual(brollWindow(old, [], { start_ms: 3000, end_ms: 5000 }), { start_ms: 3000, end_ms: 5000 })
})

test('a shot: B-roll on top, the speaker below, the part under it cut around it', () => {
  const view = { x: 0.3, y: 0.05, w: 0.35, h: 0.55 }
  const alts = [{ ref: 'pixabay:1', provider: 'pixabay' as const, title: 'rain', thumb: 't', preview: 'p', url: 'u', width: 1280, height: 720, duration: 9 }]
  const out = withBroll(parts, [{ start_ms: 3000, end_ms: 6000, videoId: 'stock-1', view, query: 'rain on a street', alts, at: 0 }])
  assert.deepEqual(out.map(p => [p.start_ms, p.end_ms, p.layout]), [[0, 3000, 'vertical'], [3000, 6000, BROLL_LAYOUT], [6000, 10000, 'vertical'], [10000, 16000, 'split'], [16000, 20000, 'vertical']])
  const shot = out[1]
  assert.deepEqual(shot.slotKfs, [[{ t_ms: 3000, ...view }]])
  const frame = shot.frame as { main_slots: number[]; row_order: number[]; items: Array<Record<string, unknown>> }
  assert.deepEqual(frame.main_slots, [0])
  assert.deepEqual(frame.row_order, [1, 0], 'the footage row (1) on top, the speaker (0) below')
  assert.equal(frame.items.length, 1)
  assert.deepEqual({ ...frame.items[0], id: undefined }, {
    id: undefined, lane: 1, kind: 'video', start_ms: 3000, end_ms: 6000, source_video_id: 'stock-1', source_offset_ms: 0, volume: 1, muted: true,
    broll: { query: 'rain on a street', alts, at: 0 },
  })
  // The speaker's views on either side keep their keyframes, cut at the shot
  assert.deepEqual(out[0].slotKfs[0].map(k => k.t_ms), [0])
  assert.deepEqual(out[2].slotKfs[0].map(k => [k.t_ms, k.x]), [[6000, 0.35], [8000, 0.4]])
})

test('a shot over a Split is not laid', () => {
  const out = withBroll(parts, [{ start_ms: 11000, end_ms: 13000, videoId: 's', view: { x: 0, y: 0, w: 0.3, h: 0.5 }, query: 'q', alts: [], at: 0 }])
  assert.deepEqual(out.map(p => p.layout), ['vertical', 'split', 'vertical'])
})

test('candidates: the searches in turn, each video once, AI-made and too-small files left out, vertical first', async () => {
  process.env.PIXABAY_API_KEY = 'test'
  delete process.env.PEXELS_API_KEY
  const hit = (id: number, w = 1920, h = 1080, extra: Record<string, unknown> = {}) => ({
    id, tags: `tag${id}`, duration: 10, ...extra,
    videos: { large: { url: `https://cdn.pixabay.com/${id}-l.mp4`, width: w, height: h, thumbnail: `https://cdn.pixabay.com/${id}-l.jpg` }, tiny: { url: `https://cdn.pixabay.com/${id}-t.mp4`, width: 640, height: 360, thumbnail: `https://cdn.pixabay.com/${id}-t.jpg` } },
  })
  const results: Record<string, unknown[]> = {
    'q one': [hit(1), hit(2, 1080, 1920), hit(3, undefined, undefined, { isAiGenerated: true })],
    'q two': [hit(2, 1080, 1920), hit(4), hit(5, 640, 360)],
  }
  const realFetch = globalThis.fetch
  globalThis.fetch = (async (url: string) => {
    const q = new URL(url).searchParams.get('q') ?? ''
    return new Response(JSON.stringify({ hits: results[q] ?? [] }), { status: 200 })
  }) as typeof fetch
  try {
    const got = await searchCandidates(['q one', 'q two'])
    // q one (vertical first): 2, 1; q two: 2 (already), 4; 3 is AI-made, 5 has no 720p file
    assert.deepEqual(got.map(c => c.ref), ['pixabay:2', 'pixabay:1', 'pixabay:4'])
    assert.ok(got.length <= CANDIDATES)
    assert.equal(got[0].thumb, 'https://cdn.pixabay.com/2-t.jpg')
    assert.equal(got[0].url, 'https://cdn.pixabay.com/2-l.mp4')
  } finally {
    globalThis.fetch = realFetch
  }
})
