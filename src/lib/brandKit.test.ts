import { test } from 'node:test'
import assert from 'node:assert/strict'

// brandKit.ts reads the database only in loadClipBrand; the functions tested here don't
process.env.DATABASE_URL ??= 'postgres://test:test@127.0.0.1:1/test'
const { clipBrand, logoSpan, brandCaptionRow } = await import('./brandKit.js')

const U = '11111111-1111-1111-1111-111111111111'
const kit = (over: Record<string, unknown> = {}) => ({
  colors: { main: '#1c1c22', accent: '#c8ff00', extra: [] },
  logo: { path: `brand/${U}/logo.png`, ratio: 0.5, corner: 'tr', x: null, y: null, size: 0.2, opacity: 0.8, show: 'all', seconds: 3 },
  captions: { on: true, template: 'beast', animation: 'hormozi', font: 'Montserrat', size: 60, color: '#FFFFFF', highlight_color: '#FF0000', uppercase: true, words_per_line: 3 },
  title: { font: 'Poppins', color: '#c8ff00', size: 70 },
  ...over,
})

test('the logo sits in its corner, a margin in, keeping its shape (as the editor puts it)', () => {
  const b = clipBrand(kit(), U)
  // 20% wide, half as tall as wide on a 1080 × 1920 frame; top right, 4% of the width in
  assert.deepEqual(b.logo && { x: b.logo.x, y: b.logo.y, w: b.logo.w, h: b.logo.h }, { x: 0.76, y: 0.08, w: 0.2, h: 0.0563 })
  // (kept clear of the apps' header unless switched off — the frontend's logoBox, Phase 4)
  assert.equal(clipBrand(kit({ logo: { ...kit().logo, safe: false } }), U).logo?.y, 0.0225)
  assert.equal(b.logo?.opacity, 0.8)
})

test('a logo placed by hand goes there, kept inside the frame', () => {
  const b = clipBrand(kit({ logo: { ...kit().logo, x: 0.95, y: 0.5 } }), U)
  assert.equal(b.logo?.x, 0.8)   // 1 − its width
  assert.equal(b.logo?.y, 0.5)
})

test('someone else’s logo file is never used', () => {
  assert.equal(clipBrand(kit({ logo: { ...kit().logo, path: 'brand/other-user/logo.png' } }), U).logo, null)
  assert.equal(clipBrand(kit({ logo: { ...kit().logo, path: `brand/${U}/../x.png` } }), U).logo, null)
})

test('when the logo shows: the whole clip, its first or its last seconds', () => {
  const logo = clipBrand(kit(), U).logo!
  assert.deepEqual(logoSpan(logo, 30000), [0, 30000])
  assert.deepEqual(logoSpan({ ...logo, show: 'start', seconds: 5 }, 30000), [0, 5000])
  assert.deepEqual(logoSpan({ ...logo, show: 'end', seconds: 5 }, 30000), [25000, 30000])
  assert.deepEqual(logoSpan({ ...logo, show: 'end', seconds: 15 }, 8000), [0, 8000], 'a clip shorter than that: the whole clip')
})

test('brand captions: off gives none (the job’s own style stays); on gives the kit’s style', () => {
  assert.equal(clipBrand(kit({ captions: { ...kit().captions, on: false } }), U).captions, null)
  const c = clipBrand(kit(), U).captions!
  assert.deepEqual(c, { template: 'beast', animation: 'hormozi', font: 'Montserrat', size: 60, color: '#ffffff', highlight_color: '#ff0000', uppercase: true, words_per_line: 3 })
})

test('the kit’s caption font only for English letters: Telugu and Hindi keep their own font', () => {
  const c = clipBrand(kit(), U).captions!
  assert.equal(brandCaptionRow(c, { font: 'roboto', language: null }).font, 'Montserrat', 'English')
  assert.equal(brandCaptionRow(c, { font: 'roboto', language: 'roman' }).font, 'Montserrat', 'Tenglish (English letters)')
  assert.equal(brandCaptionRow(c, { font: 'noto-sans-telugu', language: null }).font, 'noto-sans-telugu', 'Telugu letters')
  assert.equal(brandCaptionRow(c, { font: 'noto-sans-devanagari', language: null }).font, 'noto-sans-devanagari', 'Hindi letters')
  const row = brandCaptionRow(c, { font: 'roboto', language: null })
  assert.equal(row.template, 'beast'); assert.equal(row.position, 'bottom-center'); assert.equal(row.uppercase, true)
})

test('a kit saved before these settings existed still works (defaults fill the rest)', () => {
  const b = clipBrand({ logo: { path: `brand/${U}/l.png`, corner: 'bl' }, title: { color: '#123456' } }, U)
  assert.equal(b.captions, null)
  assert.equal(b.title.color, '#123456')
  assert.equal(b.logo?.show, 'all')
  assert.ok(b.logo && b.logo.x < 0.1 && b.logo.y > 0.7 && b.logo.y + b.logo.h <= 0.87 + 1e-3, 'bottom left, above the apps caption')
})

test('the logo moves to the clearest corner, the kit’s own first (the frontend’s pickLogoCorner)', async () => {
  const { placedLogo, logoAvoid } = await import('./brandKit.js')
  const logo = clipBrand(kit(), U).logo!
  const box = (l: typeof logo) => ({ x: l.x, y: l.y })
  assert.deepEqual(box(placedLogo(logo, [])), { x: 0.76, y: 0.08 }, 'nothing to keep off: its own corner')
  // a face right under it, top right: the next clear corner is top left
  const moved = placedLogo(logo, [{ x: 0.7, y: 0.05, w: 0.3, h: 0.2 }])
  assert.deepEqual(box(moved), { x: 0.04, y: 0.08 })
  assert.equal(moved.w, 0.2); assert.equal(moved.h, 0.0563)
  // switched off in the kit, or placed by hand: it stays
  const off = clipBrand(kit({ logo: { ...kit().logo, smart: false } }), U).logo!
  assert.deepEqual(box(placedLogo(off, [{ x: 0.7, y: 0.05, w: 0.3, h: 0.2 }])), { x: 0.76, y: 0.08 })
  const hand = clipBrand(kit({ logo: { ...kit().logo, x: 0.7, y: 0.1 } }), U).logo!
  assert.deepEqual(box(placedLogo(hand, [{ x: 0.7, y: 0.05, w: 0.3, h: 0.2 }])), { x: 0.7, y: 0.1 })
  // the captions' band: a logo at the bottom goes to the top
  const low = clipBrand(kit({ logo: { ...kit().logo, corner: 'bl' } }), U).logo!
  assert.ok(placedLogo(low, logoAvoid([], [], true)).y < 0.5)

  // faces seen through a full-frame part's view, in the frame's own shares; a split's aren't counted
  const seg = { start_ms: 0, end_ms: 2000, layout: 'vertical', slotKfs: [[{ t_ms: 0, x: 0.25, y: 0, w: 0.5, h: 1 }]] }
  const frames = [{ t_ms: 0, faces: [{ x: 0.5, y: 0.1, w: 0.2, h: 0.2 }] }, { t_ms: 1000, faces: [{ x: 0.5, y: 0.1, w: 0.2, h: 0.2 }] }]
  assert.deepEqual(logoAvoid([seg], frames, false).map(b => [b.x, b.y, b.w, b.h]), [[0.5, 0.1, 0.4, 0.2], [0.5, 0.1, 0.4, 0.2]])
  assert.deepEqual(logoAvoid([{ ...seg, layout: 'split' }], frames, false), [])
  assert.equal(logoAvoid([seg], frames, true).length, 3, 'and the captions')
})

test("a kit's progress bar and voice aren't used", () => {
  const b = clipBrand({ progress: { on: true, color: '#FF3366' }, progress_bar: { on: true }, voice: { tone: 'fun' } }, U) as Record<string, unknown>
  assert.equal(b.progress, undefined); assert.equal(b.progress_bar, undefined); assert.equal(b.voice, undefined)
})

test("the kit's end card is saved as not put on yet (the frontend's engine adds it)", async () => {
  const { appliedHere } = await import('./brandKit.js')
  assert.deepEqual(appliedHere({ colors: {} }), { colors: {} })
  assert.deepEqual(appliedHere({ ends: { card: true, seconds: 4, intro: { id: 'x', ms: 3000 } } }), { ends: { card: false, seconds: 4, intro: null, outro: null } })
})
