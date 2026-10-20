import { test } from 'node:test'
import assert from 'node:assert/strict'

// brandKit.ts reads the database only in loadClipBrand; the functions tested here don't
process.env.DATABASE_URL ??= 'postgres://test:test@127.0.0.1:1/test'
const { clipBrand, cleanProgressBar, logoSpan, brandCaptionRow } = await import('./brandKit.js')

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
  assert.deepEqual(b.logo && { x: b.logo.x, y: b.logo.y, w: b.logo.w, h: b.logo.h }, { x: 0.76, y: 0.0225, w: 0.2, h: 0.0563 })
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
  assert.ok(b.logo && b.logo.x < 0.1 && b.logo.y > 0.8, 'bottom left')
})

test("a clip's progress bar made safe; a kit's progress bar and voice aren't used", () => {
  const b = clipBrand({ progress: { on: true, color: '#FF3366' }, voice: { tone: 'fun' } }, U) as Record<string, unknown>
  assert.equal(b.progress, undefined); assert.equal(b.voice, undefined)
  assert.deepEqual(cleanProgressBar({ color: 'red', height: 999, position: 'left' }), { color: '#ffffff', height: 40, position: 'bottom' })
  assert.equal(cleanProgressBar(null), null)
})
