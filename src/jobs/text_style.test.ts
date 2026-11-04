// A text's entrance animation in the export (python/text_style.py) runs on the preview's own numbers:
// text_anim_preview.json holds the editor's (VideoPreview drawStyledText) for all 17 entrances at 11
// moments. Change one side, change the other and make that file again from the preview.
// Run: npx tsx --test src/jobs/text_style.test.ts
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const PY = fileURLToPath(new URL('../python/', import.meta.url))

test('every entrance animation matches the preview, moment by moment', () => {
  const script = `
import json, sys
sys.path.insert(0, ${JSON.stringify(PY)})
import text_style
rows = json.load(open(${JSON.stringify(PY + 'text_anim_preview.json')}))
bad = []
for r in rows:
    p = text_style._anim({'animation': r['a'], 'opacity': 0.8, 'text': 'Hello world'}, r['dt'], 1080, 1920, 1.0)
    for k in ['alpha', 'scale', 'dx', 'dy', 'spin', 'blur', 'wipe', 'glitch', 'chars']:
        a, b = r['r'][k], p[k]
        if not (a is None and b is None) and abs((a or 0) - (b or 0)) > 1e-6:
            bad.append(f"{r['a']} at {r['dt']} ms: {k} {a} vs {b}")
print(json.dumps({'n': len(rows), 'anims': len({r['a'] for r in rows}), 'bad': bad[:5]}))
`
  const out = JSON.parse(execFileSync(process.platform === 'win32' ? 'python' : 'python3', ['-c', script], { encoding: 'utf8' }))
  assert.equal(out.anims, 17)
  assert.deepEqual(out.bad, [])
})

test('a plain text stays on the plain drawtext path; any look makes it styled', () => {
  const script = `
import json, sys
sys.path.insert(0, ${JSON.stringify(PY)})
import text_style
plain = {'x': 0.1, 'y': 0.2, 'text': 'Hi', 'font': 'Roboto', 'color': '#ffffff'}
print(json.dumps([text_style.is_styled(plain), text_style.is_styled({**plain, 'shadow': 'soft', 'animation': 'none', 'opacity': 1}),
  text_style.is_styled({**plain, 'bg_color': '#ffffff'}), text_style.is_styled({**plain, 'animation': 'pop'}),
  text_style.is_styled({**plain, 'x': None, 'bg_color': '#ffffff'})]))
`
  const out = JSON.parse(execFileSync(process.platform === 'win32' ? 'python' : 'python3', ['-c', script], { encoding: 'utf8' }))
  assert.deepEqual(out, [false, false, true, true, false])
})
