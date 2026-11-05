// Every transition the export draws is one the database takes (and nothing it can't draw).
// Run: npx tsx --test src/lib/transitionKinds.test.ts
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { TRANSITION_KINDS } from './transitionKinds.js'

test('the database takes exactly the kinds render.py draws, and cut', () => {
  const py = readFileSync(fileURLToPath(new URL('../python/render.py', import.meta.url)), 'utf8')
  const table = py.slice(py.indexOf('_XFADE = {'), py.indexOf('}', py.indexOf('_XFADE = {')))
  const drawn = [...table.matchAll(/"([a-z]+)":/g)].map(m => m[1])
  assert.ok(drawn.length >= 20, `read ${drawn.length} kinds from render.py`)
  assert.deepEqual([...TRANSITION_KINDS].filter(k => k !== 'cut').sort(), [...new Set(drawn)].sort())
})
