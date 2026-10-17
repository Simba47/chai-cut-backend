import { test } from 'node:test'
import assert from 'node:assert/strict'
import { generateClipText } from './clipText.js'

const words = [{ word: 'This', start_ms: 0, end_ms: 300 }, { word: 'editor', start_ms: 300, end_ms: 700 }, { word: 'is', start_ms: 700, end_ms: 900 }, { word: 'fast', start_ms: 900, end_ms: 1300 }]
const answer = JSON.stringify({ hook: 'Editing just got fast', title: 'The fastest editor', post_caption: 'Clips in minutes.', hashtags: ['reels', 'Shortcut', 'editing', 'creator', 'india'] })

test('a brand voice goes into the prompt; its closing line and hashtags are added to the text', async () => {
  let prompt = ''
  const text = await generateClipText(words, 'Demo', async p => { prompt = p; return answer }, 'english',
    { tone: 'playful and confident', avoid: 'cheap, hack', hashtags: ['#Shortcut', 'ShortcutApp'], cta: 'Try Shortcut free — link in bio' })
  assert.match(prompt, /Brand voice: write it playful and confident\./)
  assert.match(prompt, /Never use these words or phrases: cheap, hack\./)
  assert.match(prompt, /closing line added after it/)
  assert.equal(text.post_caption, 'Clips in minutes.\n\nTry Shortcut free — link in bio')
  assert.deepEqual(text.hashtags, ['Shortcut', 'ShortcutApp', 'reels', 'editing', 'creator', 'india'], 'brand tags first, no repeats')
})

test('without a brand voice the prompt and text are as before', async () => {
  let prompt = ''
  const text = await generateClipText(words, 'Demo', async p => { prompt = p; return answer }, 'english')
  assert.doesNotMatch(prompt, /Brand voice|Never use these words|closing line/)
  assert.equal(text.post_caption, 'Clips in minutes.')
  assert.equal(text.hashtags.length, 5)
})
