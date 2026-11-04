import { test } from 'node:test'
import assert from 'node:assert/strict'
import { detectSarvamLang } from './scriptLanguage.js'

// 4 Nov test: a Kannada dub's sample came back as Tamil, so its English words were spelled in Tamil
const KANNADA = 'ತುಂಬ ಟೈಟಾಗಿ ಅಂಟಿಸ್ತು ನನಗೆ ಅದೆಲ್ಲ ಬೇಕಾಗಿಲ್ಲ ನೀ ಈ ಲೆಟರ್ ಓದಿದ್ಯಾ ಸರಿಯಾಗಾದರೂ ನೋಡಿದ್ಯಾ ಇಲ್ಲ'

test('one language: its script', () => {
  assert.equal(detectSarvamLang('ఈ సినిమా చాలా బాగుంది'), 'te-IN')
  assert.equal(detectSarvamLang('यह फिल्म बहुत अच्छी है'), 'hi-IN')
  assert.equal(detectSarvamLang(KANNADA), 'kn-IN')
  assert.equal(detectSarvamLang('ഇന്ന് വൈകിട്ട് ഞാൻ ഈ വിൽപത്രം വായിക്കും'), 'ml-IN')
})

test('a few words in another script do not decide the language', () => {
  assert.equal(detectSarvamLang(`${KANNADA} நாட் மை பிசினஸ்`), 'kn-IN')
  assert.equal(detectSarvamLang(`${KANNADA} नमस्ते`), 'kn-IN')
  assert.equal(detectSarvamLang('ఈ సినిమా చాలా బాగుంది మీరు చూడండి चलो'), 'te-IN')
})

test('English letters only, or nothing: no Indian language', () => {
  assert.equal(detectSarvamLang('not my business of course'), undefined)
  assert.equal(detectSarvamLang(''), undefined)
})
