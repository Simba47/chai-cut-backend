"""Tests for the word timings (pure Python + numpy, no models):  python3 -m unittest transcriber/test_align.py"""
import unittest

import numpy as np

from transcriber.align import forced_align, letters, word_times

# A tiny letter vocabulary in the timing model's style, with a blank
VOCAB = {"<pad>": 0, "మ": 1, "న": 2, "వ": 3, "ి": 4, "ల": 5, "ు": 6, "|": 7}
BLANK = 0


def frames(path: list[int], n_vocab: int = len(VOCAB)) -> np.ndarray:
    """Log-probabilities that clearly favour one frame-by-frame path"""
    lp = np.full((len(path), n_vocab), -10.0, dtype=np.float32)
    for t, k in enumerate(path):
        lp[t, k] = -0.01
    return lp


class Letters(unittest.TestCase):
    def test_a_word_is_its_letters(self):
        self.assertEqual(letters("విలువ", VOCAB), [3, 4, 5, 6, 3])

    def test_letters_the_model_lacks_are_left_out(self):
        self.assertEqual(letters("మxన", VOCAB), [1, 2])

    def test_a_word_with_no_known_letters_becomes_a_word_gap(self):
        self.assertEqual(letters("42", VOCAB), [7])


class ForcedAlign(unittest.TestCase):
    def test_each_token_gets_the_frames_where_the_model_heard_it(self):
        B = BLANK
        lp = frames([B, 1, 1, B, 2, B, B, 3, 4, 4, 5, B])
        self.assertEqual(forced_align(lp, [1, 2, 3, 4, 5], B), [(1, 2), (4, 4), (7, 7), (8, 9), (10, 10)])

    def test_the_same_token_twice_needs_a_blank_between(self):
        B = BLANK
        lp = frames([2, 2, B, 2, B])
        self.assertEqual(forced_align(lp, [2, 2], B), [(0, 1), (3, 3)])

    def test_a_long_piece_with_many_tokens_aligns(self):
        B = BLANK
        targets = [1, 2] * 100                       # 200 tokens: 401 places in the alignment
        lp = frames([k for t in targets for k in (t, B)])
        spans = forced_align(lp, targets, B)
        self.assertEqual(spans[0], (0, 0))
        self.assertEqual(spans[-1], (398, 398))

    def test_too_few_frames_gives_none(self):
        self.assertIsNone(forced_align(frames([1, 2]), [1, 2, 3], BLANK))


class WordTimes(unittest.TestCase):
    def test_words_start_where_their_first_letter_is_heard(self):
        B = BLANK
        lp = frames([B, B, 1, 2, B, B, B, 3, 4, 5, 6, 3, B, B])   # మన at frame 2, విలువ at frame 7
        out = word_times(["మన", "విలువ"], lp, VOCAB, B, frame_s=0.02, offset=10.0, piece_end=11.0)
        self.assertEqual([w["word"] for w in out], ["మన", "విలువ"])
        self.assertAlmostEqual(out[0]["start"], 10.04)
        self.assertAlmostEqual(out[1]["start"], 10.14)
        self.assertAlmostEqual(out[1]["end"], 10.24)
        self.assertLessEqual(out[0]["end"], out[1]["start"])

    def test_without_an_alignment_the_piece_is_shared_by_length(self):
        out = word_times(["మ", "విలువ"], frames([1]), VOCAB, BLANK, frame_s=0.02, offset=10.0, piece_end=13.0)
        self.assertAlmostEqual(out[0]["start"], 10.0)
        self.assertAlmostEqual(out[-1]["end"], 13.0)
        self.assertLess(out[0]["end"] - out[0]["start"], out[1]["end"] - out[1]["start"])

    def test_without_frames_the_piece_is_shared_by_length(self):
        out = word_times(["మ", "విలువ"], None, {}, -1, frame_s=0.02, offset=2.0, piece_end=3.0)
        self.assertAlmostEqual(out[0]["start"], 2.0)
        self.assertAlmostEqual(out[-1]["end"], 3.0)

    def test_every_word_has_some_length_and_stays_in_the_piece(self):
        B = BLANK
        lp = frames([1, 3, 4, 5, 6, 3, B])
        out = word_times(["మ", "విలువ"], lp, VOCAB, B, frame_s=0.02, offset=0.0, piece_end=0.13)
        for w in out:
            self.assertGreater(w["end"], w["start"])
            self.assertGreaterEqual(w["start"], 0.0)
            self.assertLessEqual(w["end"], 0.13)


if __name__ == "__main__":
    unittest.main()
