"""Tests for the word-timing transfer (pure Python, no models):  python3 -m unittest transcriber/test_align.py"""
import unittest

from transcriber.align import romanize, transfer_timings


def W(text: str, start: float, end: float) -> dict:
    return {"word": text, "start": start, "end": end}


class Romanize(unittest.TestCase):
    def test_telugu_and_english_end_up_in_the_same_letters(self):
        self.assertEqual(romanize("నమస్తే"), "namaste")
        self.assertEqual(romanize("Hello,"), "hello")


class TransferTimings(unittest.TestCase):
    def test_same_words_take_the_timed_words_times(self):
        timed = [W("మనిషికి", 0.2, 0.8), W("విలువ", 0.9, 1.3), W("కాదు", 1.4, 1.8)]
        out = transfer_timings(["మనిషికి", "విలువ", "కాదు"], timed, 0.0, 2.0)
        self.assertEqual([round(w["start"], 2) for w in out], [0.2, 0.9, 1.4])
        self.assertEqual([round(w["end"], 2) for w in out], [0.8, 1.3, 1.8])

    def test_different_spellings_still_match(self):
        # the text model writes టైమ్, the timing model టైం: the same word
        timed = [W("సెకండ్", 1.0, 1.5), W("టైం", 1.6, 2.0)]
        out = transfer_timings(["సెకండ్", "టైమ్"], timed, 0.5, 2.5)
        self.assertAlmostEqual(out[1]["start"], 1.6, places=1)
        self.assertAlmostEqual(out[1]["end"], 2.0, places=1)

    def test_a_word_the_timing_model_missed_sits_between_its_neighbours(self):
        timed = [W("ఒకటి", 0.0, 0.5), W("మూడు", 1.5, 2.0)]
        out = transfer_timings(["ఒకటి", "రెండు", "మూడు"], timed, 0.0, 2.0)
        self.assertGreaterEqual(out[1]["start"], out[0]["end"] - 1e-6)
        self.assertLessEqual(out[1]["end"], out[2]["start"] + 1e-6)

    def test_without_timed_words_the_piece_is_shared_by_length(self):
        out = transfer_timings(["aa", "bbbb"], [], 10.0, 13.0)
        self.assertAlmostEqual(out[0]["start"], 10.0)
        self.assertAlmostEqual(out[-1]["end"], 13.0)
        self.assertLess(out[0]["end"] - out[0]["start"], out[1]["end"] - out[1]["start"])

    def test_times_always_move_forward_and_stay_in_the_piece(self):
        timed = [W("b", 1.8, 1.9), W("a", 0.1, 0.2)]   # out of order on purpose
        out = transfer_timings(["a", "b", "c"], timed, 0.0, 2.0)
        for w in out:
            self.assertGreaterEqual(w["start"], 0.0)
            self.assertLessEqual(w["end"], 2.0)
            self.assertGreaterEqual(w["end"], w["start"])
        self.assertEqual(out, sorted(out, key=lambda w: w["start"]))
        self.assertEqual([w["word"] for w in out], ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
