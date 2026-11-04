"""python -m unittest transcriber/test_pipeline.py  (no models needed)"""
import unittest

import numpy as np

from transcriber.pipeline import pad_pieces, text_batches, video_language


class VideoLanguage(unittest.TestCase):
    def test_most_votes_wins(self):
        self.assertEqual(video_language(["ta", "ta", "kn"]), "ta")

    def test_urdu_is_read_as_hindi(self):
        # 4 Nov: a Hindi dub came back as "ur" and was written in Urdu letters; the app has no Urdu
        self.assertEqual(video_language(["ur", "ur", "hi"]), "hi")
        self.assertEqual(video_language(["ur", "hi", "ta"]), "hi")

    def test_no_votes_telugu(self):
        self.assertEqual(video_language([]), "te")


class TextBatches(unittest.TestCase):
    def test_every_piece_once_longest_first_and_no_batch_too_big(self):
        lengths = [3.0, 25.0, 0.5, 12.0, 25.0, 7.0, 1.0]
        groups = text_batches(lengths, 3)
        self.assertEqual(sorted(i for g in groups for i in g), list(range(len(lengths))))
        self.assertTrue(all(len(g) <= 3 for g in groups))
        flat = [lengths[i] for g in groups for i in g]
        self.assertEqual(flat, sorted(lengths, reverse=True))

    def test_no_pieces_no_batches(self):
        self.assertEqual(text_batches([], 4), [])


class PadPieces(unittest.TestCase):
    def test_short_piece_is_centred_in_one_second_like_the_model(self):
        short = np.ones(4, dtype=np.float32)
        batch, lens = pad_pieces([short], min_len=10)
        self.assertEqual(lens.tolist(), [10])
        self.assertEqual(batch[0].tolist(), [0, 0, 0, 1, 1, 1, 1, 0, 0, 0])

    def test_longer_pieces_are_right_padded_with_their_own_lengths(self):
        a, b = np.full(12, 2, dtype=np.float32), np.full(15, 3, dtype=np.float32)
        batch, lens = pad_pieces([a, b], min_len=10)
        self.assertEqual(tuple(batch.shape), (2, 15))
        self.assertEqual(lens.tolist(), [12, 15])
        self.assertEqual(batch[0, :12].tolist(), [2] * 12)
        self.assertEqual(batch[0, 12:].tolist(), [0] * 3)
        self.assertEqual(batch[1].tolist(), [3] * 15)


if __name__ == "__main__":
    unittest.main()
