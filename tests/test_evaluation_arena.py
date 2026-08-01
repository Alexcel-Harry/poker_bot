import unittest

from poker_arena.bots import CheckCallBot
from poker_arena.evaluation_arena import play_duplicate_match, play_rotating_match


class DuplicateEvaluationTests(unittest.TestCase):
    def test_identical_bots_cancel_exactly_under_seat_swapped_deals(self):
        result = play_duplicate_match(CheckCallBot, CheckCallBot, deals=20, random_seed=11)

        self.assertEqual(result.hands, 40)
        self.assertAlmostEqual(result.mean_chips_per_hand, 0.0)
        self.assertAlmostEqual(result.standard_error, 0.0)
        self.assertEqual(result.confidence_95, (0.0, 0.0))
        self.assertAlmostEqual(result.big_blinds_per_100, 0.0)
        self.assertEqual(result.big_blinds_per_100_confidence_95, (0.0, 0.0))
        self.assertGreater(result.telemetry["first"]["decisions"], 0)
        self.assertIn("preflop", result.telemetry["first"]["by_street"])
        self.assertIn("preflop:call", result.telemetry["first"]["by_street_action"])

    def test_match_validates_deal_count(self):
        with self.assertRaises(ValueError):
            play_duplicate_match(CheckCallBot, CheckCallBot, deals=0)

    def test_three_player_rotations_cancel_for_identical_bots(self):
        result = play_rotating_match(CheckCallBot, CheckCallBot, deals=10, seats=3, random_seed=29)

        self.assertEqual(result.hands, 30)
        self.assertAlmostEqual(result.mean_chips_per_hand, 0.0)
        self.assertAlmostEqual(result.big_blinds_per_100, 0.0)
        self.assertEqual(len(result.paired_scores), 10)
        self.assertTrue(all(abs(score) < 1e-9 for score in result.paired_scores))
        self.assertGreater(result.telemetry["tracked"]["decisions"], 0)


if __name__ == "__main__":
    unittest.main()
