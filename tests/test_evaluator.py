import unittest

from poker_arena.cards import Card
from poker_arena.evaluator import HandCategory, evaluate_best


def cards(text: str) -> list[Card]:
    return [Card.from_str(part) for part in text.split()]


class EvaluatorTests(unittest.TestCase):
    def test_evaluator_orders_major_hand_categories(self):
        straight_flush = evaluate_best(cards("As Ks Qs Js Ts 2d 3c"))
        quads = evaluate_best(cards("Ah Ad Ac As Kd Qc 2h"))
        full_house = evaluate_best(cards("Kh Kd Kc Qs Qh 2c 3d"))
        flush = evaluate_best(cards("Ah Jh 9h 5h 2h Kc Qd"))
        straight = evaluate_best(cards("9c 8d 7s 6h 5c Ah 2d"))

        self.assertEqual(straight_flush.category, HandCategory.STRAIGHT_FLUSH)
        self.assertGreater(straight_flush, quads)
        self.assertGreater(quads, full_house)
        self.assertGreater(full_house, flush)
        self.assertGreater(flush, straight)

    def test_evaluator_uses_best_five_of_seven_and_kickers(self):
        aces_with_king = evaluate_best(cards("As Ad Kc 9h 7d 3s 2c"))
        aces_with_queen = evaluate_best(cards("Ah Ac Qd 9s 7c 3d 2h"))

        self.assertEqual(aces_with_king.category, HandCategory.ONE_PAIR)
        self.assertGreater(aces_with_king, aces_with_queen)

    def test_wheel_straight_is_low_ace(self):
        hand = evaluate_best(cards("Ah 2d 3c 4s 5h Kd Qc"))

        self.assertEqual(hand.category, HandCategory.STRAIGHT)
        self.assertEqual(hand.tiebreakers, (5,))


if __name__ == "__main__":
    unittest.main()
