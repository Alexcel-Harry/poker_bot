import unittest

from poker_arena.cfr.evaluation import TabularCFRTrainer, best_response_value, exploitability, expected_utility
from poker_arena.cfr.toy_games import KuhnPokerState, LeducPokerState


class ToyGameStateTests(unittest.TestCase):
    def test_kuhn_terminal_payoffs_are_zero_sum(self):
        state = KuhnPokerState(cards=(2, 0), history="bc")

        self.assertEqual(state.utility(0), 2.0)
        self.assertEqual(state.utility(0) + state.utility(1), 0.0)

    def test_leduc_pair_beats_higher_unpaired_card(self):
        state = LeducPokerState(
            private_cards=(0, 4),
            public_card=1,
            contributions=(3, 3),
            terminal=True,
        )

        self.assertEqual(state.utility(0), 3.0)
        self.assertEqual(state.utility(1), -3.0)

    def test_uniform_kuhn_profile_has_exact_best_response(self):
        root = KuhnPokerState.initial()

        self.assertAlmostEqual(expected_utility(root, {}, 0), 0.125)
        self.assertGreater(best_response_value(root, {}, 0), 0.0)
        self.assertGreater(exploitability(root, {}).exploitability, 0.1)


class TabularCFRCorrectnessTests(unittest.TestCase):
    def test_kuhn_exploitability_converges(self):
        root = KuhnPokerState.initial()
        trainer = TabularCFRTrainer(root)

        profile = trainer.train(5_000)
        result = exploitability(root, profile)

        self.assertLess(result.exploitability, 0.02)
        self.assertAlmostEqual(result.expected_values[0], -1.0 / 18.0, delta=0.02)

    def test_leduc_exploitability_improves(self):
        root = LeducPokerState.initial()
        trainer = TabularCFRTrainer(root)
        initial = exploitability(root, {}).exploitability

        profile = trainer.train(250)
        trained = exploitability(root, profile).exploitability

        self.assertLess(trained, initial * 0.5)


if __name__ == "__main__":
    unittest.main()
