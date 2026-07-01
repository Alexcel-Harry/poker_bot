import unittest

from poker_arena.actions import LegalActions
from poker_arena.bots import CheckCallBot, RandomLegalBot, TorchPolicyBot


class BotTests(unittest.TestCase):
    def test_check_call_bot_checks_when_available(self):
        bot = CheckCallBot()
        legal = LegalActions(
            can_fold=True,
            can_check=True,
            can_call=False,
            call_amount=0,
            min_raise_to=None,
            max_raise_to=None,
            current_bet=0,
            actor_commitment=0,
        )

        self.assertEqual(bot.choose_action(view=None, legal_actions=legal).action_type.value, "check")

    def test_random_bot_returns_legal_raise_with_seeded_choice(self):
        bot = RandomLegalBot(seed=2)
        legal = LegalActions(
            can_fold=True,
            can_check=False,
            can_call=True,
            call_amount=10,
            min_raise_to=20,
            max_raise_to=30,
            current_bet=10,
            actor_commitment=0,
        )

        action = bot.choose_action(view=None, legal_actions=legal)

        self.assertIn(action.action_type.value, {"fold", "call", "raise_to"})
        if action.action_type.value == "raise_to":
            self.assertGreaterEqual(action.total, 20)
            self.assertLessEqual(action.total, 30)

    def test_torch_policy_bot_candidates_preserve_required_integer_raises(self):
        bot = TorchPolicyBot.__new__(TorchPolicyBot)
        bot.integer_action_budget = 4
        bot.required_integer_actions = (23,)
        legal = LegalActions(
            can_fold=True,
            can_check=False,
            can_call=True,
            call_amount=10,
            min_raise_to=20,
            max_raise_to=30,
            current_bet=10,
            actor_commitment=0,
        )

        actions = bot._candidate_actions(legal)
        raise_totals = [action.total for action in actions if action.action_type.value == "raise_to"]

        self.assertIn(23, raise_totals)
        self.assertIn(20, raise_totals)
        self.assertIn(30, raise_totals)
        self.assertEqual(len(raise_totals), len(set(raise_totals)))


if __name__ == "__main__":
    unittest.main()
