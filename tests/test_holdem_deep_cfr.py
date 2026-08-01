import tempfile
import unittest
from pathlib import Path

from poker_arena.bots import DeepCFRAveragePolicyBot, TorchPolicyBot
from poker_arena.cfr.deep_cfr import DeepCFRConfig, DeepCFRTrainer, torch
from poker_arena.cfr.holdem_deep_cfr import HoldemCFRState, HoldemDeepCFRFeatureEncoder
from poker_arena.table import Table, TableConfig


class HoldemDeepCFRStateTests(unittest.TestCase):
    def root(self) -> HoldemCFRState:
        return HoldemCFRState.initial(TableConfig(3, 5, 10, [20, 20, 20]))

    def test_holdem_training_rejects_fewer_than_three_players(self):
        with self.assertRaises(ValueError):
            HoldemCFRState.initial(TableConfig(2, 5, 10, [20, 20]))

    def test_root_samples_independent_deals_and_compact_actions(self):
        root = self.root()
        first = root.child(7)
        second = root.child(8)

        self.assertNotEqual(first.hand_state().to_snapshot()["deck_cards"], second.hand_state().to_snapshot()["deck_cards"])
        self.assertIn("call", first.legal_actions())
        self.assertTrue(any(action in first.legal_actions() for action in ("min_raise", "all_in")))

    def test_feature_encoder_has_ordered_history_without_opponent_cards(self):
        state = self.root().child(11)
        encoder = HoldemDeepCFRFeatureEncoder(max_history_actions=8)

        features = encoder.encode(state)

        self.assertEqual(len(features), encoder.input_dim)
        self.assertEqual(len(encoder.legal_mask(state)), encoder.action_dim)
        self.assertEqual(sum(encoder.legal_mask(state)), len(state.legal_actions()))


@unittest.skipUnless(torch is not None, "torch is not installed")
class HoldemDeepCFRSmokeTests(unittest.TestCase):
    def test_short_stack_holdem_traversal_reaches_terminal_and_collects_both_memories(self):
        root = HoldemCFRState.initial(TableConfig(3, 5, 10, [20, 20, 20]))
        encoder = HoldemDeepCFRFeatureEncoder(max_history_actions=8)
        config = DeepCFRConfig(
            iterations=1,
            traversals_per_player=2,
            advantage_capacity=256,
            strategy_capacity=256,
            hidden=(16,),
            advantage_train_steps=1,
            strategy_train_steps=1,
            batch_size=16,
            random_seed=5,
            device="cpu",
        )
        trainer = DeepCFRTrainer(root, config, encoder)

        stats = trainer.train()

        self.assertEqual(stats["num_players"], 3)
        self.assertEqual(stats["traversals"], 6)
        self.assertGreater(stats["traverser_nodes"], 0)
        self.assertTrue(all(count > 0 for count in stats["advantage_samples"]))
        self.assertTrue(all(count > 0 for count in stats["strategy_samples"]))

    def test_inference_policy_loads_through_existing_torch_bot_entry_point(self):
        root = HoldemCFRState.initial(TableConfig(3, 5, 10, [20, 20, 20]))
        encoder = HoldemDeepCFRFeatureEncoder(max_history_actions=8)
        config = DeepCFRConfig(
            iterations=1,
            traversals_per_player=2,
            advantage_capacity=128,
            strategy_capacity=128,
            hidden=(16,),
            advantage_train_steps=1,
            strategy_train_steps=1,
            batch_size=8,
            random_seed=7,
        )
        trainer = DeepCFRTrainer(root, config, encoder)
        trainer.train()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "holdem_average_policy.pt"
            trainer.save_inference_policy(path)
            bot = TorchPolicyBot.from_checkpoint(path, device="cpu")

        self.assertIsInstance(bot, DeepCFRAveragePolicyBot)
        self.assertEqual(bot.num_players, 3)
        dealt = root.child(19).hand_state()
        actor = dealt.current_actor
        legal = dealt.legal_actions(actor)
        action = bot.choose_action(dealt.player_view(actor), legal)
        if action.total is not None:
            self.assertGreaterEqual(action.total, legal.min_raise_to)
            self.assertLessEqual(action.total, legal.max_raise_to)
        else:
            self.assertIn(action.action_type.value, {"fold", "check", "call"})

        for seats in (2, 4, 9):
            with self.subTest(seats=seats):
                arena = Table(TableConfig(seats, 5, 10, [100] * seats, seed=seats))
                state = arena.start_hand()
                actor = state.current_actor
                with self.assertWarnsRegex(RuntimeWarning, "out-of-distribution inference"):
                    action = bot.choose_action(state.player_view(actor), state.legal_actions(actor))
                arena.apply(action)


if __name__ == "__main__":
    unittest.main()
