import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from poker_arena.bots import DeepCFRAveragePolicyBot, TorchPolicyBot
from poker_arena.cfr.cuda_deep_cfr import CudaDeepCFRConfig, CudaDeepCFRTrainer
from poker_arena.cfr.gpu_prefix_branch import TensorPokerState
from poker_arena.table import Table, TableConfig


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CudaDeepCFRTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0")
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(3)

    def test_fixed_cfr_action_slots_are_legal_and_deduplicated(self):
        state = TensorPokerState.new_batch(TableConfig(2, 5, 10, [100, 100]), 16, self.device, self.generator)

        action_types, totals, valid, _legal = state.cfr_candidate_actions()

        self.assertEqual(tuple(action_types.shape), (16, 8))
        self.assertTrue(torch.all(valid.any(dim=1)))
        for row in range(state.batch_size):
            raises = totals[row][valid[row] & (action_types[row] == 3)]
            self.assertEqual(len(raises), len(torch.unique(raises)))

    def test_level_synchronous_traversal_collects_regrets_and_round_trips_policy(self):
        table = TableConfig(3, 5, 10, [20, 20, 20])
        config = CudaDeepCFRConfig(
            iterations=1,
            traversals_per_player=2,
            parallel_traversals=2,
            max_traversal_depth=32,
            max_frontier_rows=4096,
            advantage_capacity=256,
            strategy_capacity=256,
            hidden=(16,),
            advantage_train_steps=1,
            strategy_train_steps=1,
            batch_size=16,
            random_seed=23,
        )
        trainer = CudaDeepCFRTrainer(table, config, self.device)

        stats = trainer.train()

        self.assertEqual(stats["num_players"], 3)
        self.assertEqual(stats["traversals"], 6)
        self.assertGreater(stats["frontier_layers"], 0)
        self.assertGreater(stats["maximum_frontier_rows"], config.parallel_traversals)
        self.assertTrue(all(count > 0 for count in stats["advantage_samples"]))
        self.assertTrue(all(count > 0 for count in stats["strategy_samples"]))
        self.assertEqual(stats["depth_limit_rollouts"], 0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy_path = root / "cuda_deep_cfr_policy.pt"
            snapshot_path = root / "cuda_deep_cfr_snapshot.pt"
            trainer.save_inference_policy(policy_path)
            trainer.save_snapshot(snapshot_path)
            bot = TorchPolicyBot.from_checkpoint(policy_path, device="cpu")
            resume_config = replace(
                config,
                iterations=7,
                parallel_traversals=1,
                max_frontier_rows=8192,
            )
            restored = CudaDeepCFRTrainer(table, resume_config, self.device)
            restored.load_snapshot(snapshot_path)
            self.assertFalse((root / ".cuda_deep_cfr_snapshot.pt.tmp").exists())

        self.assertIsInstance(bot, DeepCFRAveragePolicyBot)
        self.assertEqual(bot.num_players, 3)
        self.assertEqual(restored.traversals, trainer.traversals)
        self.assertEqual(restored.advantage_memories[0].samples_seen, trainer.advantage_memories[0].samples_seen)
        arena = Table(TableConfig(3, 5, 10, [20, 20, 20], seed=31))
        state = arena.start_hand()
        actor = state.current_actor
        action = bot.choose_action(state.player_view(actor), state.legal_actions(actor))
        arena.apply(action)


if __name__ == "__main__":
    unittest.main()
