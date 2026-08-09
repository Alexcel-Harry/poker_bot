import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path

import torch

from poker_arena.bots import DeepCFRAveragePolicyBot, TorchPolicyBot
from poker_arena.cfr.cuda_deep_cfr import CudaDeepCFRConfig, CudaDeepCFRTrainer, CudaReservoirBuffer
from poker_arena.cfr.gpu_prefix_branch import TensorPokerState
from poker_arena.table import Table, TableConfig


class CudaReservoirBufferTests(unittest.TestCase):
    def test_full_reservoir_replacement_updates_original_storage(self):
        device = torch.device("cpu")
        generator = torch.Generator(device=device)
        generator.manual_seed(7)
        capacity = 16
        memory = CudaReservoirBuffer(capacity, feature_dim=2, action_dim=3, device=device)
        initial_ids = torch.arange(-capacity, 0, dtype=torch.float32)
        memory.add(
            torch.stack((initial_ids, -initial_ids), dim=1),
            torch.ones((capacity, 3), dtype=torch.bool),
            torch.zeros((capacity, 3), dtype=torch.float32),
            iteration=1,
            generator=generator,
        )

        replacement_ids = torch.arange(1, 4097)
        replacement_features = torch.stack((replacement_ids, -replacement_ids), dim=1).float()
        replacement_masks = torch.stack(
            (
                replacement_ids.remainder(2) == 0,
                replacement_ids.remainder(3) == 0,
                torch.ones_like(replacement_ids, dtype=torch.bool),
            ),
            dim=1,
        )
        replacement_targets = torch.stack(
            (replacement_ids * 10, replacement_ids * 20, replacement_ids * 30),
            dim=1,
        ).float()
        memory.add(
            replacement_features,
            replacement_masks,
            replacement_targets,
            iteration=2,
            generator=generator,
        )

        recent = memory.iterations[: memory.size] == 2
        self.assertTrue(torch.any(recent).item())
        stored_ids = memory.features[: memory.size][recent, 0].long()
        self.assertTrue(torch.equal(memory.targets[: memory.size][recent, 0], stored_ids.float() * 10))
        self.assertTrue(torch.equal(memory.legal_masks[: memory.size][recent, 0], stored_ids.remainder(2) == 0))
        self.assertEqual(memory.iteration_range()[1], 2)
        self.assertEqual(memory.samples_seen, capacity + len(replacement_ids))


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
        self.assertEqual(stats["supported_player_counts"], [3])
        self.assertFalse(stats["randomized_player_counts"])
        self.assertEqual(stats["player_count_traversals"], {3: 6})
        self.assertEqual(stats["traversals"], 6)
        self.assertGreater(stats["frontier_layers"], 0)
        self.assertGreater(stats["maximum_frontier_rows"], config.parallel_traversals)
        self.assertTrue(all(count > 0 for count in stats["advantage_samples"]))
        self.assertTrue(all(count > 0 for count in stats["strategy_samples"]))
        self.assertEqual(stats["advantage_iteration_ranges"], [[1, 1]] * 3)
        self.assertEqual(stats["strategy_iteration_ranges"], [[1, 1]] * 3)
        self.assertTrue(stats["latest_iteration_retained"])
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

    def test_severely_stale_reservoirs_are_rejected_for_resume(self):
        table = TableConfig(3, 5, 10, [20, 20, 20])
        config = CudaDeepCFRConfig(
            iterations=1,
            traversals_per_player=1,
            parallel_traversals=1,
            advantage_capacity=8,
            strategy_capacity=8,
            hidden=(8,),
            advantage_train_steps=1,
            strategy_train_steps=1,
            batch_size=4,
        )
        trainer = CudaDeepCFRTrainer(table, config, self.device)
        trainer.completed_iterations = 100
        for memory in trainer.advantage_memories + trainer.strategy_memories:
            memory.size = memory.capacity
            memory.samples_seen = memory.capacity + 1
            memory.iterations.fill_(3)

        with self.assertRaisesRegex(ValueError, "pre-fix reservoir replacement bug"):
            trainer._validate_reservoir_freshness()

    def test_player_count_is_sampled_per_session_and_grouped_for_cuda(self):
        table = TableConfig(9, 5, 10, [20] * 9)
        config = CudaDeepCFRConfig(
            iterations=1,
            traversals_per_player=64,
            parallel_traversals=64,
            advantage_capacity=8,
            strategy_capacity=8,
            hidden=(8,),
            advantage_train_steps=1,
            strategy_train_steps=1,
            batch_size=4,
            random_seed=29,
            minimum_players=3,
        )
        trainer = CudaDeepCFRTrainer(table, config, self.device)
        collected: list[tuple[int, int, int]] = []

        def collect_group(traverser, batch_size, _iteration, table_config):
            collected.append((traverser, batch_size, table_config.seats))

        trainer._collect_batch = collect_group
        stats = trainer.train()

        self.assertEqual(stats["supported_player_counts"], list(range(3, 10)))
        self.assertTrue(stats["randomized_player_counts"])
        self.assertEqual(stats["traversals"], 9 * config.traversals_per_player)
        self.assertEqual(sum(stats["player_count_traversals"].values()), stats["traversals"])
        self.assertEqual(set(stats["player_count_traversals"]), set(range(3, 10)))
        self.assertTrue(all(count > 0 for count in stats["player_count_traversals"].values()))
        self.assertTrue(all(traverser < player_count for traverser, _batch, player_count in collected))

    def test_mixed_count_policy_uses_trained_seat_network_without_warning(self):
        table = TableConfig(9, 5, 10, [20] * 9)
        config = CudaDeepCFRConfig(
            iterations=1,
            traversals_per_player=1,
            parallel_traversals=1,
            advantage_capacity=8,
            strategy_capacity=8,
            hidden=(8,),
            advantage_train_steps=1,
            strategy_train_steps=1,
            batch_size=4,
            minimum_players=3,
        )
        trainer = CudaDeepCFRTrainer(table, config, self.device)
        bot = DeepCFRAveragePolicyBot.from_payload(trainer.inference_payload(), device="cpu", seed=17)
        arena = Table(TableConfig(5, 5, 10, [100] * 5, seed=47))
        state = arena.start_hand()
        actor = state.current_actor

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            action = bot.choose_action(state.player_view(actor), state.legal_actions(actor))

        self.assertEqual(bot.supported_player_counts, tuple(range(3, 10)))
        self.assertEqual(caught, [])
        arena.apply(action)

    def test_mixed_count_tensor_traversal_collects_for_highest_active_seat(self):
        table = TableConfig(9, 5, 10, [20] * 9)
        config = CudaDeepCFRConfig(
            iterations=1,
            traversals_per_player=1,
            parallel_traversals=2,
            max_traversal_depth=32,
            max_frontier_rows=8192,
            advantage_capacity=128,
            strategy_capacity=128,
            hidden=(8,),
            advantage_train_steps=1,
            strategy_train_steps=1,
            batch_size=8,
            random_seed=37,
            minimum_players=3,
        )
        trainer = CudaDeepCFRTrainer(table, config, self.device)

        trainer._collect_batch(
            traverser=4,
            batch_size=2,
            iteration=1,
            table_config=trainer._table_config_for(5),
        )

        self.assertGreater(trainer.advantage_memories[4].size, 0)
        self.assertEqual(trainer.advantage_memories[4].iteration_range(), [1, 1])
        self.assertTrue(any(memory.size > 0 for memory in trainer.strategy_memories[:5]))
        self.assertTrue(all(memory.size == 0 for memory in trainer.advantage_memories[5:]))

    def test_oversized_frontier_is_chunked_and_backpropagated(self):
        table = TableConfig(3, 5, 10, [20] * 3)
        config = CudaDeepCFRConfig(
            iterations=1,
            traversals_per_player=16,
            parallel_traversals=16,
            max_traversal_depth=32,
            max_frontier_rows=32,
            advantage_capacity=256,
            strategy_capacity=256,
            hidden=(8,),
            advantage_train_steps=1,
            strategy_train_steps=1,
            batch_size=8,
            random_seed=43,
            minimum_players=3,
        )
        trainer = CudaDeepCFRTrainer(table, config, self.device)

        trainer._collect_batch(
            traverser=0,
            batch_size=16,
            iteration=1,
            table_config=table,
        )

        self.assertGreater(trainer.frontier_chunk_splits, 0)
        self.assertGreater(trainer.maximum_projected_frontier_rows, config.max_frontier_rows)
        self.assertLessEqual(trainer.maximum_frontier_rows, config.max_frontier_rows)
        self.assertGreater(trainer.advantage_memories[0].size, 0)
        self.assertEqual(trainer.advantage_memories[0].iteration_range(), [1, 1])


if __name__ == "__main__":
    unittest.main()
