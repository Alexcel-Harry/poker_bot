import tempfile
import unittest
from pathlib import Path

from poker_arena.cfr.deep_cfr import DeepCFRConfig, DeepCFRTrainer, ReservoirBuffer, torch
from poker_arena.cfr.evaluation import exploitability
from poker_arena.cfr.toy_games import KuhnPokerState


class ReservoirBufferTests(unittest.TestCase):
    def test_reservoir_is_bounded_and_counts_every_sample(self):
        buffer = ReservoirBuffer[int](capacity=10, random_seed=7)

        for value in range(100):
            buffer.add(value)

        self.assertEqual(len(buffer), 10)
        self.assertEqual(buffer.samples_seen, 100)
        self.assertTrue(any(value < 50 for value in buffer.samples))
        self.assertTrue(any(value >= 50 for value in buffer.samples))


@unittest.skipUnless(torch is not None, "torch is not installed")
class DeepCFRTests(unittest.TestCase):
    def config(self) -> DeepCFRConfig:
        return DeepCFRConfig(
            iterations=8,
            traversals_per_player=64,
            advantage_capacity=10_000,
            strategy_capacity=10_000,
            hidden=(32,),
            advantage_train_steps=60,
            strategy_train_steps=200,
            batch_size=128,
            learning_rate=0.01,
            random_seed=13,
            device="cpu",
        )

    def test_external_sampling_branches_recursively_and_trains_mixed_policy(self):
        trainer = DeepCFRTrainer(KuhnPokerState.initial(), self.config())

        stats = trainer.train()
        profile = trainer.average_strategy_profile()

        self.assertGreater(stats["traverser_nodes"], stats["traversals"])
        self.assertGreater(stats["sampled_opponent_nodes"], 0)
        self.assertGreater(stats["sampled_chance_nodes"], 0)
        self.assertGreaterEqual(stats["maximum_depth"], 3)
        self.assertTrue(all(count > 0 for count in stats["advantage_samples"]))
        self.assertTrue(all(count > 0 for count in stats["strategy_samples"]))
        self.assertTrue(all(abs(sum(strategy.values()) - 1.0) < 1e-6 for strategy in profile.values()))
        self.assertLess(exploitability(KuhnPokerState.initial(), profile).exploitability, 0.2)

    def test_training_snapshot_round_trip_resumes_counters_and_policy(self):
        root = KuhnPokerState.initial()
        trainer = DeepCFRTrainer(root, self.config())
        trainer.train(iterations=2)
        before = trainer.average_strategy_profile()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deep_cfr_snapshot.pt"
            trainer.save_snapshot(path)
            restored = DeepCFRTrainer.load_snapshot(root, path, device="cpu")

        self.assertEqual(restored.completed_iterations, trainer.completed_iterations)
        self.assertEqual(restored.traversals, trainer.traversals)
        self.assertEqual(restored.average_strategy_profile(), before)
        restored.train(iterations=1)
        self.assertEqual(restored.completed_iterations, trainer.completed_iterations + 1)


if __name__ == "__main__":
    unittest.main()
