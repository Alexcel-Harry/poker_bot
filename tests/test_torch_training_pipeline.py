import importlib.util
import tempfile
import unittest
from pathlib import Path

from poker_arena import Action, Table, TableConfig
from poker_arena.cfr import ActionEmbedding
from poker_arena.cfr import PrefixBranchCFRTrainer, PrefixBranchTrainingConfig
from poker_arena.cfr.torch_model import (
    CHECKPOINT_VERSION,
    StateFeatureEncoder,
    TorchCheckpointMetadata,
    TorchTrainingSample,
    build_checkpoint_payload,
    validate_checkpoint_payload,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class TorchTrainingPurePythonTests(unittest.TestCase):
    def betting_state(self):
        table = Table(TableConfig(seats=3, small_blind=10, big_blind=20, starting_stacks=[2000, 2000, 2000], seed=31))
        state = table.start_hand()
        state = table.apply(Action.call())
        state = table.apply(Action.call())
        state = table.apply(Action.check())
        return state, state.current_actor

    def test_state_feature_encoder_dimension_is_stable(self):
        state, actor = self.betting_state()
        encoder = StateFeatureEncoder()

        features = encoder.encode_state(state, actor)

        self.assertEqual(len(features), encoder.dimension)
        self.assertGreater(sum(abs(value) for value in features), 0.0)

    def test_checkpoint_metadata_validates_required_fields(self):
        metadata = TorchCheckpointMetadata(
            state_dim=StateFeatureEncoder().dimension,
            action_dim=ActionEmbedding.dimension_without_trajectory + 14,
            hidden=(64, 32),
            dropout=0.0,
            table_defaults={"seats": 3},
            action_sampler={"integer_action_budget": 8},
            training={"iterations": 1},
        )

        payload = build_checkpoint_payload(metadata=metadata, model_state_dict={"out.weight": "placeholder"})

        self.assertEqual(payload["checkpoint_version"], CHECKPOINT_VERSION)
        validate_checkpoint_payload(payload)
        with self.assertRaises(ValueError):
            validate_checkpoint_payload({"checkpoint_version": CHECKPOINT_VERSION})

    def test_torch_training_sample_preserves_integer_action_metadata(self):
        sample = TorchTrainingSample(
            state_features=[0.1, 0.2],
            trajectory_features=[0.3],
            action_features=[0.4],
            action={"type": "raise_to", "total": 123},
            target_utility=5.0,
            weight=1.0,
        )

        self.assertEqual(sample.action["type"], "raise_to")
        self.assertEqual(sample.action["total"], 123)

    def test_prefix_branch_trainer_collects_torch_ready_samples(self):
        config = PrefixBranchTrainingConfig(
            branch_width=2,
            branch_depth=1,
            integer_action_budget=4,
            max_actions_per_episode=2,
            random_seed=19,
        )
        trainer = PrefixBranchCFRTrainer(
            table_config=TableConfig(seats=3, small_blind=10, big_blind=20, starting_stacks=[200, 200, 200], seed=17),
            config=config,
        )

        trainer.train(iterations=1)

        self.assertTrue(trainer.torch_training_samples)
        sample = trainer.torch_training_samples[0]
        self.assertEqual(len(sample.state_features), StateFeatureEncoder().dimension)
        self.assertIn(sample.action["type"], {"fold", "check", "call", "raise_to"})


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class TorchTrainingWithTorchTests(unittest.TestCase):
    def test_action_value_net_forward_shape(self):
        import torch

        from poker_arena.cfr.torch_model import ActionValueNet

        model = ActionValueNet(input_dim=5, hidden=(8, 4))
        output = model(torch.zeros((3, 5), dtype=torch.float32))

        self.assertEqual(tuple(output.shape), (3, 1))

    def test_replay_buffer_training_step_and_checkpoint_round_trip(self):
        import torch

        from poker_arena.bots import TorchPolicyBot
        from poker_arena.cfr.torch_model import (
            ActionValueNet,
            TorchReplayBuffer,
            save_checkpoint,
            train_value_model,
        )

        samples = [
            TorchTrainingSample([0.0, 0.1], [0.2], [0.3], {"type": "fold", "total": None}, -1.0, 1.0),
            TorchTrainingSample([0.1, 0.2], [0.3], [0.4], {"type": "call", "total": None}, 1.0, 1.0),
            TorchTrainingSample([0.2, 0.3], [0.4], [0.5], {"type": "raise_to", "total": 123}, 2.0, 1.0),
        ]
        buffer = TorchReplayBuffer(samples)
        model = ActionValueNet(input_dim=4, hidden=(8,))
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}

        stats = train_value_model(model, buffer, device=torch.device("cpu"), epochs=2, batch_size=2, learning_rate=0.01)

        self.assertTrue(stats["loss"])
        self.assertTrue(any(not torch.equal(before[name], value) for name, value in model.state_dict().items()))

        metadata = TorchCheckpointMetadata(
            state_dim=2,
            action_dim=2,
            hidden=(8,),
            dropout=0.0,
            table_defaults={"seats": 3},
            action_sampler={"integer_action_budget": 8},
            training={"iterations": 1},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy.pt"
            save_checkpoint(path, model, metadata, stats)
            bot = TorchPolicyBot.from_checkpoint(path, device="cpu")
            self.assertIsNotNone(bot)


if __name__ == "__main__":
    unittest.main()
