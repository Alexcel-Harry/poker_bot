import random
import unittest

import torch

from poker_arena import Action, Table
from poker_arena.cards import full_deck
from poker_arena.cfr.gpu_prefix_branch import (
    GpuPrefixBranchTrainingConfig,
    TensorPokerState,
    evaluate_seven_card_hands,
)
from poker_arena.cfr.prefix_branch import ActionEmbedding
from poker_arena.cfr.torch_model import StateFeatureEncoder
from poker_arena.embedding import TrajectoryEncoder
from poker_arena.evaluator import evaluate_best
from poker_arena.table import TableConfig


class TensorPokerEngineTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(7)
        self.table_config = TableConfig(3, 10, 20, [200, 200, 200])

    def test_training_config_rejects_a_branch_width_that_cannot_include_raises(self):
        with self.assertRaises(ValueError):
            GpuPrefixBranchTrainingConfig(branch_width=2)

    def test_new_batch_has_independent_complete_decks_and_legal_candidates(self):
        state = TensorPokerState.new_batch(self.table_config, 8, self.device, self.generator)

        self.assertEqual(tuple(state.stacks.shape), (8, 3))
        self.assertTrue(torch.all(state.deck.sort(dim=1).values == torch.arange(52)))
        action_types, totals, valid, _legal = state.candidate_actions(12, self.generator)
        self.assertEqual(tuple(action_types.shape), (8, 12))
        self.assertTrue(torch.all(valid.any(dim=1)))
        self.assertTrue(torch.all((totals[action_types == 3]) > state.current_bet[:, None].expand_as(totals)[action_types == 3]))

    def test_checkpoint_branching_does_not_mutate_the_prefix(self):
        state = TensorPokerState.new_batch(self.table_config, 4, self.device, self.generator)
        before = state.stacks.clone()
        action_types, totals, valid, _legal = state.candidate_actions(8, self.generator)
        branches = state.repeat_interleave(8)
        branches.terminal[~valid.reshape(-1)] = True

        branches.apply_actions(action_types.reshape(-1), totals.reshape(-1), valid.reshape(-1))

        self.assertTrue(torch.equal(state.stacks, before))
        self.assertEqual(branches.batch_size, 32)

    def test_tensor_model_features_match_the_deployed_python_encoders(self):
        batch = TensorPokerState.new_batch(self.table_config, 16, self.device, self.generator)
        row = torch.nonzero(batch.button == 0, as_tuple=False)[0, 0]
        state = batch.index_select(row.reshape(1))
        cards = full_deck()
        table = Table(TableConfig(3, 10, 20, [200, 200, 200], deck_order=[cards[index] for index in state.deck[0].tolist()]))
        reference = table.start_hand()
        actor = reference.current_actor
        self.assertIsNotNone(actor)

        expected_state = StateFeatureEncoder().encode_state(reference, actor)
        expected_trajectory = TrajectoryEncoder().encode_events(
            event for event in reference.events if event.event_type != "snapshot"
        )
        self.assertTrue(torch.allclose(state.state_features()[0], torch.tensor(expected_state), atol=1e-6))
        self.assertTrue(torch.allclose(state.trajectory_features()[0], torch.tensor(expected_trajectory), atol=1e-6))

        action_types, totals, valid, legal = state.candidate_actions(8, self.generator)
        tensor_features = state.action_features(action_types, totals, legal)[0]
        python_legal = reference.legal_actions(actor)
        for column in torch.nonzero(valid[0], as_tuple=False).squeeze(1).tolist():
            action_type = int(action_types[0, column])
            action = (
                Action.fold()
                if action_type == 0
                else Action.check()
                if action_type == 1
                else Action.call()
                if action_type == 2
                else Action.raise_to(int(totals[0, column]))
            )
            expected = ActionEmbedding().encode(action, python_legal)
            self.assertTrue(torch.allclose(tensor_features[column], torch.tensor(expected), atol=1e-6))

    def test_parallel_random_rollouts_settle_to_zero_sum_terminal_utilities(self):
        state = TensorPokerState.new_batch(self.table_config, 32, self.device, self.generator)

        state.rollout(128, self.generator)
        utilities = state.utilities()

        self.assertTrue(torch.all(state.terminal))
        self.assertTrue(torch.allclose(utilities.sum(dim=1), torch.zeros(32)))

    def test_tensor_hand_evaluator_matches_reference_evaluator(self):
        deck = full_deck()
        rng = random.Random(11)
        hands = [rng.sample(range(52), 7) for _ in range(64)]
        tensor_values = evaluate_seven_card_hands(torch.tensor(hands, dtype=torch.int64))
        reference_values = [evaluate_best([deck[index] for index in hand]) for hand in hands]

        for first in range(len(hands)):
            for second in range(len(hands)):
                tensor_order = int(torch.sign(tensor_values[first] - tensor_values[second]).item())
                reference_order = (reference_values[first] > reference_values[second]) - (reference_values[first] < reference_values[second])
                self.assertEqual(tensor_order, reference_order)

    def test_tensor_betting_and_side_pots_match_the_reference_engine(self):
        unequal = TableConfig(3, 10, 20, [100, 200, 300])
        batch = TensorPokerState.new_batch(unequal, 16, self.device, self.generator)
        row = torch.nonzero(batch.button == 0, as_tuple=False)[0, 0]
        state = batch.index_select(row.reshape(1))
        cards = full_deck()
        table = Table(TableConfig(3, 10, 20, [100, 200, 300], deck_order=[cards[index] for index in state.deck[0].tolist()]))
        table.start_hand()
        actions = [
            Action.raise_to(100),
            Action.raise_to(200),
            Action.call(),
            Action.check(),
            Action.check(),
            Action.check(),
        ]
        action_ids = {"fold": 0, "check": 1, "call": 2, "raise_to": 3}

        for action in actions:
            reference = table.apply(action)
            state.apply_actions(
                torch.tensor([action_ids[action.action_type.value]], dtype=torch.int64),
                torch.tensor([action.total or 0], dtype=torch.int64),
            )
            self.assertEqual(state.stacks[0].tolist(), [player.stack for player in reference.players])
            self.assertEqual(state.total_committed[0].tolist(), list(reference.total_committed.values()))
            self.assertEqual(int(state.current_actor[0]), reference.current_actor if reference.current_actor is not None else -1)

        self.assertTrue(bool(state.terminal[0]))
        self.assertEqual(state.stacks[0].tolist(), [player.stack for player in table.current_hand.players])


if __name__ == "__main__":
    unittest.main()
