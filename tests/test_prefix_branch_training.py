import unittest

from examples.train_cfr import run_training
from poker_arena import Action, Table, TableConfig
from poker_arena.cfr import (
    ActionEmbedding,
    EmbeddingCoverageIndex,
    IntegerActionSampler,
    PrefixBranchCFRTrainer,
    PrefixBranchExplorer,
    PrefixBranchTrainingConfig,
)
from poker_arena.embedding import TrajectoryEncoder


class PrefixBranchTrainingTests(unittest.TestCase):
    def betting_state(self):
        table = Table(TableConfig(seats=3, small_blind=10, big_blind=20, starting_stacks=[2000, 2000, 2000], seed=31))
        state = table.start_hand()
        state = table.apply(Action.call())
        state = table.apply(Action.call())
        state = table.apply(Action.check())
        return table, state, state.current_actor

    def test_integer_action_sampler_preserves_arbitrary_legal_raise_amounts(self):
        _table, state, actor = self.betting_state()
        sampler = IntegerActionSampler(random_seed=7)

        actions = sampler.sample(state, actor, budget=8, required_amounts=[123, 124, 99999])
        totals = [action.total for action in actions if action.action_type.value == "raise_to"]

        self.assertIn(123, totals)
        self.assertIn(124, totals)
        self.assertNotIn(99999, totals)
        self.assertEqual(len(totals), len(set(totals)))
        self.assertTrue(any(action.action_type.value == "check" for action in actions))

    def test_integer_action_sampler_preserves_required_amounts_even_with_small_budget(self):
        _table, state, actor = self.betting_state()
        sampler = IntegerActionSampler(random_seed=7)

        actions = sampler.sample(state, actor, budget=3, required_amounts=[123, 124])
        totals = [action.total for action in actions if action.action_type.value == "raise_to"]

        self.assertIn(123, totals)
        self.assertIn(124, totals)

    def test_action_embedding_keeps_nearby_integer_bets_close(self):
        _table, state, actor = self.betting_state()
        legal = state.legal_actions(actor)
        trajectory = TrajectoryEncoder().encode_events(state.events)
        embedding = ActionEmbedding()

        near_a = embedding.encode(Action.raise_to(123), legal, trajectory)
        near_b = embedding.encode(Action.raise_to(124), legal, trajectory)
        far = embedding.encode(Action.raise_to(legal.max_raise_to), legal, trajectory)

        self.assertLess(embedding.distance(near_a, near_b), embedding.distance(near_a, far))

    def test_action_embedding_adds_pot_relative_sizes_without_breaking_legacy_checkpoints(self):
        _table, state, actor = self.betting_state()
        legal = state.legal_actions(actor)
        action = Action.raise_to(123)

        vector = ActionEmbedding().encode(action, legal, pot=state.total_pot)
        legacy = ActionEmbedding(include_pot_features=False).encode(action, legal, pot=state.total_pot)

        self.assertEqual(len(vector), ActionEmbedding.dimension_without_trajectory)
        self.assertEqual(len(legacy), ActionEmbedding.legacy_dimension)
        self.assertAlmostEqual(vector[12], (123 - legal.current_bet) / state.total_pot)
        self.assertAlmostEqual(vector[13], 123 / (state.total_pot + legal.call_amount))

    def test_prefix_branch_expansion_does_not_mutate_original_table(self):
        table, state, actor = self.betting_state()
        before = state.to_snapshot()
        actions = [Action.check(), Action.raise_to(123)]
        explorer = PrefixBranchExplorer(max_workers=1, random_seed=11)

        results = explorer.expand(table, actor, actions, depth=1, width=2)

        self.assertEqual(table.current_hand.to_snapshot(), before)
        self.assertEqual([result.action for result in results], actions)
        self.assertTrue(all(len(result.utilities) == table.config.seats for result in results))
        self.assertTrue(all(result.action_embedding for result in results))
        self.assertTrue(all(abs(sum(result.utilities)) < 1e-9 for result in results))

    def test_embedding_coverage_scores_seen_regions_as_less_novel(self):
        _table, state, actor = self.betting_state()
        legal = state.legal_actions(actor)
        trajectory = TrajectoryEncoder().encode_events(state.events)
        embedding = ActionEmbedding()
        vector = embedding.encode(Action.raise_to(123), legal, trajectory)
        coverage = EmbeddingCoverageIndex()

        novel_before = coverage.novelty(vector)
        coverage.record(vector)
        novel_after = coverage.novelty(vector)

        self.assertGreater(novel_before, novel_after)

    def test_embedding_coverage_is_bounded(self):
        coverage = EmbeddingCoverageIndex(max_vectors=2)

        coverage.record([0.0])
        coverage.record([1.0])
        coverage.record([2.0])

        self.assertEqual(len(coverage.vectors), 2)
        self.assertIn([2.0], coverage.vectors)

    def test_prefix_branch_trainer_smoke_uses_neighbor_smoothing_disabled_by_default(self):
        config = PrefixBranchTrainingConfig(
            branch_width=4,
            branch_depth=1,
            integer_action_budget=4,
            max_actions_per_episode=8,
            random_seed=19,
        )
        trainer = PrefixBranchCFRTrainer(
            table_config=TableConfig(seats=3, small_blind=10, big_blind=20, starting_stacks=[200, 200, 200], seed=17),
            config=config,
        )

        result = trainer.train(iterations=2)

        self.assertEqual(config.neighbor_weight, 0.0)
        self.assertEqual(result.iterations, 2)
        self.assertGreater(result.episodes, 0)
        self.assertTrue(trainer.training_samples)
        self.assertTrue(all(-1.0 <= sample.target_utility <= 2.0 for sample in trainer.torch_training_samples))

    def test_terminal_rollouts_are_the_safe_default(self):
        config = PrefixBranchTrainingConfig()

        self.assertEqual(config.branch_depth, 0)
        with self.assertRaises(ValueError):
            PrefixBranchTrainingConfig(branch_width=0)

    def test_prefix_branch_trainer_only_tracks_actions_that_are_expanded(self):
        config = PrefixBranchTrainingConfig(
            branch_width=2,
            branch_depth=1,
            integer_action_budget=8,
            max_actions_per_episode=1,
            random_seed=23,
        )
        trainer = PrefixBranchCFRTrainer(
            table_config=TableConfig(seats=3, small_blind=10, big_blind=20, starting_stacks=[200, 200, 200], seed=29),
            config=config,
        )

        trainer.train(iterations=1)

        self.assertTrue(trainer.nodes)
        self.assertTrue(all(len(node.labels) <= config.branch_width for node in trainer.nodes.values()))

    def test_example_training_uses_prefix_branch_pipeline(self):
        result = run_training(iterations=1)

        self.assertTrue(result["prefix_branch"])
        self.assertGreater(result["training_samples"], 0)


if __name__ == "__main__":
    unittest.main()
