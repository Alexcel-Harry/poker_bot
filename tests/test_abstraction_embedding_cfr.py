import unittest

from poker_arena import Action, Card, Table, TableConfig
from poker_arena.abstraction import ActionAbstraction, AbstractAction, AbstractActionKind
from poker_arena.cfr import CFRTrainer, InformationSetEncoder, RegretMatcher
from poker_arena.embedding import EventContextBuilder, TrainableTrajectoryEncoder, TrajectoryEncoder


def deck(text: str) -> list[Card]:
    return [Card.from_str(part) for part in text.split()]


class ActionAbstractionTests(unittest.TestCase):
    def test_maps_legal_integer_raise_range_to_finite_representatives(self):
        table = Table(TableConfig(seats=3, small_blind=10, big_blind=20, starting_stacks=[2000, 2000, 2000], seed=3))
        state = table.start_hand()
        state = table.apply(Action.call())
        state = table.apply(Action.call())
        state = table.apply(Action.check())
        abstraction = ActionAbstraction()

        actions = abstraction.actions_for(state, state.current_actor)
        labels = [action.label for action in actions]
        concrete = [action.to_action() for action in actions]

        self.assertIn("check", labels)
        self.assertIn("min_raise", labels)
        self.assertIn("half_pot", labels)
        self.assertIn("pot", labels)
        self.assertIn("all_in", labels)
        self.assertTrue(all(action.action_type.value in {"fold", "check", "call", "raise_to"} for action in concrete))
        self.assertEqual(len(labels), len(set(labels)))

    def test_preserves_nearby_raise_amount_similarity_through_bucket_features(self):
        abstraction = ActionAbstraction()

        first = abstraction.describe_concrete_raise(total=100, pot=300, stack=2000, current_bet=20)
        second = abstraction.describe_concrete_raise(total=101, pot=300, stack=2000, current_bet=20)

        self.assertLess(abs(first.pot_ratio - second.pot_ratio), 0.01)
        self.assertLess(abs(first.stack_ratio - second.stack_ratio), 0.01)

    def test_compact_abstraction_uses_pot_after_call_and_deduplicates_totals(self):
        table = Table(TableConfig(seats=2, small_blind=5, big_blind=10, starting_stacks=[200, 200], seed=7))
        state = table.start_hand()
        state = table.apply(Action.raise_to(40))
        state = table.apply(Action.call())
        abstraction = ActionAbstraction.compact()

        actions = abstraction.actions_for(state, state.current_actor)
        raises = [action for action in actions if action.total is not None]

        self.assertIn("third_pot", [action.label for action in raises])
        self.assertIn("three_quarter_pot", [action.label for action in raises])
        self.assertIn("overbet", [action.label for action in raises])
        self.assertEqual(len({action.total for action in raises}), len(raises))
        third = next(action for action in raises if action.label == "third_pot")
        legal = state.legal_actions(state.current_actor)
        expected = legal.current_bet + round((state.total_pot + legal.call_amount) / 3.0)
        self.assertEqual(third.total, max(legal.min_raise_to, min(legal.max_raise_to, expected)))


class TrajectoryEmbeddingTests(unittest.TestCase):
    def test_event_contexts_use_exact_snapshot_state_before_actions(self):
        fixed = deck("As Ah Ks Kh Qs Qh 2c 3d 4s 5h 6c")
        table = Table(TableConfig(seats=3, small_blind=10, big_blind=20, starting_stacks=[2000, 2000, 2000], deck_order=fixed))
        state = table.start_hand()
        state = table.apply(Action.call())

        contexts = EventContextBuilder().build(state.events)
        first_action = next(context for context in contexts if context.event_type == "action")

        self.assertEqual(first_action.pot_before, 30)
        self.assertEqual(first_action.current_bet_before, 20)
        self.assertEqual(first_action.stacks_before, (2000, 1990, 1980))
        self.assertEqual(first_action.actor_stack_before, 2000)

    def test_nearby_bet_amounts_have_nearby_vectors(self):
        encoder = TrajectoryEncoder()
        event_100 = {"event_type": "action", "data": {"seat_id": 0, "action": {"type": "raise_to", "total": 100}}}
        event_101 = {"event_type": "action", "data": {"seat_id": 0, "action": {"type": "raise_to", "total": 101}}}

        vec_100 = encoder.event_vector(event_100, pot_before=300, stack_before=2000, street="preflop")
        vec_101 = encoder.event_vector(event_101, pot_before=300, stack_before=2000, street="preflop")

        self.assertEqual(len(vec_100), encoder.dimension)
        self.assertLess(encoder.distance(vec_100, vec_101), 0.01)

    def test_encodes_real_engine_history_to_fixed_size_vector(self):
        fixed = deck("As Ah Ks Kh Qs Qh 2c 3d 4s 5h 6c")
        table = Table(TableConfig(seats=3, small_blind=10, big_blind=20, starting_stacks=[2000, 2000, 2000], deck_order=fixed))
        state = table.start_hand()
        state = table.apply(Action.call())
        state = table.apply(Action.call())
        state = table.apply(Action.check())

        vector = TrajectoryEncoder().encode_events(state.events)

        self.assertEqual(len(vector), TrajectoryEncoder().dimension)
        self.assertGreater(sum(abs(value) for value in vector), 0.0)

    def test_trainable_encoder_learns_fixed_size_sequence_embedding(self):
        first = [{"event_type": "action", "data": {"seat_id": 0, "action": {"type": "raise_to", "total": 100}}}]
        second = [{"event_type": "action", "data": {"seat_id": 0, "action": {"type": "raise_to", "total": 101}}}]
        third = [{"event_type": "action", "data": {"seat_id": 1, "action": {"type": "fold", "total": None}}}]
        encoder = TrainableTrajectoryEncoder(embedding_dim=4, random_seed=7)

        before = encoder.reconstruction_loss([first, second, third])
        history = encoder.fit([first, second, third], epochs=12, learning_rate=0.05)
        after = encoder.reconstruction_loss([first, second, third])
        vec_100 = encoder.transform(first)
        vec_101 = encoder.transform(second)

        self.assertEqual(len(vec_100), 4)
        self.assertLess(after, before)
        self.assertEqual(len(history), 12)
        self.assertLess(encoder.distance(vec_100, vec_101), 0.05)


class CFRCoreTests(unittest.TestCase):
    def test_information_set_hides_opponent_private_cards(self):
        fixed = deck("As Ah Ks Kh Qs Qh 2c 3d 4s 5h 6c")
        table = Table(TableConfig(seats=3, small_blind=10, big_blind=20, starting_stacks=[2000, 2000, 2000], deck_order=fixed))
        state = table.start_hand()

        key = InformationSetEncoder().encode(state, seat_id=0)

        self.assertIn("As", key)
        self.assertIn("Kh", key)
        self.assertNotIn("Ks", key)
        self.assertNotIn("Ah", key)

    def test_information_set_can_use_trajectory_embedding_history(self):
        fixed = deck("As Ah Ks Kh Qs Qh 2c 3d 4s 5h 6c")
        table = Table(TableConfig(seats=3, small_blind=10, big_blind=20, starting_stacks=[2000, 2000, 2000], deck_order=fixed))
        state = table.start_hand()
        state = table.apply(Action.call())

        key = InformationSetEncoder(trajectory_encoder=TrajectoryEncoder(), embedding_precision=3).encode(state, seat_id=1)

        self.assertIn("trajectory=(", key)
        self.assertNotIn("history=small_blind", key)

    def test_regret_matching_uses_positive_regrets_and_uniform_fallback(self):
        matcher = RegretMatcher()
        actions = [AbstractAction(AbstractActionKind.FOLD), AbstractAction(AbstractActionKind.CALL), AbstractAction(AbstractActionKind.ALL_IN, total=200)]

        uniform = matcher.strategy_for([0.0, -1.0, 0.0], actions)
        biased = matcher.strategy_for([-2.0, 1.0, 3.0], actions)

        self.assertAlmostEqual(sum(uniform.values()), 1.0)
        self.assertAlmostEqual(uniform["fold"], 1 / 3)
        self.assertAlmostEqual(biased["call"], 0.25)
        self.assertAlmostEqual(biased["all_in"], 0.75)

    def test_mccfr_trainer_updates_strategy_tables_against_holdem_engine(self):
        trainer = CFRTrainer(
            table_config=TableConfig(seats=3, small_blind=10, big_blind=20, starting_stacks=[200, 200, 200], seed=13),
            iterations=4,
            max_actions_per_episode=12,
            random_seed=5,
        )

        result = trainer.train()

        self.assertEqual(result.iterations, 4)
        self.assertGreater(result.information_sets, 0)
        self.assertTrue(trainer.strategy_profile())


if __name__ == "__main__":
    unittest.main()
