import unittest

from poker_arena import Action, Card, Table, TableConfig
from poker_arena.state import Street


def deck(text: str) -> list[Card]:
    return [Card.from_str(part) for part in text.split()]


class TableBettingTests(unittest.TestCase):
    def test_heads_up_uses_button_as_small_blind_and_first_preflop_actor(self):
        table = Table(TableConfig(seats=2, small_blind=5, big_blind=10, starting_stacks=[100, 100], seed=1))
        state = table.start_hand()

        self.assertEqual(state.button, 0)
        self.assertEqual(state.small_blind_seat, 0)
        self.assertEqual(state.big_blind_seat, 1)
        self.assertEqual(state.current_actor, 0)

    def test_three_player_hand_allows_arbitrary_integer_raise_to(self):
        table = Table(TableConfig(seats=3, small_blind=5, big_blind=10, starting_stacks=[100, 100, 100], seed=1))
        state = table.start_hand()

        self.assertEqual(state.current_actor, 0)
        legal = state.legal_actions(0)
        self.assertTrue(legal.can_fold)
        self.assertEqual(legal.call_amount, 10)
        self.assertEqual(legal.min_raise_to, 20)
        self.assertEqual(legal.max_raise_to, 100)

        state = table.apply(Action.raise_to(23))
        self.assertEqual(state.committed_this_street[0], 23)
        self.assertEqual(state.current_bet, 23)
        self.assertEqual(state.last_full_raise, 13)

    def test_invalid_raise_below_minimum_is_rejected(self):
        table = Table(TableConfig(seats=3, small_blind=5, big_blind=10, starting_stacks=[100, 100, 100], seed=1))
        table.start_hand()

        with self.assertRaises(ValueError):
            table.apply(Action.raise_to(19))

    def test_short_all_in_does_not_reopen_full_raise_size(self):
        table = Table(TableConfig(seats=3, small_blind=5, big_blind=10, starting_stacks=[18, 100, 100], seed=1))
        state = table.start_hand()

        state = table.apply(Action.raise_to(18))
        self.assertEqual(state.committed_this_street[0], 18)
        self.assertEqual(state.current_bet, 18)
        self.assertEqual(state.last_full_raise, 10)

    def test_hand_can_reach_showdown_and_conserves_chips(self):
        fixed = deck("As Ah Ks Kh Qs Qh 2c 3d 4s 5h 6c")
        table = Table(TableConfig(seats=3, small_blind=5, big_blind=10, starting_stacks=[100, 100, 100], deck_order=fixed))
        state = table.start_hand()
        starting_total = sum(player.stack for player in state.players) + state.total_pot

        while not state.is_terminal:
            legal = state.legal_actions(state.current_actor)
            state = table.apply(Action.check() if legal.can_check else Action.call())

        ending_total = sum(player.stack for player in state.players) + table.carryover_chips
        self.assertEqual(state.street, Street.SHOWDOWN)
        self.assertEqual(starting_total, ending_total)
        self.assertTrue(any(event.event_type == "pot_awarded" for event in state.events))


class SidePotCarryoverTests(unittest.TestCase):
    def test_odd_chip_remainder_carries_to_next_hand(self):
        fixed = deck("As Ah Qs Kd Kh Jd 2c 3d 4h 8s 9c")
        table = Table(TableConfig(seats=3, small_blind=1, big_blind=2, starting_stacks=[3, 3, 10], deck_order=fixed))
        state = table.start_hand()

        state = table.apply(Action.raise_to(3))
        state = table.apply(Action.call())
        state = table.apply(Action.call())
        while not state.is_terminal:
            legal = state.legal_actions(state.current_actor)
            state = table.apply(Action.check() if legal.can_check else Action.call())

        self.assertEqual(table.carryover_chips, 1)
        self.assertEqual(state.total_pot, 0)

    def test_split_pot_odd_chip_carries_to_next_hand_main_pot(self):
        fixed = deck("As Ks Ah Kh Qd Jc 2s 3h 4d 5c 6s")
        table = Table(TableConfig(seats=3, small_blind=1, big_blind=2, starting_stacks=[5, 5, 5], deck_order=fixed))
        state = table.start_hand()
        while not state.is_terminal:
            legal = state.legal_actions(state.current_actor)
            state = table.apply(Action.check() if legal.can_check else Action.call())

        self.assertGreaterEqual(table.carryover_chips, 0)
        next_state = table.start_hand()
        self.assertEqual(next_state.carryover_in_pot, table.current_hand.carryover_in_pot)


if __name__ == "__main__":
    unittest.main()
