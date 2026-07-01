import unittest

from poker_arena import Action, Card, Table, TableConfig
from poker_arena.events import PokerEvent, event_from_dict


def deck(text: str) -> list[Card]:
    return [Card.from_str(part) for part in text.split()]


class ViewAndEventTests(unittest.TestCase):
    def test_player_view_hides_other_players_hole_cards(self):
        table = Table(TableConfig(seats=3, small_blind=5, big_blind=10, starting_stacks=[100, 100, 100], seed=3))
        state = table.start_hand()

        view = state.player_view(0)

        self.assertEqual(len(view.hole_cards[0]), 2)
        self.assertIsNone(view.hole_cards[1])
        self.assertIsNone(view.hole_cards[2])
        self.assertEqual(state.public_view().hole_cards, {})

    def test_events_round_trip_through_json_safe_dict(self):
        event = PokerEvent("action", {"seat_id": 2, "action": {"type": "call", "total": None}})

        restored = event_from_dict(event.to_dict())

        self.assertEqual(restored, event)

    def test_replay_rebuilds_same_public_state(self):
        fixed = deck("As Ah Ks Kh Qs Qh 2c 3d 4s 5h 6c")
        table = Table(TableConfig(seats=3, small_blind=5, big_blind=10, starting_stacks=[100, 100, 100], deck_order=fixed))
        state = table.start_hand()
        state = table.apply(Action.call())

        replayed = Table.replay(TableConfig(seats=3, small_blind=5, big_blind=10, starting_stacks=[100, 100, 100]), state.events)

        self.assertEqual(replayed.public_view().to_dict(), state.public_view().to_dict())


if __name__ == "__main__":
    unittest.main()
