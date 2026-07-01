import unittest

from poker_arena.cards import Card, Deck, Rank, Suit


class CardDeckTests(unittest.TestCase):
    def test_seeded_decks_are_deterministic(self):
        first = Deck(seed=17).draw(5)
        second = Deck(seed=17).draw(5)

        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 5)

    def test_injected_deck_order_is_drawn_first(self):
        ordered = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
        deck = Deck(cards=ordered)

        self.assertEqual(deck.draw_one(), Card(Rank.ACE, Suit.SPADES))
        self.assertEqual(deck.draw_one(), Card(Rank.KING, Suit.HEARTS))

    def test_card_parses_common_text_notation(self):
        self.assertEqual(Card.from_str("As"), Card(Rank.ACE, Suit.SPADES))
        self.assertEqual(Card.from_str("td"), Card(Rank.TEN, Suit.DIAMONDS))


if __name__ == "__main__":
    unittest.main()
