from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
import random


class Suit(str, Enum):
    CLUBS = "c"
    DIAMONDS = "d"
    HEARTS = "h"
    SPADES = "s"


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


RANK_TO_SYMBOL: dict[Rank, str] = {
    Rank.TWO: "2",
    Rank.THREE: "3",
    Rank.FOUR: "4",
    Rank.FIVE: "5",
    Rank.SIX: "6",
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE: "9",
    Rank.TEN: "T",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
    Rank.ACE: "A",
}
SYMBOL_TO_RANK = {value: key for key, value in RANK_TO_SYMBOL.items()}
SYMBOL_TO_RANK.update({value.lower(): key for key, value in RANK_TO_SYMBOL.items()})


@dataclass(frozen=True, order=True)
class Card:
    rank: Rank
    suit: Suit

    @classmethod
    def from_str(cls, text: str) -> Card:
        if len(text) != 2:
            raise ValueError(f"Card text must have rank and suit, got {text!r}")
        rank_text, suit_text = text[0], text[1].lower()
        try:
            rank = SYMBOL_TO_RANK[rank_text]
            suit = Suit(suit_text)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid card text {text!r}") from exc
        return cls(rank=rank, suit=suit)

    def to_str(self) -> str:
        return f"{RANK_TO_SYMBOL[self.rank]}{self.suit.value}"

    def __str__(self) -> str:
        return self.to_str()


def full_deck() -> list[Card]:
    return [Card(rank, suit) for suit in Suit for rank in Rank]


class Deck:
    def __init__(self, seed: int | None = None, cards: list[Card] | None = None) -> None:
        if cards is None:
            self._cards = full_deck()
            random.Random(seed).shuffle(self._cards)
        else:
            if len(set(cards)) != len(cards):
                raise ValueError("Injected deck order contains duplicate cards")
            self._cards = list(cards)

    @classmethod
    def from_text(cls, text: str) -> Deck:
        return cls(cards=[Card.from_str(part) for part in text.split()])

    def draw_one(self) -> Card:
        if not self._cards:
            raise ValueError("Deck is empty")
        return self._cards.pop(0)

    def draw(self, count: int) -> list[Card]:
        if count < 0:
            raise ValueError("Cannot draw a negative number of cards")
        return [self.draw_one() for _ in range(count)]

    def remaining(self) -> int:
        return len(self._cards)

    def to_list(self) -> list[str]:
        return [card.to_str() for card in self._cards]
