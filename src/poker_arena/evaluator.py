from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations

from poker_arena.cards import Card


class HandCategory(IntEnum):
    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8


@dataclass(frozen=True)
class HandValue:
    category: HandCategory
    tiebreakers: tuple[int, ...]
    cards: tuple[Card, ...]

    def _key(self) -> tuple[int, tuple[int, ...]]:
        return int(self.category), self.tiebreakers

    def __lt__(self, other: HandValue) -> bool:
        return self._key() < other._key()

    def __le__(self, other: HandValue) -> bool:
        return self._key() <= other._key()

    def __gt__(self, other: HandValue) -> bool:
        return self._key() > other._key()

    def __ge__(self, other: HandValue) -> bool:
        return self._key() >= other._key()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HandValue):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())


def _straight_high(ranks: list[int]) -> int | None:
    unique = sorted(set(ranks), reverse=True)
    if 14 in unique:
        unique.append(1)
    for index in range(len(unique) - 4):
        window = unique[index : index + 5]
        if window[0] - window[4] == 4 and len(window) == 5:
            return 5 if window == [5, 4, 3, 2, 1] else window[0]
    return None


def evaluate_five(cards: tuple[Card, ...]) -> HandValue:
    if len(cards) != 5:
        raise ValueError("evaluate_five requires exactly five cards")

    ranks = sorted((int(card.rank) for card in cards), reverse=True)
    counts = Counter(ranks)
    grouped = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    flush = len({card.suit for card in cards}) == 1
    straight = _straight_high(ranks)

    if flush and straight:
        return HandValue(HandCategory.STRAIGHT_FLUSH, (straight,), cards)

    if grouped[0][1] == 4:
        quad = grouped[0][0]
        kicker = max(rank for rank in ranks if rank != quad)
        return HandValue(HandCategory.FOUR_OF_A_KIND, (quad, kicker), cards)

    if grouped[0][1] == 3 and grouped[1][1] == 2:
        return HandValue(HandCategory.FULL_HOUSE, (grouped[0][0], grouped[1][0]), cards)

    if flush:
        return HandValue(HandCategory.FLUSH, tuple(ranks), cards)

    if straight:
        return HandValue(HandCategory.STRAIGHT, (straight,), cards)

    if grouped[0][1] == 3:
        trip = grouped[0][0]
        kickers = tuple(rank for rank in ranks if rank != trip)
        return HandValue(HandCategory.THREE_OF_A_KIND, (trip, *kickers), cards)

    pairs = [rank for rank, count in grouped if count == 2]
    if len(pairs) == 2:
        high_pair, low_pair = sorted(pairs, reverse=True)
        kicker = max(rank for rank in ranks if rank not in pairs)
        return HandValue(HandCategory.TWO_PAIR, (high_pair, low_pair, kicker), cards)

    if len(pairs) == 1:
        pair = pairs[0]
        kickers = tuple(rank for rank in ranks if rank != pair)
        return HandValue(HandCategory.ONE_PAIR, (pair, *kickers), cards)

    return HandValue(HandCategory.HIGH_CARD, tuple(ranks), cards)


def evaluate_best(cards: list[Card] | tuple[Card, ...]) -> HandValue:
    if len(cards) < 5:
        raise ValueError("At least five cards are required")
    return max(evaluate_five(tuple(combo)) for combo in combinations(cards, 5))
