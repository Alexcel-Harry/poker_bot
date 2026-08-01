from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Hashable, Protocol, Sequence


GameAction = Hashable
CHANCE_PLAYER = -1


class ExtensiveFormState(Protocol):
    """Minimal immutable game-state contract used by solver correctness tests."""

    @property
    def is_terminal(self) -> bool: ...

    @property
    def current_player(self) -> int: ...

    def legal_actions(self) -> tuple[GameAction, ...]: ...

    def chance_outcomes(self) -> tuple[tuple[GameAction, float], ...]: ...

    def child(self, action: GameAction) -> ExtensiveFormState: ...

    def information_set_key(self, player: int) -> str: ...

    def utility(self, player: int) -> float: ...


@dataclass(frozen=True)
class KuhnPokerState:
    """Two-player Kuhn poker with one-chip antes and a one-chip bet."""

    cards: tuple[int, int] | None = None
    history: str = ""

    _terminal_histories = frozenset({"cc", "bc", "bf", "cbc", "cbf"})

    @classmethod
    def initial(cls) -> KuhnPokerState:
        return cls()

    @property
    def is_terminal(self) -> bool:
        return self.history in self._terminal_histories

    @property
    def current_player(self) -> int:
        if self.cards is None:
            return CHANCE_PLAYER
        if self.is_terminal:
            raise ValueError("Terminal states do not have a current player")
        return 0 if self.history in {"", "cb"} else 1

    def legal_actions(self) -> tuple[GameAction, ...]:
        if self.cards is None or self.is_terminal:
            return ()
        if self.history in {"", "c"}:
            return ("check", "bet")
        return ("call", "fold")

    def chance_outcomes(self) -> tuple[tuple[GameAction, float], ...]:
        if self.cards is not None:
            return ()
        deals = tuple((first, second) for first in range(3) for second in range(3) if first != second)
        return tuple((deal, 1.0 / len(deals)) for deal in deals)

    def child(self, action: GameAction) -> KuhnPokerState:
        if self.cards is None:
            if not isinstance(action, tuple) or len(action) != 2:
                raise ValueError("Kuhn chance action must be an ordered two-card deal")
            return replace(self, cards=(int(action[0]), int(action[1])))
        if action not in self.legal_actions():
            raise ValueError(f"Illegal Kuhn action {action!r} after {self.history!r}")
        symbol = {"check": "c", "bet": "b", "call": "c", "fold": "f"}[str(action)]
        return replace(self, history=self.history + symbol)

    def information_set_key(self, player: int) -> str:
        if self.cards is None or player not in (0, 1):
            raise ValueError("Kuhn information sets require a dealt state and player 0 or 1")
        return f"kuhn:p{player}:card={self.cards[player]}:history={self.history}"

    def utility(self, player: int) -> float:
        if not self.is_terminal or self.cards is None:
            raise ValueError("Kuhn utility is only defined at terminal states")
        if self.history == "bf":
            utility_zero = 1.0
        elif self.history == "cbf":
            utility_zero = -1.0
        else:
            stake = 1.0 if self.history == "cc" else 2.0
            utility_zero = stake if self.cards[0] > self.cards[1] else -stake
        return utility_zero if player == 0 else -utility_zero


@dataclass(frozen=True)
class LeducPokerState:
    """Standard two-player limit Leduc with two raises allowed per round."""

    private_cards: tuple[int, int] | None = None
    remaining_cards: tuple[int, ...] = ()
    public_card: int | None = None
    street: int = 0
    actor: int = CHANCE_PLAYER
    contributions: tuple[int, int] = (1, 1)
    street_contributions: tuple[int, int] = (0, 0)
    raises: int = 0
    consecutive_checks: int = 0
    histories: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    folded_player: int | None = None
    terminal: bool = False
    pending_public_card: bool = False
    max_raises: int = 2

    @classmethod
    def initial(cls, max_raises: int = 2) -> LeducPokerState:
        if max_raises < 0:
            raise ValueError("max_raises must be non-negative")
        return cls(max_raises=max_raises)

    @property
    def is_terminal(self) -> bool:
        return self.terminal

    @property
    def current_player(self) -> int:
        if self.terminal:
            raise ValueError("Terminal states do not have a current player")
        return self.actor

    @staticmethod
    def rank(card: int) -> int:
        if not 0 <= card < 6:
            raise ValueError(f"Invalid Leduc card {card}")
        return card // 2

    def legal_actions(self) -> tuple[GameAction, ...]:
        if self.terminal or self.actor == CHANCE_PLAYER:
            return ()
        outstanding = max(self.street_contributions) - self.street_contributions[self.actor]
        if outstanding > 0:
            actions: list[GameAction] = ["fold", "call"]
        else:
            actions = ["check"]
        if self.raises < self.max_raises:
            actions.append("raise")
        return tuple(actions)

    def chance_outcomes(self) -> tuple[tuple[GameAction, float], ...]:
        if self.terminal or self.actor != CHANCE_PLAYER:
            return ()
        if self.private_cards is None:
            deals = tuple((first, second) for first in range(6) for second in range(6) if first != second)
            return tuple((deal, 1.0 / len(deals)) for deal in deals)
        if self.pending_public_card:
            probability = 1.0 / len(self.remaining_cards)
            return tuple((card, probability) for card in self.remaining_cards)
        return ()

    def child(self, action: GameAction) -> LeducPokerState:
        if self.actor == CHANCE_PLAYER:
            return self._chance_child(action)
        if action not in self.legal_actions():
            raise ValueError(f"Illegal Leduc action {action!r}")
        actor = self.actor
        opponent = 1 - actor
        history = list(self.histories[self.street])
        history.append(str(action))
        histories = list(self.histories)
        histories[self.street] = tuple(history)

        if action == "fold":
            return replace(self, histories=tuple(histories), folded_player=actor, terminal=True)

        contributions = list(self.contributions)
        street_contributions = list(self.street_contributions)
        outstanding = max(street_contributions) - street_contributions[actor]
        if action == "call":
            street_contributions[actor] += outstanding
            contributions[actor] += outstanding
            return self._finish_betting_round(tuple(contributions), tuple(street_contributions), tuple(histories))

        if action == "check":
            if self.consecutive_checks == 1:
                return self._finish_betting_round(self.contributions, self.street_contributions, tuple(histories))
            return replace(self, actor=opponent, consecutive_checks=1, histories=tuple(histories))

        bet_size = 2 if self.street == 0 else 4
        target = max(street_contributions) + bet_size
        paid = target - street_contributions[actor]
        street_contributions[actor] = target
        contributions[actor] += paid
        return replace(
            self,
            actor=opponent,
            contributions=tuple(contributions),
            street_contributions=tuple(street_contributions),
            raises=self.raises + 1,
            consecutive_checks=0,
            histories=tuple(histories),
        )

    def _chance_child(self, action: GameAction) -> LeducPokerState:
        if self.private_cards is None:
            if not isinstance(action, tuple) or len(action) != 2:
                raise ValueError("Initial Leduc chance action must deal two private cards")
            first, second = int(action[0]), int(action[1])
            if first == second or first not in range(6) or second not in range(6):
                raise ValueError("Invalid Leduc private-card deal")
            remaining = tuple(card for card in range(6) if card not in (first, second))
            return replace(self, private_cards=(first, second), remaining_cards=remaining, actor=0)
        if not self.pending_public_card or not isinstance(action, int) or action not in self.remaining_cards:
            raise ValueError("Invalid Leduc public-card chance action")
        remaining = tuple(card for card in self.remaining_cards if card != action)
        return replace(
            self,
            public_card=action,
            remaining_cards=remaining,
            street=1,
            actor=0,
            street_contributions=(0, 0),
            raises=0,
            consecutive_checks=0,
            pending_public_card=False,
        )

    def _finish_betting_round(
        self,
        contributions: tuple[int, int],
        street_contributions: tuple[int, int],
        histories: tuple[tuple[str, ...], tuple[str, ...]],
    ) -> LeducPokerState:
        if self.street == 0:
            return replace(
                self,
                actor=CHANCE_PLAYER,
                contributions=contributions,
                street_contributions=street_contributions,
                histories=histories,
                pending_public_card=True,
            )
        return replace(
            self,
            contributions=contributions,
            street_contributions=street_contributions,
            histories=histories,
            terminal=True,
        )

    def information_set_key(self, player: int) -> str:
        if self.private_cards is None or player not in (0, 1):
            raise ValueError("Leduc information sets require private cards and player 0 or 1")
        private_rank = self.rank(self.private_cards[player])
        public_rank = "-" if self.public_card is None else str(self.rank(self.public_card))
        first = ",".join(self.histories[0])
        second = ",".join(self.histories[1])
        return f"leduc:p{player}:private={private_rank}:public={public_rank}:rounds={first}/{second}"

    def utility(self, player: int) -> float:
        if not self.terminal or self.private_cards is None:
            raise ValueError("Leduc utility is only defined at terminal states")
        pot = sum(self.contributions)
        if self.folded_player is not None:
            winners: Sequence[int] = (1 - self.folded_player,)
        else:
            assert self.public_card is not None
            public_rank = self.rank(self.public_card)
            private_ranks = [self.rank(card) for card in self.private_cards]
            strengths = [(1 if rank == public_rank else 0, rank) for rank in private_ranks]
            best = max(strengths)
            winners = tuple(index for index, strength in enumerate(strengths) if strength == best)
        payout = pot / len(winners) if player in winners else 0.0
        return payout - self.contributions[player]

