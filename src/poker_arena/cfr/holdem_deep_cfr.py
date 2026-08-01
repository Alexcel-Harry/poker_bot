from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Mapping, Sequence

from poker_arena.abstraction import ActionAbstraction, AbstractActionKind
from poker_arena.cfr.core import InformationSetEncoder
from poker_arena.cfr.toy_games import CHANCE_PLAYER, GameAction
from poker_arena.cfr.torch_model import StateFeatureEncoder
from poker_arena.embedding import EventContextBuilder
from poker_arena.embedding import TrajectoryEncoder
from poker_arena.state import HandState
from poker_arena.table import Table, TableConfig


@dataclass
class HoldemCFRState:
    """Immutable-by-convention multi-player Hold'em state for Deep CFR traversal."""

    config: TableConfig
    table: Table | None = None
    abstraction: ActionAbstraction | None = None

    def __post_init__(self) -> None:
        if self.config.seats < 3:
            raise ValueError("Hold'em Deep CFR training requires at least three players")
        if self.abstraction is None:
            self.abstraction = ActionAbstraction.compact()

    @property
    def num_players(self) -> int:
        return self.config.seats

    @classmethod
    def initial(
        cls,
        config: TableConfig,
        abstraction: ActionAbstraction | None = None,
    ) -> HoldemCFRState:
        return cls(config=config, abstraction=abstraction)

    @property
    def is_terminal(self) -> bool:
        return bool(self.table is not None and self.table.current_hand is not None and self.table.current_hand.is_terminal)

    @property
    def current_player(self) -> int:
        if self.table is None:
            return CHANCE_PLAYER
        state = self._hand()
        if state.is_terminal or state.current_actor is None:
            raise ValueError("Terminal Hold'em states do not have a current player")
        return state.current_actor

    def legal_actions(self) -> tuple[GameAction, ...]:
        if self.table is None or self.is_terminal:
            return ()
        state = self._hand()
        assert self.abstraction is not None
        return tuple(action.label for action in self.abstraction.actions_for(state, state.current_actor))

    def chance_outcomes(self) -> tuple[tuple[GameAction, float], ...]:
        # A full deck permutation is far too large to enumerate. DeepCFRTrainer
        # detects sample_chance_action() and samples one determinization instead.
        return ()

    def sample_chance_action(self, rng: random.Random) -> int:
        if self.table is not None:
            raise ValueError("Only the Hold'em root is a chance node")
        return rng.randrange(2**31)

    def child(self, action: GameAction) -> HoldemCFRState:
        if self.table is None:
            if not isinstance(action, int):
                raise ValueError("Hold'em root chance action must be an integer deck seed")
            config = TableConfig(
                seats=self.config.seats,
                small_blind=self.config.small_blind,
                big_blind=self.config.big_blind,
                starting_stacks=list(self.config.starting_stacks),
                seed=action,
            )
            table = Table(config)
            table.start_hand()
            return HoldemCFRState(config=config, table=table, abstraction=self.abstraction)

        state = self._hand()
        assert self.abstraction is not None
        concrete = {abstract.label: abstract.to_action() for abstract in self.abstraction.actions_for(state, state.current_actor)}
        if action not in concrete:
            raise ValueError(f"Illegal abstract Hold'em action {action!r}")
        table = self._clone_table(self.table)
        table.apply(concrete[action])
        return HoldemCFRState(config=table.config, table=table, abstraction=self.abstraction)

    def information_set_key(self, player: int) -> str:
        if self.table is None:
            raise ValueError("The Hold'em root has no player information set")
        return InformationSetEncoder().encode(self._hand(), player)

    def utility(self, player: int) -> float:
        state = self._hand()
        if not state.is_terminal:
            raise ValueError("Hold'em utility is only defined at terminal states")
        return float(state.player_by_seat(player).stack - self.config.starting_stacks[player])

    def hand_state(self) -> HandState:
        return self._hand()

    def _hand(self) -> HandState:
        if self.table is None or self.table.current_hand is None:
            raise ValueError("Hold'em cards have not been dealt")
        return self.table.current_hand

    @staticmethod
    def _clone_table(table: Table) -> Table:
        clone = Table(table.config)
        clone.stacks = list(table.stacks)
        clone.button = table.button
        clone.hand_number = table.hand_number
        clone.carryover_chips = table.carryover_chips
        clone._seed_counter = table._seed_counter
        if table.current_hand is not None:
            clone.current_hand = HandState.from_snapshot(
                table.config,
                table.current_hand.to_snapshot(),
                list(table.current_hand.events),
            )
        return clone


class OrderedPublicHistoryEncoder:
    """Fixed-width ordered public action history; no opponent cards are exposed."""

    features_per_action = 10

    def __init__(self, max_actions: int = 24) -> None:
        if max_actions <= 0:
            raise ValueError("max_actions must be positive")
        self.max_actions = max_actions

    @property
    def dimension(self) -> int:
        return self.max_actions * self.features_per_action

    def encode(self, state: HandState) -> tuple[float, ...]:
        return self.encode_events(state.events)

    def encode_events(self, events: Sequence[Any]) -> tuple[float, ...]:
        contexts = [context for context in EventContextBuilder().build(events) if context.event_type == "action"]
        contexts = contexts[-self.max_actions :]
        result = [0.0] * self.dimension
        offset_slots = self.max_actions - len(contexts)
        type_index = {"fold": 0, "check": 1, "call": 2, "raise_to": 3}
        street_index = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
        for local_index, context in enumerate(contexts):
            raw = context.event.to_dict() if hasattr(context.event, "to_dict") else context.event
            data = raw.get("data", {})
            action = data.get("action", {}) if isinstance(data, Mapping) else {}
            base = (offset_slots + local_index) * self.features_per_action
            action_type = str(action.get("type")) if isinstance(action, Mapping) else ""
            if action_type in type_index:
                result[base + type_index[action_type]] = 1.0
            total = action.get("total") if isinstance(action, Mapping) else None
            if isinstance(total, int):
                result[base + 4] = total / max(1.0, float(context.pot_before + context.actor_stack_before))
            street = street_index.get(context.street_before)
            if street is not None:
                result[base + 5 + street] = 1.0
            seat = data.get("seat_id") if isinstance(data, Mapping) else None
            if isinstance(seat, int):
                result[base + 9] = seat / 8.0
        return tuple(result)


class HoldemDeepCFRFeatureEncoder:
    """Private-safe numeric infoset and fixed compact-action encoder."""

    actions: tuple[GameAction, ...] = (
        AbstractActionKind.FOLD.value,
        AbstractActionKind.CHECK.value,
        AbstractActionKind.CALL.value,
        AbstractActionKind.MIN_RAISE.value,
        AbstractActionKind.THIRD_POT.value,
        AbstractActionKind.THREE_QUARTER_POT.value,
        AbstractActionKind.OVERBET.value,
        AbstractActionKind.ALL_IN.value,
    )

    def __init__(self, max_history_actions: int = 24) -> None:
        self.state_encoder = StateFeatureEncoder()
        self.history_encoder = OrderedPublicHistoryEncoder(max_history_actions)
        self.action_index = {action: index for index, action in enumerate(self.actions)}

    @property
    def input_dim(self) -> int:
        return self.state_encoder.dimension + self.history_encoder.dimension

    @property
    def action_dim(self) -> int:
        return len(self.actions)

    def encode(self, state: HoldemCFRState) -> tuple[float, ...]:
        hand = state.hand_state()
        actor = hand.current_actor
        if actor is None:
            raise ValueError("Cannot encode a terminal Hold'em state")
        return tuple(self.state_encoder.encode_state(hand, actor)) + self.history_encoder.encode(hand)

    def encode_view(self, view: Any, legal_actions: Any) -> tuple[float, ...]:
        return tuple(self.state_encoder.encode_view(view, legal_actions)) + self.history_encoder.encode_events(view.events)

    def legal_mask(self, state: HoldemCFRState) -> tuple[bool, ...]:
        legal = set(state.legal_actions())
        return tuple(action in legal for action in self.actions)

    def action_vector(self, values: Mapping[GameAction, float]) -> tuple[float, ...]:
        return tuple(float(values.get(action, 0.0)) for action in self.actions)

    def state_dict(self) -> dict[str, Any]:
        return {
            "encoder": "holdem_private_state_ordered_history_v1",
            "state_dimension": self.state_encoder.dimension,
            "max_history_actions": self.history_encoder.max_actions,
            "actions": self.actions,
        }

    def validate_state_dict(self, payload: Mapping[str, Any]) -> None:
        if dict(payload) != self.state_dict():
            raise ValueError("Deep CFR snapshot uses an incompatible Hold'em encoder or action abstraction")


class TensorHoldemDeepCFRFeatureEncoder:
    """Deployment encoder matching the level-synchronous tensor trainer."""

    actions = HoldemDeepCFRFeatureEncoder.actions

    def __init__(self) -> None:
        self.state_encoder = StateFeatureEncoder()
        self.trajectory_encoder = TrajectoryEncoder()
        self.action_index = {action: index for index, action in enumerate(self.actions)}

    @property
    def input_dim(self) -> int:
        return self.state_encoder.dimension + self.trajectory_encoder.dimension

    @property
    def action_dim(self) -> int:
        return len(self.actions)

    def encode_view(self, view: Any, legal_actions: Any) -> tuple[float, ...]:
        return tuple(self.state_encoder.encode_view(view, legal_actions)) + tuple(
            self.trajectory_encoder.encode_events(view.events)
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "encoder": "holdem_tensor_state_trajectory_v1",
            "state_dimension": self.state_encoder.dimension,
            "trajectory_dimension": self.trajectory_encoder.dimension,
            "actions": self.actions,
        }

    def validate_state_dict(self, payload: Mapping[str, Any]) -> None:
        if dict(payload) != self.state_dict():
            raise ValueError("Deep CFR policy uses an incompatible tensor Hold'em encoder")
