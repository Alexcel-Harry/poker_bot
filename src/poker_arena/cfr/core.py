from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

from poker_arena.abstraction import AbstractAction, ActionAbstraction
from poker_arena.actions import Action
from poker_arena.events import PokerEvent
from poker_arena.state import HandState
from poker_arena.table import Table, TableConfig


class InformationSetEncoder:
    def __init__(self, trajectory_encoder: object | None = None, embedding_precision: int = 4) -> None:
        self.trajectory_encoder = trajectory_encoder
        self.embedding_precision = embedding_precision

    def encode(self, state: HandState, seat_id: int) -> str:
        player = state.player_by_seat(seat_id)
        legal = state.legal_actions(seat_id) if state.current_actor == seat_id and not state.is_terminal else None
        hole = ",".join(card.to_str() for card in sorted(player.hole_cards))
        board = ",".join(card.to_str() for card in state.board)
        public_events = [event for event in state.events if event.event_type != "snapshot"]
        legal_part = legal.to_dict() if legal is not None else {}
        return (
            f"seat={seat_id}|street={state.street.value}|button={state.button}|"
            f"hole={hole}|board={board}|pot={state.total_pot}|bet={state.current_bet}|"
            f"stacks={tuple(player.stack for player in state.players)}|legal={legal_part}|"
            f"{self._history_part(public_events)}"
        )

    def _history_part(self, public_events: list[PokerEvent]) -> str:
        if self.trajectory_encoder is None:
            public_actions = []
            for event in public_events:
                if event.event_type == "action":
                    action = event.data.get("action")
                    public_actions.append(f"{event.data.get('seat_id')}:{action}")
                elif event.event_type in {"small_blind", "big_blind", "street_dealt"}:
                    public_actions.append(f"{event.event_type}:{event.data}")
            return f"history={'/'.join(public_actions)}"

        vector = self._embed(public_events)
        formatted = ",".join(f"{value:.{self.embedding_precision}f}" for value in vector)
        return f"trajectory=({formatted})"

    def _embed(self, public_events: list[PokerEvent]) -> list[float]:
        if hasattr(self.trajectory_encoder, "encode_events"):
            return list(self.trajectory_encoder.encode_events(public_events))  # type: ignore[attr-defined]
        if hasattr(self.trajectory_encoder, "transform"):
            return list(self.trajectory_encoder.transform(public_events))  # type: ignore[attr-defined]
        raise TypeError("trajectory_encoder must expose encode_events(events) or transform(events)")


class RegretMatcher:
    def strategy_for(self, regrets: Iterable[float], actions: list[AbstractAction]) -> dict[str, float]:
        labels = [action.label for action in actions]
        positive = [max(0.0, value) for value in regrets]
        normalizer = sum(positive)
        if normalizer <= 0:
            probability = 1.0 / len(actions)
            return {label: probability for label in labels}
        return {label: value / normalizer for label, value in zip(labels, positive)}


@dataclass(frozen=True)
class CFRTrainingResult:
    iterations: int
    information_sets: int
    episodes: int


@dataclass
class CFRNode:
    actions: list[AbstractAction]
    regrets: list[float]
    strategy_sum: list[float]
    visits: int = 0

    @classmethod
    def from_actions(cls, actions: list[AbstractAction]) -> CFRNode:
        return cls(actions=actions, regrets=[0.0] * len(actions), strategy_sum=[0.0] * len(actions))


class CFRTrainer:
    """Engine-connected sampled CFR scaffold for No Limit Hold'em.

    This trainer uses finite action abstraction and sampled rollouts so it can
    operate on the full Hold'em engine without enumerating the full no-limit
    tree. It is intended as the production-facing solver spine, not a toy game.
    """

    def __init__(
        self,
        table_config: TableConfig,
        iterations: int = 100,
        max_actions_per_episode: int = 200,
        random_seed: int | None = None,
        action_abstraction: ActionAbstraction | None = None,
        infoset_encoder: InformationSetEncoder | None = None,
    ) -> None:
        self.table_config = table_config
        self.iterations = iterations
        self.max_actions_per_episode = max_actions_per_episode
        self.rng = random.Random(random_seed)
        self.action_abstraction = action_abstraction or ActionAbstraction()
        self.infoset_encoder = infoset_encoder or InformationSetEncoder()
        self.matcher = RegretMatcher()
        self.nodes: dict[str, CFRNode] = {}

    def train(self) -> CFRTrainingResult:
        for iteration in range(self.iterations):
            table = Table(self._iteration_config(iteration))
            table.start_hand()
            self._run_episode(table)
        return CFRTrainingResult(iterations=self.iterations, information_sets=len(self.nodes), episodes=self.iterations)

    def strategy_profile(self) -> dict[str, dict[str, float]]:
        profile: dict[str, dict[str, float]] = {}
        for key, node in self.nodes.items():
            total = sum(node.strategy_sum)
            if total <= 0:
                profile[key] = self.matcher.strategy_for(node.regrets, node.actions)
            else:
                profile[key] = {action.label: value / total for action, value in zip(node.actions, node.strategy_sum)}
        return profile

    def _run_episode(self, table: Table) -> list[float]:
        trajectory: list[tuple[str, int, int, dict[str, float], list[AbstractAction], list[float]]] = []
        for _ in range(self.max_actions_per_episode):
            state = table.current_hand
            if state is None or state.is_terminal:
                break
            actor = state.current_actor
            if actor is None:
                break
            actions = self.action_abstraction.actions_for(state, actor)
            if not actions:
                break
            key = self.infoset_encoder.encode(state, actor)
            node = self._node_for(key, actions)
            strategy = self.matcher.strategy_for(node.regrets, node.actions)
            utilities = self._estimate_action_utilities(table, actor, node.actions)
            chosen_index = self._sample_index(strategy, node.actions)
            node.visits += 1
            for index, action in enumerate(node.actions):
                node.strategy_sum[index] += strategy[action.label]
            trajectory.append((key, actor, chosen_index, strategy, node.actions, utilities))
            try:
                table.apply(node.actions[chosen_index].to_action())
            except ValueError:
                table.apply(self._fallback_action(state, actor))

        terminal_utility = self._terminal_utilities(table)
        self._update_regrets(trajectory)
        return terminal_utility

    def _estimate_action_utilities(self, table: Table, actor: int, actions: list[AbstractAction]) -> list[float]:
        utilities = []
        for action in actions:
            branch = self._clone_table(table)
            try:
                branch.apply(action.to_action())
                utilities.append(self._rollout(branch)[actor])
            except ValueError:
                utilities.append(-1.0)
        return utilities

    def _rollout(self, table: Table) -> list[float]:
        for _ in range(self.max_actions_per_episode):
            state = table.current_hand
            if state is None or state.is_terminal:
                break
            actor = state.current_actor
            if actor is None:
                break
            actions = self.action_abstraction.actions_for(state, actor)
            safe_actions = [action for action in actions if action.label in {"check", "call"}] or actions
            action = self.rng.choice(safe_actions)
            try:
                table.apply(action.to_action())
            except ValueError:
                table.apply(self._fallback_action(state, actor))
        return self._terminal_utilities(table)

    def _terminal_utilities(self, table: Table) -> list[float]:
        state = table.current_hand
        if state is None:
            return [0.0] * self.table_config.seats
        starting = self.table_config.starting_stacks
        return [player.stack - starting[player.seat_id] for player in state.players]

    def _update_regrets(self, trajectory: list[tuple[str, int, int, dict[str, float], list[AbstractAction], list[float]]]) -> None:
        for key, _actor, _chosen_index, strategy, actions, utilities in trajectory:
            node = self.nodes[key]
            expected = sum(strategy[action.label] * utility for action, utility in zip(actions, utilities))
            for index, utility in enumerate(utilities):
                node.regrets[index] += utility - expected

    def _node_for(self, key: str, actions: list[AbstractAction]) -> CFRNode:
        existing = self.nodes.get(key)
        labels = [action.label for action in actions]
        if existing is not None and [action.label for action in existing.actions] == labels:
            return existing
        node = CFRNode.from_actions(actions)
        self.nodes[key] = node
        return node

    def _sample_index(self, strategy: dict[str, float], actions: list[AbstractAction]) -> int:
        draw = self.rng.random()
        cumulative = 0.0
        for index, action in enumerate(actions):
            cumulative += strategy[action.label]
            if draw <= cumulative:
                return index
        return len(actions) - 1

    def _iteration_config(self, iteration: int) -> TableConfig:
        seed = None if self.table_config.seed is None else self.table_config.seed + iteration
        return TableConfig(
            seats=self.table_config.seats,
            small_blind=self.table_config.small_blind,
            big_blind=self.table_config.big_blind,
            starting_stacks=list(self.table_config.starting_stacks),
            seed=seed,
            deck_order=self.table_config.deck_order,
        )

    def _clone_table(self, table: Table) -> Table:
        clone = Table(table.config)
        clone.stacks = list(table.stacks)
        clone.button = table.button
        clone.hand_number = table.hand_number
        clone.carryover_chips = table.carryover_chips
        clone._seed_counter = table._seed_counter  # Preserve deterministic future chance state.
        if table.current_hand is not None:
            events = list(table.current_hand.events)
            clone.current_hand = HandState.from_snapshot(table.config, table.current_hand.to_snapshot(), events)
        return clone

    @staticmethod
    def _fallback_action(state: HandState, actor: int) -> Action:
        legal = state.legal_actions(actor)
        if legal.can_check:
            return Action.check()
        if legal.can_call:
            return Action.call()
        return Action.fold()
