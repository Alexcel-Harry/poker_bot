from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from poker_arena.cfr.toy_games import CHANCE_PLAYER, ExtensiveFormState, GameAction


Strategy = Mapping[str, Mapping[GameAction, float]]


def _strategy_for(state: ExtensiveFormState, profile: Strategy) -> dict[GameAction, float]:
    actions = state.legal_actions()
    if not actions:
        return {}
    configured = profile.get(state.information_set_key(state.current_player), {})
    positive = {action: max(0.0, float(configured.get(action, 0.0))) for action in actions}
    total = sum(positive.values())
    if total <= 0:
        probability = 1.0 / len(actions)
        return {action: probability for action in actions}
    return {action: positive[action] / total for action in actions}


def expected_utility(state: ExtensiveFormState, profile: Strategy, player: int = 0) -> float:
    if state.is_terminal:
        return state.utility(player)
    if state.current_player == CHANCE_PLAYER:
        return sum(
            probability * expected_utility(state.child(action), profile, player)
            for action, probability in state.chance_outcomes()
        )
    strategy = _strategy_for(state, profile)
    return sum(
        probability * expected_utility(state.child(action), profile, player)
        for action, probability in strategy.items()
    )


def best_response_value(state: ExtensiveFormState, profile: Strategy, player: int) -> float:
    """Compute an exact pure best response with consistent infoset decisions."""

    infosets: dict[str, list[tuple[ExtensiveFormState, float, int]]] = {}

    def collect(node: ExtensiveFormState, counterfactual_reach: float, depth: int) -> None:
        if node.is_terminal:
            return
        actor = node.current_player
        if actor == CHANCE_PLAYER:
            for action, probability in node.chance_outcomes():
                collect(node.child(action), counterfactual_reach * probability, depth + 1)
            return
        if actor == player:
            key = node.information_set_key(player)
            infosets.setdefault(key, []).append((node, counterfactual_reach, depth))
            for action in node.legal_actions():
                collect(node.child(action), counterfactual_reach, depth + 1)
            return
        for action, probability in _strategy_for(node, profile).items():
            collect(node.child(action), counterfactual_reach * probability, depth + 1)

    collect(state, 1.0, 0)
    decisions: dict[str, GameAction] = {}

    def value(node: ExtensiveFormState) -> float:
        if node.is_terminal:
            return node.utility(player)
        actor = node.current_player
        if actor == CHANCE_PLAYER:
            return sum(probability * value(node.child(action)) for action, probability in node.chance_outcomes())
        if actor == player:
            key = node.information_set_key(player)
            if key not in decisions:
                raise AssertionError(f"Best-response infoset {key!r} was evaluated out of order")
            return value(node.child(decisions[key]))
        return sum(probability * value(node.child(action)) for action, probability in _strategy_for(node, profile).items())

    ordered = sorted(infosets.items(), key=lambda item: max(entry[2] for entry in item[1]), reverse=True)
    for key, entries in ordered:
        actions = entries[0][0].legal_actions()
        action_values = {
            action: sum(reach * value(node.child(action)) for node, reach, _depth in entries)
            for action in actions
        }
        decisions[key] = max(actions, key=lambda action: action_values[action])
    return value(state)


@dataclass(frozen=True)
class ExploitabilityResult:
    expected_values: tuple[float, float]
    best_response_values: tuple[float, float]
    nash_conv: float
    exploitability: float


def exploitability(state: ExtensiveFormState, profile: Strategy) -> ExploitabilityResult:
    expected = (expected_utility(state, profile, 0), expected_utility(state, profile, 1))
    best = (best_response_value(state, profile, 0), best_response_value(state, profile, 1))
    nash_conv = (best[0] - expected[0]) + (best[1] - expected[1])
    return ExploitabilityResult(expected, best, nash_conv, nash_conv / 2.0)


@dataclass
class TabularCFRNode:
    actions: tuple[GameAction, ...]
    regrets: dict[GameAction, float] = field(default_factory=dict)
    strategy_sum: dict[GameAction, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for action in self.actions:
            self.regrets.setdefault(action, 0.0)
            self.strategy_sum.setdefault(action, 0.0)

    def current_strategy(self) -> dict[GameAction, float]:
        positive = {action: max(0.0, self.regrets[action]) for action in self.actions}
        total = sum(positive.values())
        if total <= 0:
            return {action: 1.0 / len(self.actions) for action in self.actions}
        return {action: positive[action] / total for action in self.actions}

    def average_strategy(self) -> dict[GameAction, float]:
        total = sum(self.strategy_sum.values())
        if total <= 0:
            return {action: 1.0 / len(self.actions) for action in self.actions}
        return {action: self.strategy_sum[action] / total for action in self.actions}


class TabularCFRTrainer:
    """Full-tree CFR reference solver intended for small correctness games."""

    def __init__(self, root: ExtensiveFormState) -> None:
        self.root = root
        self.nodes: dict[str, TabularCFRNode] = {}
        self.iterations = 0

    def train(self, iterations: int) -> Strategy:
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        for _ in range(iterations):
            self._traverse(self.root, reach_zero=1.0, reach_one=1.0, chance_reach=1.0)
            self.iterations += 1
        return self.average_strategy()

    def average_strategy(self) -> dict[str, dict[GameAction, float]]:
        return {key: node.average_strategy() for key, node in self.nodes.items()}

    def _node_for(self, state: ExtensiveFormState) -> TabularCFRNode:
        key = state.information_set_key(state.current_player)
        actions = state.legal_actions()
        existing = self.nodes.get(key)
        if existing is not None:
            if existing.actions != actions:
                raise ValueError(f"Action set changed inside infoset {key!r}")
            return existing
        node = TabularCFRNode(actions)
        self.nodes[key] = node
        return node

    def _traverse(
        self,
        state: ExtensiveFormState,
        reach_zero: float,
        reach_one: float,
        chance_reach: float,
    ) -> float:
        if state.is_terminal:
            return state.utility(0)
        if state.current_player == CHANCE_PLAYER:
            return sum(
                probability
                * self._traverse(
                    state.child(action),
                    reach_zero,
                    reach_one,
                    chance_reach * probability,
                )
                for action, probability in state.chance_outcomes()
            )

        player = state.current_player
        node = self._node_for(state)
        strategy = node.current_strategy()
        action_values: dict[GameAction, float] = {}
        for action, probability in strategy.items():
            action_values[action] = self._traverse(
                state.child(action),
                reach_zero * probability if player == 0 else reach_zero,
                reach_one * probability if player == 1 else reach_one,
                chance_reach,
            )
        node_value = sum(strategy[action] * action_values[action] for action in node.actions)
        opponent_reach = reach_one if player == 0 else reach_zero
        own_reach = reach_zero if player == 0 else reach_one
        sign = 1.0 if player == 0 else -1.0
        for action in node.actions:
            node.regrets[action] += chance_reach * opponent_reach * sign * (action_values[action] - node_value)
            node.strategy_sum[action] += chance_reach * own_reach * strategy[action]
        return node_value

