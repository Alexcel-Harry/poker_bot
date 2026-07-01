from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import random
from typing import Sequence

from poker_arena.actions import Action, ActionType, LegalActions
from poker_arena.cfr.core import CFRTrainingResult, InformationSetEncoder
from poker_arena.cfr.torch_model import StateFeatureEncoder, TorchTrainingSample
from poker_arena.embedding import TrajectoryEncoder
from poker_arena.state import HandState
from poker_arena.table import Table, TableConfig


def _action_label(action: Action) -> str:
    if action.action_type == ActionType.RAISE_TO:
        return f"raise_to:{action.total}"
    return action.action_type.value


def _action_key(action: Action) -> tuple[str, int | None]:
    return action.action_type.value, action.total


@dataclass
class EmbeddingCoverageIndex:
    vectors: list[list[float]] = field(default_factory=list)

    def novelty(self, vector: Sequence[float]) -> float:
        if not self.vectors:
            return 1.0
        nearest = min(TrajectoryEncoder.distance(vector, existing) for existing in self.vectors)
        return nearest / (1.0 + nearest)

    def record(self, vector: Sequence[float]) -> None:
        self.vectors.append([float(value) for value in vector])


class ActionEmbedding:
    """Continuous features for concrete actions, including arbitrary integer raises."""

    dimension_without_trajectory = 12

    def encode(
        self,
        action: Action,
        legal_context: LegalActions,
        trajectory_embedding: Sequence[float] | None = None,
    ) -> list[float]:
        vector = [0.0] * self.dimension_without_trajectory
        action_index = {
            ActionType.FOLD: 0,
            ActionType.CHECK: 1,
            ActionType.CALL: 2,
            ActionType.RAISE_TO: 3,
        }[action.action_type]
        vector[action_index] = 1.0

        min_raise = legal_context.min_raise_to or legal_context.current_bet
        max_raise = legal_context.max_raise_to or max(min_raise, legal_context.current_bet)
        span = max(1, max_raise - min_raise)
        total = action.total or legal_context.current_bet
        added = max(0, total - legal_context.current_bet)
        actor_stack_before = legal_context.actor_commitment + max(0, max_raise - legal_context.actor_commitment)
        actor_stack_before = max(1, actor_stack_before)

        vector[4] = total / max(1.0, float(max_raise))
        vector[5] = (total - min_raise) / float(span)
        vector[6] = added / float(span)
        vector[7] = legal_context.call_amount / actor_stack_before
        vector[8] = legal_context.current_bet / actor_stack_before
        vector[9] = legal_context.actor_commitment / actor_stack_before
        vector[10] = min_raise / actor_stack_before
        vector[11] = max_raise / actor_stack_before
        if trajectory_embedding is not None:
            vector.extend(float(value) for value in trajectory_embedding)
        return vector

    @staticmethod
    def distance(first: Sequence[float], second: Sequence[float]) -> float:
        return TrajectoryEncoder.distance(first, second)


class IntegerActionSampler:
    """Samples concrete legal integer actions without collapsing raises into fixed labels."""

    def __init__(
        self,
        random_seed: int | None = None,
        novelty_weight: float = 0.0,
        action_embedding: ActionEmbedding | None = None,
    ) -> None:
        self.rng = random.Random(random_seed)
        self.novelty_weight = novelty_weight
        self.action_embedding = action_embedding or ActionEmbedding()

    def sample(
        self,
        state: HandState,
        seat_id: int,
        budget: int,
        required_amounts: Sequence[int] = (),
        coverage_index: EmbeddingCoverageIndex | None = None,
        trajectory_embedding: Sequence[float] | None = None,
    ) -> list[Action]:
        legal = state.legal_actions(seat_id)
        actions: list[Action] = []
        if legal.can_fold:
            actions.append(Action.fold())
        if legal.can_check:
            actions.append(Action.check())
        if legal.can_call:
            actions.append(Action.call())

        if legal.can_raise and legal.min_raise_to is not None and legal.max_raise_to is not None:
            raise_totals = self._sample_raise_totals(
                legal,
                budget=max(0, budget - len(actions)),
                required_amounts=required_amounts,
                coverage_index=coverage_index,
                trajectory_embedding=trajectory_embedding,
            )
            actions.extend(Action.raise_to(total) for total in raise_totals)
        return self._dedupe(actions)

    def _sample_raise_totals(
        self,
        legal: LegalActions,
        budget: int,
        required_amounts: Sequence[int],
        coverage_index: EmbeddingCoverageIndex | None,
        trajectory_embedding: Sequence[float] | None,
    ) -> list[int]:
        assert legal.min_raise_to is not None
        assert legal.max_raise_to is not None
        minimum = legal.min_raise_to
        maximum = legal.max_raise_to
        required = sorted({int(amount) for amount in required_amounts if minimum <= amount <= maximum})
        capacity = max(budget, len(required))
        totals: list[int] = list(required)

        for amount in (minimum, maximum):
            if len(set(totals)) >= capacity:
                break
            totals.append(amount)

        remaining = max(0, capacity - len(set(totals)))
        if remaining:
            pool_size = max(remaining * 4, remaining)
            pool = [self.rng.randint(minimum, maximum) for _ in range(pool_size)]
            if coverage_index is not None and self.novelty_weight > 0 and trajectory_embedding is not None:
                pool = sorted(
                    pool,
                    key=lambda total: coverage_index.novelty(
                        self.action_embedding.encode(Action.raise_to(total), legal, trajectory_embedding)
                    ),
                    reverse=True,
                )
            totals.extend(pool[:remaining])

        return sorted(set(totals))

    @staticmethod
    def _dedupe(actions: Sequence[Action]) -> list[Action]:
        seen: set[tuple[str, int | None]] = set()
        deduped: list[Action] = []
        for action in actions:
            key = _action_key(action)
            if key not in seen:
                seen.add(key)
                deduped.append(action)
        return deduped


@dataclass(frozen=True)
class BranchResult:
    action: Action
    utilities: list[float]
    terminal: bool
    steps: int
    trajectory_embedding: list[float]
    action_embedding: list[float]


class PrefixBranchExplorer:
    def __init__(
        self,
        trajectory_encoder: TrajectoryEncoder | None = None,
        action_embedding: ActionEmbedding | None = None,
        max_actions_per_rollout: int = 200,
        random_seed: int | None = None,
        max_workers: int = 1,
    ) -> None:
        self.trajectory_encoder = trajectory_encoder or TrajectoryEncoder()
        self.action_embedding = action_embedding or ActionEmbedding()
        self.max_actions_per_rollout = max_actions_per_rollout
        self.rng = random.Random(random_seed)
        self.max_workers = max(1, max_workers)

    def expand(
        self,
        table: Table,
        actor: int,
        actions: Sequence[Action],
        depth: int,
        width: int,
    ) -> list[BranchResult]:
        selected = list(actions)[:width]
        seeds = [self.rng.randrange(2**31) for _ in selected]
        if self.max_workers == 1 or len(selected) <= 1:
            return [self._evaluate_branch(table, actor, action, depth, seed) for action, seed in zip(selected, seeds)]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._evaluate_branch, table, actor, action, depth, seed)
                for action, seed in zip(selected, seeds)
            ]
            return [future.result() for future in futures]

    def _evaluate_branch(self, table: Table, actor: int, action: Action, depth: int, seed: int) -> BranchResult:
        branch = self._clone_table(table)
        before = branch.current_hand
        if before is None:
            return BranchResult(action, [0.0] * table.config.seats, True, 0, [], [])
        legal = before.legal_actions(actor)
        trajectory_before = self.trajectory_encoder.encode_events(before.events)
        action_vector = self.action_embedding.encode(action, legal, trajectory_before)
        steps = 0
        try:
            branch.apply(action)
            steps += 1
        except ValueError:
            utilities = [-1.0 if seat == actor else 0.0 for seat in range(table.config.seats)]
            return BranchResult(action, utilities, False, steps, trajectory_before, action_vector)

        rollout_rng = random.Random(seed)
        limit = self.max_actions_per_rollout if depth <= 0 else min(depth, self.max_actions_per_rollout)
        for _ in range(max(0, limit - 1)):
            state = branch.current_hand
            if state is None or state.is_terminal or state.current_actor is None:
                break
            rollout_action = self._rollout_action(state, state.current_actor, rollout_rng)
            try:
                branch.apply(rollout_action)
            except ValueError:
                branch.apply(self._fallback_action(state, state.current_actor))
            steps += 1

        current = branch.current_hand
        trajectory_after = self.trajectory_encoder.encode_events(current.events) if current is not None else trajectory_before
        return BranchResult(
            action=action,
            utilities=self._utilities(branch),
            terminal=bool(current is None or current.is_terminal),
            steps=steps,
            trajectory_embedding=trajectory_after,
            action_embedding=action_vector,
        )

    def _rollout_action(self, state: HandState, actor: int, rng: random.Random) -> Action:
        legal = state.legal_actions(actor)
        if legal.can_check:
            return Action.check()
        if legal.can_call:
            return Action.call()
        if legal.can_fold:
            return Action.fold()
        if legal.can_raise and legal.min_raise_to is not None and legal.max_raise_to is not None:
            return Action.raise_to(rng.randint(legal.min_raise_to, legal.max_raise_to))
        return Action.fold()

    def _utilities(self, table: Table) -> list[float]:
        state = table.current_hand
        if state is None:
            return [0.0] * table.config.seats
        starting = table.config.starting_stacks
        return [player.stack - starting[player.seat_id] for player in state.players]

    def _clone_table(self, table: Table) -> Table:
        clone = Table(table.config)
        clone.stacks = list(table.stacks)
        clone.button = table.button
        clone.hand_number = table.hand_number
        clone.carryover_chips = table.carryover_chips
        clone._seed_counter = table._seed_counter
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


@dataclass(frozen=True)
class PrefixBranchTrainingConfig:
    branch_width: int = 32
    branch_depth: int = 8
    integer_action_budget: int = 32
    novelty_weight: float = 1.0
    neighbor_weight: float = 0.0
    required_integer_actions: tuple[int, ...] = ()
    max_actions_per_episode: int = 200
    random_seed: int | None = None
    max_workers: int = 1


@dataclass(frozen=True)
class ActionTrainingSample:
    infoset_key: str
    action: Action
    action_embedding: list[float]
    trajectory_embedding: list[float]
    target_utility: float
    weight: float


@dataclass
class IntegerCFRNode:
    labels: list[str]
    regrets: list[float]
    strategy_sum: list[float]
    visits: int = 0

    @classmethod
    def from_labels(cls, labels: Sequence[str]) -> "IntegerCFRNode":
        return cls(labels=list(labels), regrets=[0.0] * len(labels), strategy_sum=[0.0] * len(labels))


class PrefixBranchCFRTrainer:
    def __init__(
        self,
        table_config: TableConfig,
        config: PrefixBranchTrainingConfig | None = None,
        infoset_encoder: InformationSetEncoder | None = None,
        trajectory_encoder: TrajectoryEncoder | None = None,
        action_sampler: IntegerActionSampler | None = None,
        explorer: PrefixBranchExplorer | None = None,
    ) -> None:
        self.table_config = table_config
        self.config = config or PrefixBranchTrainingConfig()
        self.rng = random.Random(self.config.random_seed)
        self.infoset_encoder = infoset_encoder or InformationSetEncoder()
        self.trajectory_encoder = trajectory_encoder or TrajectoryEncoder()
        self.state_feature_encoder = StateFeatureEncoder()
        self.action_embedding = ActionEmbedding()
        self.coverage_index = EmbeddingCoverageIndex()
        self.action_sampler = action_sampler or IntegerActionSampler(
            random_seed=self.config.random_seed,
            novelty_weight=self.config.novelty_weight,
            action_embedding=self.action_embedding,
        )
        self.explorer = explorer or PrefixBranchExplorer(
            trajectory_encoder=self.trajectory_encoder,
            action_embedding=self.action_embedding,
            max_actions_per_rollout=self.config.max_actions_per_episode,
            random_seed=self.config.random_seed,
            max_workers=self.config.max_workers,
        )
        self.nodes: dict[str, IntegerCFRNode] = {}
        self.training_samples: list[ActionTrainingSample] = []
        self.torch_training_samples: list[TorchTrainingSample] = []

    def train(self, iterations: int) -> CFRTrainingResult:
        for iteration in range(iterations):
            table = Table(self._iteration_config(iteration))
            table.start_hand()
            self._run_episode(table)
        return CFRTrainingResult(iterations=iterations, information_sets=len(self.nodes), episodes=iterations)

    def _run_episode(self, table: Table) -> None:
        for _ in range(self.config.max_actions_per_episode):
            state = table.current_hand
            if state is None or state.is_terminal or state.current_actor is None:
                break
            actor = state.current_actor
            legal = state.legal_actions(actor)
            state_features = self.state_feature_encoder.encode_state(state, actor)
            trajectory = self.trajectory_encoder.encode_events(state.events)
            actions = self.action_sampler.sample(
                state,
                actor,
                budget=self.config.integer_action_budget,
                required_amounts=self.config.required_integer_actions,
                coverage_index=self.coverage_index,
                trajectory_embedding=trajectory,
            )
            if not actions:
                break
            branched_actions = actions[: self.config.branch_width]
            key = self.infoset_encoder.encode(state, actor)
            labels = [_action_label(action) for action in branched_actions]
            node = self._node_for(key, labels)
            branches = self.explorer.expand(table, actor, branched_actions, depth=self.config.branch_depth, width=self.config.branch_width)
            if not branches:
                break
            utilities = [branch.utilities[actor] for branch in branches]
            strategy = self._strategy_for(node)
            expected = sum(strategy[label] * utility for label, utility in zip(node.labels, utilities))
            for index, utility in enumerate(utilities):
                node.regrets[index] += utility - expected
                node.strategy_sum[index] += strategy[node.labels[index]]
                branch = branches[index]
                self.training_samples.append(
                    ActionTrainingSample(
                        infoset_key=key,
                        action=branch.action,
                        action_embedding=branch.action_embedding,
                        trajectory_embedding=branch.trajectory_embedding,
                        target_utility=utility,
                        weight=1.0,
                    )
                )
                self.torch_training_samples.append(
                    TorchTrainingSample(
                        state_features=state_features,
                        trajectory_features=trajectory,
                        action_features=self.action_embedding.encode(branch.action, legal),
                        action=branch.action.to_dict(),
                        target_utility=utility,
                        weight=1.0,
                    )
                )
                self.coverage_index.record(branch.action_embedding)
            node.visits += 1

            chosen = self._sample_branch(branches, strategy, node.labels)
            try:
                table.apply(chosen.action)
            except ValueError:
                table.apply(PrefixBranchExplorer._fallback_action(state, actor))

    def _node_for(self, key: str, labels: Sequence[str]) -> IntegerCFRNode:
        existing = self.nodes.get(key)
        if existing is not None and existing.labels == list(labels):
            return existing
        node = IntegerCFRNode.from_labels(labels)
        self.nodes[key] = node
        return node

    def _strategy_for(self, node: IntegerCFRNode) -> dict[str, float]:
        positives = [max(0.0, regret) for regret in node.regrets]
        normalizer = sum(positives)
        if normalizer <= 0:
            probability = 1.0 / len(node.labels)
            return {label: probability for label in node.labels}
        return {label: value / normalizer for label, value in zip(node.labels, positives)}

    def _sample_branch(
        self,
        branches: Sequence[BranchResult],
        strategy: dict[str, float],
        labels: Sequence[str],
    ) -> BranchResult:
        draw = self.rng.random()
        cumulative = 0.0
        for branch, label in zip(branches, labels):
            cumulative += strategy[label]
            if draw <= cumulative:
                return branch
        return branches[-1]

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
