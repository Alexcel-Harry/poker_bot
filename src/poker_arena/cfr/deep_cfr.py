from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any, Callable, Generic, Mapping, Protocol, Sequence, TypeVar

from poker_arena.cfr.toy_games import CHANCE_PLAYER, ExtensiveFormState, GameAction


try:  # pragma: no cover - availability depends on the training environment.
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


T = TypeVar("T")
DEEP_CFR_SNAPSHOT_VERSION = 1
DEEP_CFR_POLICY_VERSION = 1


def _require_torch() -> Any:
    if torch is None or nn is None:
        raise ModuleNotFoundError("Install the training extra with `pip install -e .[train]` to use Deep CFR")
    return torch


class ReservoirBuffer(Generic[T]):
    """Uniform reservoir sample over every item ever observed."""

    def __init__(self, capacity: int, random_seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("Reservoir capacity must be positive")
        self.capacity = capacity
        self.samples: list[T] = []
        self.samples_seen = 0
        self.rng = random.Random(random_seed)

    def add(self, sample: T) -> None:
        self.samples_seen += 1
        if len(self.samples) < self.capacity:
            self.samples.append(sample)
            return
        replacement = self.rng.randrange(self.samples_seen)
        if replacement < self.capacity:
            self.samples[replacement] = sample

    def sample(self, count: int, rng: random.Random) -> list[T]:
        if count <= 0:
            raise ValueError("Sample count must be positive")
        if not self.samples:
            raise ValueError("Cannot sample an empty reservoir")
        if count >= len(self.samples):
            return list(self.samples)
        return rng.sample(self.samples, count)

    def __len__(self) -> int:
        return len(self.samples)

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "samples": list(self.samples),
            "samples_seen": self.samples_seen,
            "rng_state": self.rng.getstate(),
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> ReservoirBuffer[Any]:
        buffer = cls(int(payload["capacity"]))
        buffer.samples = list(payload["samples"])
        buffer.samples_seen = int(payload["samples_seen"])
        buffer.rng.setstate(payload["rng_state"])
        return buffer


@dataclass(frozen=True)
class AdvantageSample:
    features: tuple[float, ...]
    legal_mask: tuple[bool, ...]
    advantages: tuple[float, ...]
    iteration: int


@dataclass(frozen=True)
class StrategySample:
    features: tuple[float, ...]
    legal_mask: tuple[bool, ...]
    probabilities: tuple[float, ...]
    iteration: int


class DeepCFRFeatureEncoder(Protocol):
    actions: tuple[GameAction, ...]
    action_index: Mapping[GameAction, int]

    @property
    def input_dim(self) -> int: ...

    @property
    def action_dim(self) -> int: ...

    def encode(self, state: ExtensiveFormState) -> tuple[float, ...]: ...

    def legal_mask(self, state: ExtensiveFormState) -> tuple[bool, ...]: ...

    def action_vector(self, values: Mapping[GameAction, float]) -> tuple[float, ...]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def validate_state_dict(self, payload: Mapping[str, Any]) -> None: ...


class GameTreeFeatureEncoder:
    """Exact one-hot infoset encoder for finite validation games."""

    def __init__(self, root: ExtensiveFormState) -> None:
        states: dict[str, ExtensiveFormState] = {}
        actions: set[GameAction] = set()

        def visit(state: ExtensiveFormState) -> None:
            if state.is_terminal:
                return
            if state.current_player == CHANCE_PLAYER:
                for action, _probability in state.chance_outcomes():
                    visit(state.child(action))
                return
            key = state.information_set_key(state.current_player)
            legal = state.legal_actions()
            existing = states.get(key)
            if existing is not None and existing.legal_actions() != legal:
                raise ValueError(f"Action set changed inside infoset {key!r}")
            states.setdefault(key, state)
            actions.update(legal)
            for action in legal:
                visit(state.child(action))

        visit(root)
        self.infoset_keys = tuple(sorted(states))
        self.actions = tuple(sorted(actions, key=lambda action: repr(action)))
        self.infoset_index = {key: index for index, key in enumerate(self.infoset_keys)}
        self.action_index = {action: index for index, action in enumerate(self.actions)}
        self.states_by_key = {key: states[key] for key in self.infoset_keys}

    @property
    def input_dim(self) -> int:
        return len(self.infoset_keys)

    @property
    def action_dim(self) -> int:
        return len(self.actions)

    def encode(self, state: ExtensiveFormState) -> tuple[float, ...]:
        index = self.infoset_index[state.information_set_key(state.current_player)]
        return tuple(1.0 if position == index else 0.0 for position in range(self.input_dim))

    def legal_mask(self, state: ExtensiveFormState) -> tuple[bool, ...]:
        legal = set(state.legal_actions())
        return tuple(action in legal for action in self.actions)

    def action_vector(self, values: Mapping[GameAction, float]) -> tuple[float, ...]:
        return tuple(float(values.get(action, 0.0)) for action in self.actions)

    def state_dict(self) -> dict[str, Any]:
        return {"infoset_keys": self.infoset_keys, "actions": self.actions}

    def validate_state_dict(self, payload: Mapping[str, Any]) -> None:
        if tuple(payload["infoset_keys"]) != self.infoset_keys or tuple(payload["actions"]) != self.actions:
            raise ValueError("Deep CFR snapshot does not match this game tree")


@dataclass(frozen=True)
class DeepCFRConfig:
    iterations: int = 100
    traversals_per_player: int = 100
    advantage_capacity: int = 100_000
    strategy_capacity: int = 100_000
    hidden: tuple[int, ...] = (128, 128)
    advantage_train_steps: int = 200
    strategy_train_steps: int = 1_000
    batch_size: int = 256
    learning_rate: float = 1e-3
    linear_weighting: bool = True
    random_seed: int = 17
    device: str = "cpu"

    def __post_init__(self) -> None:
        positive = (
            "iterations",
            "traversals_per_player",
            "advantage_capacity",
            "strategy_capacity",
            "advantage_train_steps",
            "strategy_train_steps",
            "batch_size",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.hidden or any(width <= 0 for width in self.hidden):
            raise ValueError("hidden must contain positive layer widths")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


if nn is not None:

    class DeepCFRNetwork(nn.Module):  # type: ignore[misc]
        def __init__(self, input_dim: int, action_dim: int, hidden: Sequence[int]) -> None:
            super().__init__()
            layers: list[Any] = []
            previous = input_dim
            for width in hidden:
                layers.extend((nn.Linear(previous, width), nn.ReLU()))
                previous = width
            layers.append(nn.Linear(previous, action_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, inputs: Any) -> Any:
            return self.net(inputs)

else:

    class DeepCFRNetwork:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_torch()


class DeepCFRTrainer:
    """Reference external-sampling Deep CFR for finite or multi-player games."""

    def __init__(
        self,
        root: ExtensiveFormState,
        config: DeepCFRConfig | None = None,
        encoder: DeepCFRFeatureEncoder | None = None,
    ) -> None:
        torch_module = _require_torch()
        self.root = root
        self.num_players = int(getattr(root, "num_players", 2))
        if self.num_players < 2:
            raise ValueError("Deep CFR requires at least two players")
        self.config = config or DeepCFRConfig()
        self.encoder = encoder or GameTreeFeatureEncoder(root)
        self.device = torch_module.device(self.config.device)
        self.rng = random.Random(self.config.random_seed)
        torch_module.manual_seed(self.config.random_seed)
        if self.device.type == "cuda":
            torch_module.cuda.manual_seed_all(self.config.random_seed)
        self.advantage_networks = [self._new_network(zero_output=True) for _ in range(self.num_players)]
        self.strategy_networks = [self._new_network(zero_output=True) for _ in range(self.num_players)]
        self.advantage_memories = [
            ReservoirBuffer[AdvantageSample](self.config.advantage_capacity, self.config.random_seed + player * 101)
            for player in range(self.num_players)
        ]
        self.strategy_memories = [
            ReservoirBuffer[StrategySample](self.config.strategy_capacity, self.config.random_seed + 10_000 + player * 101)
            for player in range(self.num_players)
        ]
        self.completed_iterations = 0
        self.traversals = 0
        self.traverser_nodes = 0
        self.sampled_opponent_nodes = 0
        self.sampled_chance_nodes = 0
        self.maximum_depth = 0
        self.losses: dict[str, list[float]] = {
            f"{kind}_{player}": []
            for kind in ("advantage", "strategy")
            for player in range(self.num_players)
        }

    def _new_network(self, zero_output: bool = False) -> DeepCFRNetwork:
        network = DeepCFRNetwork(self.encoder.input_dim, self.encoder.action_dim, self.config.hidden).to(self.device)
        if zero_output:
            last = next(module for module in reversed(list(network.modules())) if isinstance(module, nn.Linear))
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        return network

    def train(
        self,
        iterations: int | None = None,
        progress_callback: Callable[[int, int, int], None] | None = None,
    ) -> dict[str, Any]:
        count = self.config.iterations if iterations is None else iterations
        if count <= 0:
            raise ValueError("iterations must be positive")
        target = self.completed_iterations + count
        while self.completed_iterations < target:
            iteration = self.completed_iterations + 1
            for traverser in range(self.num_players):
                for _ in range(self.config.traversals_per_player):
                    self._traverse(self.root, traverser, iteration, depth=0)
                    self.traversals += 1
                loss = self._train_advantage_network(traverser)
                self.losses[f"advantage_{traverser}"].append(loss)
            self.completed_iterations = iteration
            if progress_callback is not None:
                progress_callback(iteration, target, self.traversals)
        for player in range(self.num_players):
            loss = self._train_strategy_network(player)
            self.losses[f"strategy_{player}"].append(loss)
        return self.stats()

    def _traverse(self, state: ExtensiveFormState, traverser: int, iteration: int, depth: int) -> float:
        self.maximum_depth = max(self.maximum_depth, depth)
        if state.is_terminal:
            return state.utility(traverser)
        actor = state.current_player
        if actor == CHANCE_PLAYER:
            self.sampled_chance_nodes += 1
            chance_sampler = getattr(state, "sample_chance_action", None)
            if callable(chance_sampler):
                action = chance_sampler(self.rng)
            else:
                outcomes = state.chance_outcomes()
                action = self._sample_distribution(dict(outcomes))
            return self._traverse(state.child(action), traverser, iteration, depth + 1)

        strategy = self.current_strategy(state)
        if actor == traverser:
            self.traverser_nodes += 1
            action_values = {
                action: self._traverse(state.child(action), traverser, iteration, depth + 1)
                for action in state.legal_actions()
            }
            expected = sum(strategy[action] * action_values[action] for action in state.legal_actions())
            advantages = {action: action_values[action] - expected for action in state.legal_actions()}
            self.advantage_memories[traverser].add(
                AdvantageSample(
                    features=self.encoder.encode(state),
                    legal_mask=self.encoder.legal_mask(state),
                    advantages=self.encoder.action_vector(advantages),
                    iteration=iteration,
                )
            )
            return expected

        self.sampled_opponent_nodes += 1
        self.strategy_memories[actor].add(
            StrategySample(
                features=self.encoder.encode(state),
                legal_mask=self.encoder.legal_mask(state),
                probabilities=self.encoder.action_vector(strategy),
                iteration=iteration,
            )
        )
        action = self._sample_distribution(strategy)
        return self._traverse(state.child(action), traverser, iteration, depth + 1)

    def _sample_distribution(self, probabilities: Mapping[GameAction, float]) -> GameAction:
        draw = self.rng.random()
        cumulative = 0.0
        last: GameAction | None = None
        for action, probability in probabilities.items():
            last = action
            cumulative += probability
            if draw <= cumulative:
                return action
        if last is None:
            raise ValueError("Cannot sample an empty distribution")
        return last

    def current_strategy(self, state: ExtensiveFormState) -> dict[GameAction, float]:
        actor = state.current_player
        if not 0 <= actor < self.num_players:
            raise ValueError("Strategies are only defined at player nodes")
        outputs = self._network_output(self.advantage_networks[actor], self.encoder.encode(state))
        legal = state.legal_actions()
        positive = {action: max(0.0, outputs[self.encoder.action_index[action]]) for action in legal}
        total = sum(positive.values())
        if total <= 0:
            return {action: 1.0 / len(legal) for action in legal}
        return {action: positive[action] / total for action in legal}

    def average_strategy(self, state: ExtensiveFormState) -> dict[GameAction, float]:
        actor = state.current_player
        if not 0 <= actor < self.num_players:
            raise ValueError("Strategies are only defined at player nodes")
        outputs = self._network_output(self.strategy_networks[actor], self.encoder.encode(state))
        legal = state.legal_actions()
        logits = torch.tensor(
            [outputs[self.encoder.action_index[action]] for action in legal],
            dtype=torch.float64,
        )
        probabilities = torch.softmax(logits, dim=0).tolist()
        return {action: float(probability) for action, probability in zip(legal, probabilities)}

    def _network_output(self, network: DeepCFRNetwork, features: Sequence[float]) -> list[float]:
        torch_module = _require_torch()
        inputs = torch_module.tensor([features], dtype=torch_module.float32, device=self.device)
        network.eval()
        with torch_module.no_grad():
            return network(inputs)[0].detach().cpu().tolist()

    def _sample_tensors(self, samples: Sequence[AdvantageSample | StrategySample]) -> tuple[Any, Any, Any, Any]:
        torch_module = _require_torch()
        features = torch_module.tensor([sample.features for sample in samples], dtype=torch_module.float32, device=self.device)
        masks = torch_module.tensor([sample.legal_mask for sample in samples], dtype=torch_module.bool, device=self.device)
        targets = torch_module.tensor(
            [sample.advantages if isinstance(sample, AdvantageSample) else sample.probabilities for sample in samples],
            dtype=torch_module.float32,
            device=self.device,
        )
        weights = torch_module.tensor(
            [sample.iteration if self.config.linear_weighting else 1.0 for sample in samples],
            dtype=torch_module.float32,
            device=self.device,
        )
        weights /= weights.mean().clamp_min(1e-8)
        return features, masks, targets, weights

    def _train_advantage_network(self, player: int) -> float:
        memory = self.advantage_memories[player]
        if not memory.samples:
            return 0.0
        network = self._new_network()
        optimizer = torch.optim.Adam(network.parameters(), lr=self.config.learning_rate)
        total = 0.0
        network.train()
        for _ in range(self.config.advantage_train_steps):
            batch = memory.sample(min(self.config.batch_size, len(memory)), self.rng)
            features, masks, targets, weights = self._sample_tensors(batch)
            optimizer.zero_grad(set_to_none=True)
            predictions = network(features)
            per_sample = (((predictions - targets) ** 2) * masks.float()).sum(dim=1) / masks.sum(dim=1).clamp_min(1)
            loss = (per_sample * weights).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu().item())
        self.advantage_networks[player] = network
        return total / self.config.advantage_train_steps

    def _train_strategy_network(self, player: int) -> float:
        memory = self.strategy_memories[player]
        if not memory.samples:
            return 0.0
        network = self._new_network()
        optimizer = torch.optim.Adam(network.parameters(), lr=self.config.learning_rate)
        total = 0.0
        network.train()
        for _ in range(self.config.strategy_train_steps):
            batch = memory.sample(min(self.config.batch_size, len(memory)), self.rng)
            features, masks, targets, weights = self._sample_tensors(batch)
            optimizer.zero_grad(set_to_none=True)
            logits = network(features).masked_fill(~masks, -1e9)
            log_probabilities = torch.log_softmax(logits, dim=1)
            per_sample = -(targets * log_probabilities).sum(dim=1)
            loss = (per_sample * weights).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu().item())
        self.strategy_networks[player] = network
        return total / self.config.strategy_train_steps

    def average_strategy_profile(self) -> dict[str, dict[GameAction, float]]:
        states = getattr(self.encoder, "states_by_key", None)
        if states is None:
            raise ValueError("This encoder does not enumerate a finite strategy profile")
        return {
            key: self.average_strategy(state)
            for key, state in states.items()
        }

    def sample_action(self, state: ExtensiveFormState, average: bool = True) -> GameAction:
        strategy = self.average_strategy(state) if average else self.current_strategy(state)
        return self._sample_distribution(strategy)

    def stats(self) -> dict[str, Any]:
        return {
            "algorithm": "external_sampling_deep_cfr",
            "num_players": self.num_players,
            "iterations": self.completed_iterations,
            "traversals": self.traversals,
            "traverser_nodes": self.traverser_nodes,
            "sampled_opponent_nodes": self.sampled_opponent_nodes,
            "sampled_chance_nodes": self.sampled_chance_nodes,
            "maximum_depth": self.maximum_depth,
            "advantage_samples": [len(memory) for memory in self.advantage_memories],
            "advantage_samples_seen": [memory.samples_seen for memory in self.advantage_memories],
            "strategy_samples": [len(memory) for memory in self.strategy_memories],
            "strategy_samples_seen": [memory.samples_seen for memory in self.strategy_memories],
            "loss": {key: list(values) for key, values in self.losses.items()},
        }

    def snapshot_payload(self) -> dict[str, Any]:
        return {
            "snapshot_version": DEEP_CFR_SNAPSHOT_VERSION,
            "num_players": self.num_players,
            "config": asdict(self.config),
            "encoder": self.encoder.state_dict(),
            "advantage_networks": [network.state_dict() for network in self.advantage_networks],
            "strategy_networks": [network.state_dict() for network in self.strategy_networks],
            "advantage_memories": [memory.state_dict() for memory in self.advantage_memories],
            "strategy_memories": [memory.state_dict() for memory in self.strategy_memories],
            "rng_state": self.rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else [],
            "completed_iterations": self.completed_iterations,
            "stats": self.stats(),
        }

    def save_snapshot(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.snapshot_payload(), target)

    @classmethod
    def load_snapshot(
        cls,
        root: ExtensiveFormState,
        path: str | Path,
        device: str | None = None,
        encoder: DeepCFRFeatureEncoder | None = None,
    ) -> DeepCFRTrainer:
        payload = _require_torch().load(Path(path), map_location=device or "cpu", weights_only=False)
        if int(payload.get("snapshot_version", -1)) != DEEP_CFR_SNAPSHOT_VERSION:
            raise ValueError(f"Unsupported Deep CFR snapshot version: {payload.get('snapshot_version')}")
        config_data = dict(payload["config"])
        if device is not None:
            config_data["device"] = device
        config_data["hidden"] = tuple(config_data["hidden"])
        trainer = cls(root, DeepCFRConfig(**config_data), encoder=encoder)
        if int(payload.get("num_players", 2)) != trainer.num_players:
            raise ValueError("Deep CFR snapshot player count does not match the game")
        for key in ("advantage_networks", "strategy_networks", "advantage_memories", "strategy_memories"):
            if len(payload[key]) != trainer.num_players:
                raise ValueError(f"Deep CFR snapshot has the wrong number of {key}")
        trainer.encoder.validate_state_dict(payload["encoder"])
        for network, state in zip(trainer.advantage_networks, payload["advantage_networks"]):
            network.load_state_dict(state)
        for network, state in zip(trainer.strategy_networks, payload["strategy_networks"]):
            network.load_state_dict(state)
        trainer.advantage_memories = [ReservoirBuffer.from_state_dict(state) for state in payload["advantage_memories"]]
        trainer.strategy_memories = [ReservoirBuffer.from_state_dict(state) for state in payload["strategy_memories"]]
        trainer.rng.setstate(payload["rng_state"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if trainer.device.type == "cuda" and payload.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_states"]])
        trainer.completed_iterations = int(payload["completed_iterations"])
        stats = payload["stats"]
        trainer.traversals = int(stats["traversals"])
        trainer.traverser_nodes = int(stats["traverser_nodes"])
        trainer.sampled_opponent_nodes = int(stats["sampled_opponent_nodes"])
        trainer.sampled_chance_nodes = int(stats["sampled_chance_nodes"])
        trainer.maximum_depth = int(stats["maximum_depth"])
        trainer.losses = {key: list(values) for key, values in stats["loss"].items()}
        return trainer

    def inference_payload(self) -> dict[str, Any]:
        return {
            "policy_version": DEEP_CFR_POLICY_VERSION,
            "algorithm": "deep_cfr_average_strategy",
            "num_players": self.num_players,
            "encoder": self.encoder.state_dict(),
            "hidden": self.config.hidden,
            "strategy_networks": [network.state_dict() for network in self.strategy_networks],
            "training_stats": self.stats(),
        }

    def save_inference_policy(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.inference_payload(), target)
