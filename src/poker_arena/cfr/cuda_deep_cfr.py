from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from poker_arena.cfr.deep_cfr import DEEP_CFR_POLICY_VERSION, DeepCFRNetwork
from poker_arena.cfr.gpu_prefix_branch import TensorPokerState
from poker_arena.cfr.holdem_deep_cfr import TensorHoldemDeepCFRFeatureEncoder
from poker_arena.table import TableConfig


CUDA_DEEP_CFR_SNAPSHOT_VERSION = 1
RESUME_RUNTIME_CONFIG_FIELDS = frozenset({"iterations", "parallel_traversals", "max_frontier_rows"})


def _atomic_torch_save(payload: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


@dataclass(frozen=True)
class CudaDeepCFRConfig:
    iterations: int = 100
    traversals_per_player: int = 4096
    parallel_traversals: int = 192
    max_traversal_depth: int = 128
    max_frontier_rows: int = 131_072
    advantage_capacity: int = 300_000
    strategy_capacity: int = 300_000
    hidden: tuple[int, ...] = (512, 512, 256)
    advantage_train_steps: int = 1_000
    strategy_train_steps: int = 4_000
    batch_size: int = 8192
    learning_rate: float = 1e-3
    pot_fractions: tuple[float, float, float] = (1.0 / 3.0, 0.75, 1.5)
    linear_weighting: bool = True
    random_seed: int = 17
    minimum_players: int = 3

    def __post_init__(self) -> None:
        positive = (
            "iterations",
            "traversals_per_player",
            "parallel_traversals",
            "max_traversal_depth",
            "max_frontier_rows",
            "advantage_capacity",
            "strategy_capacity",
            "advantage_train_steps",
            "strategy_train_steps",
            "batch_size",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if len(self.pot_fractions) != 3 or any(value <= 0 for value in self.pot_fractions):
            raise ValueError("pot_fractions must contain exactly three positive values")
        if not self.hidden or any(width <= 0 for width in self.hidden):
            raise ValueError("hidden must contain positive layer widths")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 3 <= self.minimum_players <= 9:
            raise ValueError("minimum_players must be between 3 and 9")
        if self.max_frontier_rows < 8:
            raise ValueError("max_frontier_rows must allow at least one full abstract-action expansion")


class CudaReservoirBuffer:
    """Fixed-shape CUDA reservoir retaining an unbiased sample of all iterations."""

    def __init__(self, capacity: int, feature_dim: int, action_dim: int, device: torch.device) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.features = torch.empty((capacity, feature_dim), dtype=torch.float32, device=device)
        self.legal_masks = torch.empty((capacity, action_dim), dtype=torch.bool, device=device)
        self.targets = torch.empty((capacity, action_dim), dtype=torch.float32, device=device)
        self.iterations = torch.empty((capacity,), dtype=torch.float32, device=device)
        self.size = 0
        self.samples_seen = 0

    def add(
        self,
        features: torch.Tensor,
        legal_masks: torch.Tensor,
        targets: torch.Tensor,
        iteration: int,
        generator: torch.Generator,
    ) -> None:
        features = features.detach()
        legal_masks = legal_masks.detach()
        targets = targets.detach()
        count = int(features.shape[0])
        if legal_masks.shape[0] != count or targets.shape[0] != count:
            raise ValueError("CUDA reservoir tensors must contain the same number of samples")
        if count == 0:
            return
        start_seen = self.samples_seen
        fill = min(count, self.capacity - self.size)
        if fill:
            destination = slice(self.size, self.size + fill)
            self.features[destination].copy_(features[:fill])
            self.legal_masks[destination].copy_(legal_masks[:fill])
            self.targets[destination].copy_(targets[:fill])
            self.iterations[destination].fill_(float(iteration))
            self.size += fill

        remaining = count - fill
        if remaining:
            source_rows = torch.arange(fill, count, device=features.device)
            seen_counts = torch.arange(
                start_seen + fill + 1,
                start_seen + count + 1,
                dtype=torch.float64,
                device=features.device,
            )
            draws = torch.floor(
                torch.rand((remaining,), dtype=torch.float64, device=features.device, generator=generator)
                * seen_counts
            ).to(torch.int64)
            selected = draws < self.capacity
            selected_sources = source_rows[selected].tolist()
            selected_destinations = draws[selected].tolist()
            # Resolve collisions in stream order: the later sample is the one
            # that a sequential reservoir update would leave in the slot.
            replacements: dict[int, int] = {}
            for source, destination in zip(selected_sources, selected_destinations):
                replacements[int(destination)] = int(source)
            if replacements:
                destinations = torch.tensor(list(replacements), dtype=torch.int64, device=features.device)
                sources = torch.tensor(list(replacements.values()), dtype=torch.int64, device=features.device)
                # Advanced indexing returns a temporary tensor in PyTorch, so
                # ``self.features[destinations].copy_(...)`` does not update the
                # reservoir.  Use indexed in-place operations on the original
                # storage for every batched replacement.
                self.features.index_copy_(0, destinations, features.index_select(0, sources))
                self.legal_masks.index_copy_(0, destinations, legal_masks.index_select(0, sources))
                self.targets.index_copy_(0, destinations, targets.index_select(0, sources))
                self.iterations.index_fill_(0, destinations, float(iteration))
        self.samples_seen += count

    def iteration_range(self) -> list[int] | None:
        """Return the oldest and newest iteration currently retained."""
        if self.size == 0:
            return None
        retained = self.iterations[: self.size]
        return [int(retained.min().item()), int(retained.max().item())]

    def sample(self, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, ...]:
        if self.size == 0:
            raise ValueError("Cannot sample an empty CUDA reservoir")
        indices = torch.randint(0, self.size, (min(batch_size, self.size),), device=self.features.device, generator=generator)
        return self.features[indices], self.legal_masks[indices], self.targets[indices], self.iterations[indices]

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "features": self.features[: self.size].detach().cpu(),
            "legal_masks": self.legal_masks[: self.size].detach().cpu(),
            "targets": self.targets[: self.size].detach().cpu(),
            "iterations": self.iterations[: self.size].detach().cpu(),
            "size": self.size,
            "samples_seen": self.samples_seen,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if int(payload["capacity"]) != self.capacity:
            raise ValueError("CUDA reservoir capacity does not match snapshot")
        size = int(payload["size"])
        for destination, source in (
            (self.features, payload["features"]),
            (self.legal_masks, payload["legal_masks"]),
            (self.targets, payload["targets"]),
            (self.iterations, payload["iterations"]),
        ):
            destination[:size].copy_(source.to(destination.device))
        self.size = size
        self.samples_seen = int(payload["samples_seen"])


@dataclass
class _FrontierLayer:
    parent_count: int
    child_parents: torch.Tensor
    child_slots: torch.Tensor
    traverser_mask: torch.Tensor
    features: torch.Tensor
    legal_masks: torch.Tensor
    strategy: torch.Tensor


class CudaDeepCFRTrainer:
    """Level-synchronous Deep CFR with independently randomized table sizes."""

    def __init__(
        self,
        table_config: TableConfig,
        config: CudaDeepCFRConfig,
        device: torch.device,
    ) -> None:
        if table_config.seats < 3:
            raise ValueError("CUDA Deep CFR training requires at least three players")
        if config.minimum_players > table_config.seats:
            raise ValueError("minimum_players cannot exceed the maximum table size")
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("CudaDeepCFRTrainer requires an available CUDA device")
        self.table_config = table_config
        self.num_players = table_config.seats
        self.config = config
        self.device = device
        self.encoder = TensorHoldemDeepCFRFeatureEncoder()
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        self.advantage_networks = [self._new_network(zero_output=True) for _ in range(self.num_players)]
        self.strategy_networks = [self._new_network(zero_output=True) for _ in range(self.num_players)]
        self.advantage_memories = [
            CudaReservoirBuffer(config.advantage_capacity, self.encoder.input_dim, self.encoder.action_dim, device)
            for _ in range(self.num_players)
        ]
        self.strategy_memories = [
            CudaReservoirBuffer(config.strategy_capacity, self.encoder.input_dim, self.encoder.action_dim, device)
            for _ in range(self.num_players)
        ]
        self.completed_iterations = 0
        self.traversals = 0
        self.frontier_layers = 0
        self.maximum_frontier_rows = 0
        self.maximum_projected_frontier_rows = 0
        self.frontier_chunk_splits = 0
        self.depth_limit_rollouts = 0
        self.player_count_traversals = {
            player_count: 0
            for player_count in range(self.config.minimum_players, self.num_players + 1)
        }
        self.losses: dict[str, list[float]] = {
            f"{kind}_{player}": []
            for kind in ("advantage", "strategy")
            for player in range(self.num_players)
        }

    def _new_network(self, zero_output: bool = False) -> DeepCFRNetwork:
        network = DeepCFRNetwork(self.encoder.input_dim, self.encoder.action_dim, self.config.hidden).to(self.device)
        if zero_output:
            last = next(module for module in reversed(list(network.modules())) if isinstance(module, torch.nn.Linear))
            torch.nn.init.zeros_(last.weight)
            torch.nn.init.zeros_(last.bias)
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
                remaining = self.config.traversals_per_player
                while remaining > 0:
                    batch = min(remaining, self.config.parallel_traversals)
                    player_counts = self._sample_player_counts(traverser, batch)
                    unique_counts, group_sizes = torch.unique(player_counts, return_counts=True)
                    for player_count, group_size in zip(unique_counts.tolist(), group_sizes.tolist()):
                        session_count = int(group_size)
                        self._collect_batch(
                            traverser,
                            session_count,
                            iteration,
                            self._table_config_for(int(player_count)),
                        )
                        self.player_count_traversals[int(player_count)] += session_count
                    self.traversals += batch
                    remaining -= batch
                self.losses[f"advantage_{traverser}"].append(self._train_network(traverser, strategy=False))
            self.completed_iterations = iteration
            if progress_callback is not None:
                progress_callback(iteration, target, self.traversals)
        for player in range(self.num_players):
            self.losses[f"strategy_{player}"].append(self._train_network(player, strategy=True))
        return self.stats()

    def _sample_player_counts(self, traverser: int, batch_size: int) -> torch.Tensor:
        """Sample one valid table size per traversal session on the CUDA RNG."""
        minimum = max(self.config.minimum_players, traverser + 1)
        if minimum == self.num_players:
            return torch.full((batch_size,), self.num_players, dtype=torch.int64, device=self.device)
        return torch.randint(
            minimum,
            self.num_players + 1,
            (batch_size,),
            dtype=torch.int64,
            device=self.device,
            generator=self.generator,
        )

    def _table_config_for(self, player_count: int) -> TableConfig:
        return TableConfig(
            player_count,
            self.table_config.small_blind,
            self.table_config.big_blind,
            list(self.table_config.starting_stacks[:player_count]),
        )

    def _collect_batch(
        self,
        traverser: int,
        batch_size: int,
        iteration: int,
        table_config: TableConfig,
    ) -> None:
        state = TensorPokerState.new_batch(
            table_config,
            batch_size,
            self.device,
            self.generator,
        )
        self._traverse_state(
            state,
            traverser,
            iteration,
            float(table_config.starting_stacks[traverser]),
            self.config.max_traversal_depth,
        )

    def _traverse_state(
        self,
        state: TensorPokerState,
        traverser: int,
        iteration: int,
        utility_scale: float,
        remaining_depth: int,
    ) -> torch.Tensor:
        """Traverse a frontier, recursively chunking oversized projected children."""
        layers: list[_FrontierLayer] = []
        values: torch.Tensor | None = None
        while remaining_depth > 0:
            if bool(state.terminal.all().item()):
                break
            parent_count = state.batch_size
            features = torch.cat((state.state_features(), state.trajectory_features()), dim=1)
            action_types, totals, valid, _legal = state.cfr_candidate_actions(self.config.pot_fractions)
            logits = torch.zeros((parent_count, self.encoder.action_dim), dtype=torch.float32, device=self.device)
            live = ~state.terminal
            for player in range(state.seats):
                rows = live & (state.current_actor == player)
                if bool(rows.any().item()):
                    self.advantage_networks[player].eval()
                    with torch.no_grad():
                        logits[rows] = self.advantage_networks[player](features[rows])
            positive = torch.relu(logits) * valid.float()
            normalizer = positive.sum(dim=1, keepdim=True)
            fallback = valid.float() / valid.sum(dim=1, keepdim=True).clamp_min(1)
            strategy = torch.where(normalizer > 0, positive / normalizer.clamp_min(1e-12), fallback)

            traverser_mask = live & (state.current_actor == traverser)
            opponent_mask = live & ~traverser_mask
            terminal_parents = torch.nonzero(state.terminal, as_tuple=False).squeeze(1)
            traverser_edges = torch.nonzero(traverser_mask[:, None] & valid, as_tuple=False)
            opponent_parents = torch.nonzero(opponent_mask, as_tuple=False).squeeze(1)
            projected_rows = int(
                terminal_parents.numel() + traverser_edges.shape[0] + opponent_parents.numel()
            )
            self.maximum_projected_frontier_rows = max(self.maximum_projected_frontier_rows, projected_rows)

            if projected_rows > self.config.max_frontier_rows:
                if parent_count == 1:
                    raise RuntimeError(
                        f"A single CUDA Deep CFR parent projected {projected_rows} rows, exceeding the "
                        f"{self.config.max_frontier_rows}-row chunk budget"
                    )
                required_chunks = (projected_rows + self.config.max_frontier_rows - 1) // self.config.max_frontier_rows
                chunk_size = max(1, (parent_count + required_chunks - 1) // required_chunks)
                self.frontier_chunk_splits += 1
                del (
                    features,
                    action_types,
                    totals,
                    valid,
                    _legal,
                    logits,
                    live,
                    rows,
                    positive,
                    normalizer,
                    fallback,
                    strategy,
                    traverser_mask,
                    opponent_mask,
                    terminal_parents,
                    traverser_edges,
                    opponent_parents,
                )
                chunk_values: list[torch.Tensor] = []
                for start in range(0, parent_count, chunk_size):
                    stop = min(start + chunk_size, parent_count)
                    indices = torch.arange(start, stop, dtype=torch.int64, device=self.device)
                    chunk = state.index_select(indices)
                    chunk_values.append(
                        self._traverse_state(
                            chunk,
                            traverser,
                            iteration,
                            utility_scale,
                            remaining_depth,
                        )
                    )
                values = torch.cat(chunk_values)
                break

            if bool(opponent_mask.any().item()):
                for opponent in range(state.seats):
                    opponent_rows = opponent_mask & (state.current_actor == opponent)
                    if bool(opponent_rows.any().item()):
                        self.strategy_memories[opponent].add(
                            features[opponent_rows],
                            valid[opponent_rows],
                            strategy[opponent_rows],
                            iteration,
                            self.generator,
                        )

            if opponent_parents.numel():
                sampled_slots = torch.multinomial(
                    strategy[opponent_parents],
                    1,
                    generator=self.generator,
                ).squeeze(1)
            else:
                sampled_slots = torch.empty((0,), dtype=torch.int64, device=self.device)

            child_parents = torch.cat((terminal_parents, traverser_edges[:, 0], opponent_parents))
            child_slots = torch.cat(
                (
                    torch.full_like(terminal_parents, -1),
                    traverser_edges[:, 1],
                    sampled_slots,
                )
            )
            child = state.index_select(child_parents)
            actionable = child_slots >= 0
            if bool(actionable.any().item()):
                parent_rows = child_parents[actionable]
                slots = child_slots[actionable]
                selected_types = torch.full(
                    (child.batch_size,),
                    -1,
                    dtype=torch.int64,
                    device=self.device,
                )
                selected_totals = torch.zeros(
                    (child.batch_size,),
                    dtype=torch.int64,
                    device=self.device,
                )
                selected_types[actionable] = action_types[parent_rows, slots]
                selected_totals[actionable] = totals[parent_rows, slots]
                child.apply_actions(
                    selected_types,
                    selected_totals,
                    actionable,
                )
            layers.append(
                _FrontierLayer(
                    parent_count=parent_count,
                    child_parents=child_parents,
                    child_slots=child_slots,
                    traverser_mask=traverser_mask,
                    features=features,
                    legal_masks=valid,
                    strategy=strategy,
                )
            )
            state = child
            remaining_depth -= 1
            self.frontier_layers += 1
            self.maximum_frontier_rows = max(self.maximum_frontier_rows, state.batch_size)

        if values is None:
            if bool((~state.terminal).any().item()):
                self.depth_limit_rollouts += int((~state.terminal).sum().item())
                state.rollout(128, self.generator)
            values = state.utilities()[:, traverser] / utility_scale

        for layer in reversed(layers):
            parent_values = torch.zeros((layer.parent_count,), dtype=torch.float32, device=self.device)
            identity = layer.child_slots < 0
            if bool(identity.any().item()):
                parent_values[layer.child_parents[identity]] = values[identity]
            non_traverser_edges = (~identity) & ~layer.traverser_mask[layer.child_parents]
            if bool(non_traverser_edges.any().item()):
                parent_values[layer.child_parents[non_traverser_edges]] = values[non_traverser_edges]

            traverser_edges = (~identity) & layer.traverser_mask[layer.child_parents]
            action_values = torch.zeros(
                (layer.parent_count, self.encoder.action_dim),
                dtype=torch.float32,
                device=self.device,
            )
            if bool(traverser_edges.any().item()):
                action_values[
                    layer.child_parents[traverser_edges],
                    layer.child_slots[traverser_edges],
                ] = values[traverser_edges]
            if bool(layer.traverser_mask.any().item()):
                expected = (action_values * layer.strategy).sum(dim=1)
                parent_values[layer.traverser_mask] = expected[layer.traverser_mask]
                advantages = (action_values - expected[:, None]) * layer.legal_masks.float()
                self.advantage_memories[traverser].add(
                    layer.features[layer.traverser_mask],
                    layer.legal_masks[layer.traverser_mask],
                    advantages[layer.traverser_mask],
                    iteration,
                    self.generator,
                )
            values = parent_values
        return values

    def _train_network(self, player: int, strategy: bool) -> float:
        memory = self.strategy_memories[player] if strategy else self.advantage_memories[player]
        steps = self.config.strategy_train_steps if strategy else self.config.advantage_train_steps
        if memory.size == 0:
            return 0.0
        network = self._new_network()
        optimizer = torch.optim.Adam(network.parameters(), lr=self.config.learning_rate)
        total = 0.0
        network.train()
        for _ in range(steps):
            features, masks, targets, iterations = memory.sample(self.config.batch_size, self.generator)
            weights = iterations if self.config.linear_weighting else torch.ones_like(iterations)
            weights = weights / weights.mean().clamp_min(1e-8)
            optimizer.zero_grad(set_to_none=True)
            outputs = network(features)
            if strategy:
                log_probabilities = torch.log_softmax(outputs.masked_fill(~masks, -1e9), dim=1)
                per_sample = -(targets * log_probabilities).sum(dim=1)
            else:
                per_sample = (((outputs - targets) ** 2) * masks.float()).sum(dim=1) / masks.sum(dim=1).clamp_min(1)
            loss = (per_sample * weights).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().item())
        if strategy:
            self.strategy_networks[player] = network
        else:
            self.advantage_networks[player] = network
        return total / steps

    def stats(self) -> dict[str, Any]:
        advantage_iteration_ranges = [memory.iteration_range() for memory in self.advantage_memories]
        strategy_iteration_ranges = [memory.iteration_range() for memory in self.strategy_memories]
        all_ranges = advantage_iteration_ranges + strategy_iteration_ranges
        return {
            "algorithm": "level_synchronous_cuda_deep_cfr",
            "num_players": self.num_players,
            "minimum_players": self.config.minimum_players,
            "maximum_players": self.num_players,
            "supported_player_counts": list(range(self.config.minimum_players, self.num_players + 1)),
            "randomized_player_counts": self.config.minimum_players < self.num_players,
            "player_count_traversals": dict(self.player_count_traversals),
            "iterations": self.completed_iterations,
            "traversals": self.traversals,
            "frontier_layers": self.frontier_layers,
            "maximum_frontier_rows": self.maximum_frontier_rows,
            "maximum_projected_frontier_rows": self.maximum_projected_frontier_rows,
            "frontier_chunk_splits": self.frontier_chunk_splits,
            "depth_limit_rollouts": self.depth_limit_rollouts,
            "advantage_samples": [memory.size for memory in self.advantage_memories],
            "advantage_samples_seen": [memory.samples_seen for memory in self.advantage_memories],
            "advantage_iteration_ranges": advantage_iteration_ranges,
            "strategy_samples": [memory.size for memory in self.strategy_memories],
            "strategy_samples_seen": [memory.samples_seen for memory in self.strategy_memories],
            "strategy_iteration_ranges": strategy_iteration_ranges,
            "latest_iteration_retained": bool(
                self.completed_iterations > 0
                and all(retained is not None and retained[1] == self.completed_iterations for retained in all_ranges)
            ),
            "loss": {key: list(value) for key, value in self.losses.items()},
        }

    def _validate_reservoir_freshness(self) -> None:
        """Reject snapshots exhibiting the pre-fix frozen-reservoir signature."""
        if self.completed_iterations < 4:
            return
        minimum_plausible_latest = self.completed_iterations // 2
        stale: list[str] = []
        for kind, memories in (
            ("advantage", self.advantage_memories),
            ("strategy", self.strategy_memories),
        ):
            for player, memory in enumerate(memories):
                retained = memory.iteration_range()
                if (
                    memory.samples_seen > memory.capacity
                    and (retained is None or retained[1] < minimum_plausible_latest)
                ):
                    stale.append(f"{kind}[{player}]={retained}")
        if stale:
            details = ", ".join(stale)
            raise ValueError(
                "CUDA Deep CFR snapshot reservoirs are severely stale "
                f"at completed iteration {self.completed_iterations} ({details}). "
                "This snapshot is affected by the pre-fix reservoir replacement bug; start a fresh run."
            )

    def inference_payload(self) -> dict[str, Any]:
        return {
            "policy_version": DEEP_CFR_POLICY_VERSION,
            "algorithm": "deep_cfr_average_strategy",
            "trainer": "level_synchronous_cuda_deep_cfr",
            "num_players": self.num_players,
            "supported_player_counts": list(range(self.config.minimum_players, self.num_players + 1)),
            "encoder": self.encoder.state_dict(),
            "hidden": self.config.hidden,
            "strategy_networks": [network.state_dict() for network in self.strategy_networks],
            "table_config": {
                "seats": self.table_config.seats,
                "small_blind": self.table_config.small_blind,
                "big_blind": self.table_config.big_blind,
                "starting_stacks": list(self.table_config.starting_stacks),
            },
            "training_stats": self.stats(),
        }

    def save_inference_policy(self, path: str | Path) -> None:
        target = Path(path)
        _atomic_torch_save(self.inference_payload(), target)

    def snapshot_payload(self) -> dict[str, Any]:
        return {
            "snapshot_version": CUDA_DEEP_CFR_SNAPSHOT_VERSION,
            "table_config": self.inference_payload()["table_config"],
            "config": asdict(self.config),
            "encoder": self.encoder.state_dict(),
            "advantage_networks": [network.state_dict() for network in self.advantage_networks],
            "strategy_networks": [network.state_dict() for network in self.strategy_networks],
            "advantage_memories": [memory.state_dict() for memory in self.advantage_memories],
            "strategy_memories": [memory.state_dict() for memory in self.strategy_memories],
            "generator_state": self.generator.get_state().cpu(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": [state.cpu() for state in torch.cuda.get_rng_state_all()],
            "stats": self.stats(),
        }

    def save_snapshot(self, path: str | Path) -> None:
        target = Path(path)
        _atomic_torch_save(self.snapshot_payload(), target)

    def load_snapshot(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        if int(payload.get("snapshot_version", -1)) != CUDA_DEEP_CFR_SNAPSHOT_VERSION:
            raise ValueError(f"Unsupported CUDA Deep CFR snapshot version: {payload.get('snapshot_version')}")
        stored_config = dict(payload["config"])
        stored_config.setdefault("minimum_players", int(payload["table_config"]["seats"]))
        current_config = asdict(self.config)
        incompatible = {
            key: (stored_config.get(key), current_config.get(key))
            for key in current_config
            if key not in RESUME_RUNTIME_CONFIG_FIELDS and stored_config.get(key) != current_config.get(key)
        }
        if incompatible or payload["encoder"] != self.encoder.state_dict():
            raise ValueError("CUDA Deep CFR snapshot configuration is incompatible")
        expected_table = self.inference_payload()["table_config"]
        if payload["table_config"] != expected_table:
            raise ValueError("CUDA Deep CFR snapshot table configuration is incompatible")
        for key in ("advantage_networks", "strategy_networks", "advantage_memories", "strategy_memories"):
            if len(payload[key]) != self.num_players:
                raise ValueError(f"CUDA Deep CFR snapshot has the wrong number of {key}")
        for network, state in zip(self.advantage_networks, payload["advantage_networks"]):
            network.load_state_dict(state)
        for network, state in zip(self.strategy_networks, payload["strategy_networks"]):
            network.load_state_dict(state)
        for memory, state in zip(self.advantage_memories, payload["advantage_memories"]):
            memory.load_state_dict(state)
        for memory, state in zip(self.strategy_memories, payload["strategy_memories"]):
            memory.load_state_dict(state)
        self.generator.set_state(payload["generator_state"].cpu())
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_states"]])
        stats = payload["stats"]
        self.completed_iterations = int(stats["iterations"])
        self.traversals = int(stats["traversals"])
        self.frontier_layers = int(stats["frontier_layers"])
        self.maximum_frontier_rows = int(stats["maximum_frontier_rows"])
        self.maximum_projected_frontier_rows = int(
            stats.get("maximum_projected_frontier_rows", self.maximum_frontier_rows)
        )
        self.frontier_chunk_splits = int(stats.get("frontier_chunk_splits", 0))
        self.depth_limit_rollouts = int(stats["depth_limit_rollouts"])
        stored_player_count_traversals = stats.get("player_count_traversals")
        if isinstance(stored_player_count_traversals, dict):
            self.player_count_traversals = {
                int(player_count): int(traversals)
                for player_count, traversals in stored_player_count_traversals.items()
            }
        else:
            self.player_count_traversals = {self.num_players: self.traversals}
        self.losses = {key: list(value) for key, value in stats["loss"].items()}
        self._validate_reservoir_freshness()
