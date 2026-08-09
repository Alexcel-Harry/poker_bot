from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from poker_arena.actions import Action, LegalActions
from poker_arena.state import PlayerHandView


class BotPolicy(Protocol):
    def choose_action(self, view: PlayerHandView | None, legal_actions: LegalActions) -> Action:
        ...


@dataclass
class CheckCallBot:
    def choose_action(self, view: PlayerHandView | None, legal_actions: LegalActions) -> Action:
        if legal_actions.can_check:
            return Action.check()
        if legal_actions.can_call:
            return Action.call()
        return Action.fold()


@dataclass
class RandomLegalBot:
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def choose_action(self, view: PlayerHandView | None, legal_actions: LegalActions) -> Action:
        actions: list[Action] = []
        if legal_actions.can_fold:
            actions.append(Action.fold())
        if legal_actions.can_check:
            actions.append(Action.check())
        if legal_actions.can_call:
            actions.append(Action.call())
        if legal_actions.can_raise and legal_actions.min_raise_to is not None and legal_actions.max_raise_to is not None:
            actions.append(Action.raise_to(self._rng.randint(legal_actions.min_raise_to, legal_actions.max_raise_to)))
        if not actions:
            return Action.fold()
        return self._rng.choice(actions)


class TorchPolicyBot:
    """Loads a Torch `.pt` action-value checkpoint and selects the best legal action."""

    def __init__(self, model: object, metadata: object, payload: dict[str, object], device: object) -> None:
        from poker_arena.cfr import ActionEmbedding, StateFeatureEncoder
        from poker_arena.embedding import TrajectoryEncoder

        self.model = model
        self.metadata = metadata
        self.payload = payload
        self.device = device
        self.state_encoder = StateFeatureEncoder()
        state_dim = getattr(metadata, "state_dim", None)
        if state_dim != self.state_encoder.dimension:
            raise ValueError(
                f"Checkpoint state dimension {state_dim} does not match encoder dimension {self.state_encoder.dimension}"
            )
        self.trajectory_encoder = TrajectoryEncoder()
        action_dim = int(getattr(metadata, "action_dim", ActionEmbedding.legacy_dimension))
        if action_dim not in (ActionEmbedding.legacy_dimension, ActionEmbedding.dimension_without_trajectory):
            raise ValueError(f"Unsupported checkpoint action dimension: {action_dim}")
        self.action_embedding = ActionEmbedding(
            include_pot_features=action_dim == ActionEmbedding.dimension_without_trajectory
        )
        model_config = payload.get("model_config", {})
        action_sampler = model_config.get("action_sampler", {}) if isinstance(model_config, dict) else {}
        self.integer_action_budget = int(action_sampler.get("integer_action_budget", 32)) if isinstance(action_sampler, dict) else 32
        required = action_sampler.get("required_integer_actions", ()) if isinstance(action_sampler, dict) else ()
        self.required_integer_actions = tuple(int(amount) for amount in required)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str = "auto",
        seed: int | None = None,
    ) -> "TorchPolicyBot":
        from poker_arena.cfr.torch_model import load_checkpoint, torch

        if torch is None:
            raise ModuleNotFoundError("Install the training extra with `pip install -e .[train]` to load Torch policy bots")
        target_device = device
        if device == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        raw_payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if isinstance(raw_payload, dict) and raw_payload.get("algorithm") == "deep_cfr_average_strategy":
            return DeepCFRAveragePolicyBot.from_payload(  # type: ignore[return-value]
                raw_payload,
                device=target_device,
                seed=seed,
            )
        model, metadata, payload = load_checkpoint(path, device=target_device)
        return cls(model=model, metadata=metadata, payload=payload, device=torch.device(target_device))

    def choose_action(self, view: PlayerHandView | None, legal_actions: LegalActions) -> Action:
        from poker_arena.cfr.torch_model import torch

        if torch is None:
            raise ModuleNotFoundError("Install the training extra with `pip install -e .[train]` to use Torch policy bots")
        candidates = self._candidate_actions(legal_actions)
        if not candidates:
            return Action.fold()
        state_features = self.state_encoder.encode_view(view, legal_actions)
        trajectory_features = self.trajectory_encoder.encode_events(view.events) if view is not None else [0.0] * self.trajectory_encoder.dimension
        rows = [
            state_features
            + trajectory_features
            + self.action_embedding.encode(
                action,
                legal_actions,
                pot=view.pot if view is not None else None,
            )
            for action in candidates
        ]
        tensor = torch.tensor(rows, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            scores = self.model(tensor).squeeze(-1)
        best_index = int(torch.argmax(scores).detach().cpu().item())
        return candidates[best_index]

    def _candidate_actions(self, legal_actions: LegalActions) -> list[Action]:
        actions: list[Action] = []
        if legal_actions.can_fold:
            actions.append(Action.fold())
        if legal_actions.can_check:
            actions.append(Action.check())
        if legal_actions.can_call:
            actions.append(Action.call())
        if legal_actions.can_raise and legal_actions.min_raise_to is not None and legal_actions.max_raise_to is not None:
            minimum = legal_actions.min_raise_to
            maximum = legal_actions.max_raise_to
            totals = {minimum, maximum}
            totals.update(amount for amount in self.required_integer_actions if minimum <= amount <= maximum)
            budget = max(1, self.integer_action_budget)
            if budget > 1 and maximum > minimum:
                for index in range(budget):
                    fraction = index / max(1, budget - 1)
                    totals.add(round(minimum + fraction * (maximum - minimum)))
            actions.extend(Action.raise_to(total) for total in sorted(totals))
        return self._dedupe(actions)

    @staticmethod
    def _dedupe(actions: list[Action]) -> list[Action]:
        seen: set[tuple[str, int | None]] = set()
        result: list[Action] = []
        for action in actions:
            key = (action.action_type.value, action.total)
            if key not in seen:
                seen.add(key)
                result.append(action)
        return result


class DeepCFRAveragePolicyBot:
    """Samples the learned Deep CFR average strategy at each information set."""

    def __init__(
        self,
        networks: list[object],
        encoder: object,
        payload: dict[str, object],
        device: object,
        seed: int | None = None,
    ) -> None:
        from poker_arena.abstraction import ActionAbstraction

        self.networks = networks
        self.num_players = len(networks)
        supported = payload.get("supported_player_counts", [self.num_players])
        self.supported_player_counts = tuple(int(player_count) for player_count in supported)  # type: ignore[arg-type]
        self.encoder = encoder
        self.payload = payload
        self.device = device
        self.abstraction = ActionAbstraction.compact()
        self.rng = random.Random(seed)
        self._warned_player_counts: set[int] = set()

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, object],
        device: str = "cpu",
        seed: int | None = None,
    ) -> "DeepCFRAveragePolicyBot":
        from poker_arena.cfr.deep_cfr import DEEP_CFR_POLICY_VERSION, DeepCFRNetwork, torch
        from poker_arena.cfr.holdem_deep_cfr import HoldemDeepCFRFeatureEncoder, TensorHoldemDeepCFRFeatureEncoder

        if torch is None:
            raise ModuleNotFoundError("Install the training extra to load Deep CFR policies")
        if int(payload.get("policy_version", -1)) != DEEP_CFR_POLICY_VERSION:
            raise ValueError(f"Unsupported Deep CFR policy version: {payload.get('policy_version')}")
        encoder_payload = payload.get("encoder")
        if not isinstance(encoder_payload, dict):
            raise ValueError("Deep CFR policy is missing its encoder schema")
        if encoder_payload.get("encoder") == "holdem_private_state_ordered_history_v1":
            encoder = HoldemDeepCFRFeatureEncoder(int(encoder_payload["max_history_actions"]))
        elif encoder_payload.get("encoder") == "holdem_tensor_state_trajectory_v1":
            encoder = TensorHoldemDeepCFRFeatureEncoder()
        else:
            raise ValueError("Only Hold'em Deep CFR inference policies can be loaded as arena bots")
        encoder.validate_state_dict(encoder_payload)
        hidden = tuple(int(value) for value in payload["hidden"])  # type: ignore[arg-type]
        target_device = torch.device(device)
        network_states = payload.get("strategy_networks")
        if not isinstance(network_states, list):
            raise ValueError("Deep CFR policy is missing its average-strategy networks")
        num_players = int(payload.get("num_players", len(network_states)))
        if num_players < 2 or len(network_states) != num_players:
            raise ValueError("Deep CFR policy must contain one average-strategy network per player")
        supported_player_counts = payload.get("supported_player_counts", [num_players])
        if (
            not isinstance(supported_player_counts, list)
            or not supported_player_counts
            or any(
                not isinstance(player_count, int) or not 2 <= player_count <= num_players
                for player_count in supported_player_counts
            )
            or supported_player_counts != sorted(set(supported_player_counts))
        ):
            raise ValueError("Deep CFR policy has invalid supported player counts")
        table_config = payload.get("table_config")
        if isinstance(table_config, dict) and int(table_config.get("seats", num_players)) != num_players:
            raise ValueError("Deep CFR policy player count disagrees with its table configuration")
        networks: list[object] = []
        for state in network_states:
            network = DeepCFRNetwork(encoder.input_dim, encoder.action_dim, hidden).to(target_device)
            network.load_state_dict(state)
            network.eval()
            networks.append(network)
        return cls(networks, encoder, payload, target_device, seed=seed)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str = "auto",
        seed: int | None = None,
    ) -> "DeepCFRAveragePolicyBot":
        from poker_arena.cfr.deep_cfr import torch

        if torch is None:
            raise ModuleNotFoundError("Install the training extra to load Deep CFR policies")
        target = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("Deep CFR checkpoint payload must be a dictionary")
        return cls.from_payload(payload, device=target, seed=seed)

    def choose_action(self, view: PlayerHandView | None, legal_actions: LegalActions) -> Action:
        from poker_arena.cfr.deep_cfr import torch

        if torch is None:
            raise ModuleNotFoundError("Install the training extra to use Deep CFR policies")
        if view is None:
            return CheckCallBot().choose_action(view, legal_actions)
        table_players = len(view.stacks)
        if not 2 <= table_players <= 9:
            raise ValueError(f"Deep CFR inference supports tables with 2-9 players, but received {table_players}")
        actor = view.current_actor
        if actor is None or actor not in view.stacks:
            raise ValueError("Deep CFR policy requires a live actor present in the table view")
        player_count_supported = table_players in self.supported_player_counts
        if not player_count_supported and table_players not in self._warned_player_counts:
            trained_description = (
                str(self.supported_player_counts[0])
                if len(self.supported_player_counts) == 1
                else f"{self.supported_player_counts[0]}-{self.supported_player_counts[-1]}"
            )
            warnings.warn(
                f"Deep CFR policy was trained for {trained_description} players but is being used with "
                f"{table_players}; averaging its seat networks for experimental, "
                "out-of-distribution inference",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_player_counts.add(table_players)
        stack = view.stacks.get(actor, legal_actions.max_raise_to or 1) + legal_actions.actor_commitment
        abstract_actions = self.abstraction.actions_from_legal(legal_actions, pot=view.pot, stack=max(1, stack))
        concrete = {action.label: action.to_action() for action in abstract_actions}
        labels = list(concrete)
        if not labels:
            return Action.fold()
        features = self.encoder.encode_view(view, legal_actions)
        inputs = torch.tensor([features], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if not player_count_supported:
                outputs = torch.stack([network(inputs)[0] for network in self.networks]).mean(dim=0)
            else:
                outputs = self.networks[actor](inputs)[0]
        logits = torch.stack([outputs[self.encoder.action_index[label]] for label in labels])
        probabilities = torch.softmax(logits, dim=0).detach().cpu().tolist()
        draw = self.rng.random()
        cumulative = 0.0
        for label, probability in zip(labels, probabilities):
            cumulative += float(probability)
            if draw <= cumulative:
                return concrete[label]
        return concrete[labels[-1]]
