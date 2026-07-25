from __future__ import annotations

import random
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
        self.action_embedding = ActionEmbedding()
        model_config = payload.get("model_config", {})
        action_sampler = model_config.get("action_sampler", {}) if isinstance(model_config, dict) else {}
        self.integer_action_budget = int(action_sampler.get("integer_action_budget", 32)) if isinstance(action_sampler, dict) else 32
        required = action_sampler.get("required_integer_actions", ()) if isinstance(action_sampler, dict) else ()
        self.required_integer_actions = tuple(int(amount) for amount in required)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto") -> "TorchPolicyBot":
        from poker_arena.cfr.torch_model import load_checkpoint, torch

        if torch is None:
            raise ModuleNotFoundError("Install the training extra with `pip install -e .[train]` to load Torch policy bots")
        target_device = device
        if device == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
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
            state_features + trajectory_features + self.action_embedding.encode(action, legal_actions)
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
