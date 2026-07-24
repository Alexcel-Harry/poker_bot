from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

from poker_arena.actions import Action, ActionType
from poker_arena.cards import Card
from poker_arena.state import HandState, PlayerHandView, Street


CHECKPOINT_VERSION = 2


try:  # pragma: no cover - import branch depends on optional dependency.
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without torch.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def _require_torch() -> Any:
    if torch is None or nn is None:
        raise ModuleNotFoundError("Install the training extra with `pip install -e .[train]` to use Torch models")
    return torch


class StateFeatureEncoder:
    """Fixed numeric features for an engine state or player view."""

    max_seats = 9
    street_values = (Street.PREFLOP.value, Street.FLOP.value, Street.TURN.value, Street.RIVER.value, Street.SHOWDOWN.value)
    base_dimension = 31
    card_dimension = 52
    dimension = base_dimension + card_dimension * 2
    suit_offsets = {"c": 0, "d": 13, "h": 26, "s": 39}

    def encode_state(self, state: HandState, actor: int) -> list[float]:
        actor_player = state.player_by_seat(actor)
        stacks = [player.stack for player in state.players]
        folded = [1.0 if player.folded else 0.0 for player in state.players]
        all_in = [1.0 if player.all_in else 0.0 for player in state.players]
        return self._encode(
            seats=state.config.seats,
            actor=actor,
            button=state.button,
            street=state.street.value,
            pot=state.total_pot,
            current_bet=state.current_bet,
            actor_stack=actor_player.stack,
            stacks=stacks,
            folded=folded,
            all_in=all_in,
            board_count=len(state.board),
            hole_cards=actor_player.hole_cards,
            board=state.board,
        )

    def encode_view(self, view: PlayerHandView | None, legal_actions: Any) -> list[float]:
        if view is None:
            return self._encode(
                seats=1,
                actor=0,
                button=0,
                street=Street.PREFLOP.value,
                pot=0,
                current_bet=getattr(legal_actions, "current_bet", 0),
                actor_stack=max(1, getattr(legal_actions, "max_raise_to", 1) or 1),
                stacks=[],
                folded=[],
                all_in=[],
                board_count=0,
                hole_cards=[],
                board=[],
            )
        stacks = [view.stacks[seat] for seat in sorted(view.stacks)]
        folded = [1.0 if view.folded[seat] else 0.0 for seat in sorted(view.folded)]
        all_in = [1.0 if view.all_in[seat] else 0.0 for seat in sorted(view.all_in)]
        actor = view.current_actor if view.current_actor is not None else 0
        actor_stack = view.stacks.get(actor, max(stacks) if stacks else 1)
        hole_cards = next((cards for cards in view.hole_cards.values() if cards is not None), ())
        return self._encode(
            seats=len(stacks),
            actor=actor,
            button=view.button,
            street=view.street.value,
            pot=view.pot,
            current_bet=getattr(legal_actions, "current_bet", 0),
            actor_stack=actor_stack,
            stacks=stacks,
            folded=folded,
            all_in=all_in,
            board_count=len(view.board),
            hole_cards=hole_cards,
            board=view.board,
        )

    def _encode(
        self,
        seats: int,
        actor: int,
        button: int,
        street: str,
        pot: int,
        current_bet: int,
        actor_stack: int,
        stacks: Sequence[int],
        folded: Sequence[float],
        all_in: Sequence[float],
        board_count: int,
        hole_cards: Sequence[Card],
        board: Sequence[Card],
    ) -> list[float]:
        denom = max(1.0, float(pot + sum(stacks)), float(pot + actor_stack))
        features: list[float] = [
            seats / self.max_seats,
            actor / max(1, self.max_seats - 1),
            button / max(1, self.max_seats - 1),
            pot / denom,
            current_bet / max(1.0, float(actor_stack + current_bet)),
            actor_stack / denom,
            board_count / 5.0,
        ]
        features.extend(1.0 if street == value else 0.0 for value in self.street_values)
        padded_stacks = list(stacks[: self.max_seats]) + [0] * max(0, self.max_seats - len(stacks))
        padded_folded = list(folded[: self.max_seats]) + [0.0] * max(0, self.max_seats - len(folded))
        features.extend(stack / denom for stack in padded_stacks)
        features.extend(padded_folded)
        features.append(sum(all_in) / max(1, seats))
        features.extend(self._encode_cards(hole_cards))
        features.extend(self._encode_cards(board))
        if len(features) != self.dimension:
            raise AssertionError(f"State feature dimension changed: {len(features)} != {self.dimension}")
        return features

    @classmethod
    def _encode_cards(cls, cards: Sequence[Card]) -> list[float]:
        features = [0.0] * cls.card_dimension
        for card in cards:
            index = cls.suit_offsets[card.suit.value] + int(card.rank) - 2
            features[index] = 1.0
        return features


@dataclass(frozen=True)
class TorchTrainingSample:
    state_features: list[float]
    trajectory_features: list[float]
    action_features: list[float]
    action: dict[str, int | str | None]
    target_utility: float
    weight: float = 1.0

    def feature_vector(self) -> list[float]:
        return list(self.state_features) + list(self.trajectory_features) + list(self.action_features)


@dataclass(frozen=True)
class TorchCheckpointMetadata:
    state_dim: int
    action_dim: int
    hidden: tuple[int, ...]
    dropout: float
    table_defaults: dict[str, Any]
    action_sampler: dict[str, Any]
    training: dict[str, Any]
    trajectory_dim: int = 14

    @property
    def input_dim(self) -> int:
        return self.state_dim + self.trajectory_dim + self.action_dim

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_dim": self.state_dim,
            "trajectory_dim": self.trajectory_dim,
            "action_dim": self.action_dim,
            "input_dim": self.input_dim,
            "hidden": list(self.hidden),
            "dropout": self.dropout,
            "table_defaults": dict(self.table_defaults),
            "action_sampler": dict(self.action_sampler),
            "training": dict(self.training),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TorchCheckpointMetadata":
        return cls(
            state_dim=int(data["state_dim"]),
            trajectory_dim=int(data.get("trajectory_dim", 14)),
            action_dim=int(data["action_dim"]),
            hidden=tuple(int(value) for value in data["hidden"]),
            dropout=float(data.get("dropout", 0.0)),
            table_defaults=dict(data.get("table_defaults", {})),
            action_sampler=dict(data.get("action_sampler", {})),
            training=dict(data.get("training", {})),
        )


if nn is not None:

    class ActionValueNet(nn.Module):  # type: ignore[misc]
        def __init__(self, input_dim: int, hidden: tuple[int, ...], dropout: float = 0.0) -> None:
            super().__init__()
            layers: list[Any] = []
            last = input_dim
            for width in hidden:
                layers.append(nn.Linear(last, width))
                layers.append(nn.LayerNorm(width))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                last = width
            layers.append(nn.Linear(last, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x: Any) -> Any:
            return self.net(x)

else:

    class ActionValueNet:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_torch()


class TorchReplayBuffer:
    def __init__(self, samples: Iterable[TorchTrainingSample] = ()) -> None:
        self.samples = list(samples)

    def add(self, sample: TorchTrainingSample) -> None:
        self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def batches(self, batch_size: int, shuffle: bool = True, seed: int | None = None) -> Iterable[list[TorchTrainingSample]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = list(range(len(self.samples)))
        if shuffle:
            random.Random(seed).shuffle(indices)
        for start in range(0, len(indices), batch_size):
            yield [self.samples[index] for index in indices[start : start + batch_size]]

    def tensors_for(self, batch: Sequence[TorchTrainingSample], device: Any) -> tuple[Any, Any, Any]:
        torch_module = _require_torch()
        x = torch_module.tensor([sample.feature_vector() for sample in batch], dtype=torch_module.float32, device=device)
        y = torch_module.tensor([[sample.target_utility] for sample in batch], dtype=torch_module.float32, device=device)
        w = torch_module.tensor([[sample.weight] for sample in batch], dtype=torch_module.float32, device=device)
        return x, y, w


def train_value_model(
    model: Any,
    buffer: TorchReplayBuffer,
    device: Any,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    use_amp: bool = False,
) -> dict[str, Any]:
    torch_module = _require_torch()
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if len(buffer) == 0:
        raise ValueError("Cannot train from an empty replay buffer")
    model.to(device)
    model.train()
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    amp_enabled = bool(use_amp and device_type == "cuda")
    amp_dtype = torch_module.bfloat16 if amp_enabled and torch_module.cuda.is_bf16_supported() else torch_module.float16
    scaler = torch_module.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch_module.float16)
    if device_type == "cuda":
        torch_module.set_float32_matmul_precision("high")
    optimizer = torch_module.optim.AdamW(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    for epoch in range(epochs):
        epoch_losses: list[float] = []
        for batch in buffer.batches(batch_size=batch_size, shuffle=True, seed=epoch):
            x, y, w = buffer.tensors_for(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch_module.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                pred = model(x)
                loss = (((pred - y) ** 2) * w).mean()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        if epoch_losses:
            losses.append(sum(epoch_losses) / len(epoch_losses))
    return {
        "loss": losses,
        "samples": len(buffer),
        "epochs": epochs,
        "device": str(device),
        "amp": amp_enabled,
        "amp_dtype": str(amp_dtype).removeprefix("torch.") if amp_enabled else None,
    }


def build_checkpoint_payload(
    metadata: TorchCheckpointMetadata,
    model_state_dict: Mapping[str, Any],
    training_stats: Mapping[str, Any] | None = None,
    normalization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state_dict": dict(model_state_dict),
        "model_config": metadata.to_dict(),
        "normalization": dict(normalization or {}),
        "training_stats": dict(training_stats or {}),
    }


def validate_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    required = {"checkpoint_version", "model_state_dict", "model_config", "normalization", "training_stats"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {', '.join(missing)}")
    if int(payload["checkpoint_version"]) != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version: {payload['checkpoint_version']}")
    TorchCheckpointMetadata.from_dict(payload["model_config"])


def save_checkpoint(
    path: str | Path,
    model: Any,
    metadata: TorchCheckpointMetadata,
    training_stats: Mapping[str, Any] | None = None,
    normalization: Mapping[str, Any] | None = None,
) -> None:
    torch_module = _require_torch()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch_module.save(
        build_checkpoint_payload(metadata, state_dict, training_stats=training_stats, normalization=normalization),
        target,
    )


def load_checkpoint(path: str | Path, device: str | Any = "cpu") -> tuple[Any, TorchCheckpointMetadata, dict[str, Any]]:
    torch_module = _require_torch()
    payload = torch_module.load(Path(path), map_location=device)
    validate_checkpoint_payload(payload)
    metadata = TorchCheckpointMetadata.from_dict(payload["model_config"])
    model = ActionValueNet(input_dim=metadata.input_dim, hidden=metadata.hidden, dropout=metadata.dropout)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, metadata, dict(payload)
