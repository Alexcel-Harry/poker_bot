from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import MutableMapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from examples.train_prefix_branch import configure_visible_gpus, parse_gpu_ids, parse_required_amounts  # noqa: E402
from poker_arena.cfr import PrefixBranchCFRTrainer, PrefixBranchTrainingConfig  # noqa: E402
from poker_arena.table import TableConfig  # noqa: E402


def parse_hidden(raw: str) -> tuple[int, ...]:
    hidden = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not hidden or any(width <= 0 for width in hidden):
        raise ValueError("Hidden layers must be positive comma-separated integers")
    return hidden


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a deployable Torch policy from prefix-branch samples.")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seats", type=int, default=3)
    parser.add_argument("--small-blind", type=int, default=10)
    parser.add_argument("--big-blind", type=int, default=20)
    parser.add_argument("--starting-stack", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--branch-width", type=int, default=32)
    parser.add_argument("--branch-depth", type=int, default=8)
    parser.add_argument("--integer-action-budget", type=int, default=32)
    parser.add_argument("--novelty-weight", type=float, default=1.0)
    parser.add_argument("--neighbor-weight", type=float, default=0.0)
    parser.add_argument("--required-integer-actions", default="")
    parser.add_argument("--max-actions-per-episode", type=int, default=200)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpus", default="none", help="Comma-separated CUDA GPU ids, e.g. 0 or 0,1. Use 'none' for CPU.")
    parser.add_argument("--hidden", default="512,512,256")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--model-out", type=Path, default=Path("runs/poker_policy.pt"))
    parser.add_argument("--summary-out", type=Path, default=None)
    return parser


def _select_device(requested: str, torch_module: object) -> object:
    if requested == "cpu":
        return torch_module.device("cpu")  # type: ignore[attr-defined]
    if requested == "cuda":
        if not torch_module.cuda.is_available():  # type: ignore[attr-defined]
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return torch_module.device("cuda")  # type: ignore[attr-defined]
    if torch_module.cuda.is_available():  # type: ignore[attr-defined]
        return torch_module.device("cuda")  # type: ignore[attr-defined]
    return torch_module.device("cpu")  # type: ignore[attr-defined]


def run_from_args(args: argparse.Namespace, env: MutableMapping[str, str] | None = None) -> dict[str, object]:
    gpu_ids = parse_gpu_ids(args.gpus)
    configure_visible_gpus(gpu_ids, env)

    from poker_arena.cfr.torch_model import (  # noqa: PLC0415
        ActionValueNet,
        TorchCheckpointMetadata,
        TorchReplayBuffer,
        save_checkpoint,
        torch,
        train_value_model,
    )

    if torch is None:
        raise ModuleNotFoundError("Install the training extra with `pip install -e .[train]` to run GPU training")

    required_amounts = parse_required_amounts(args.required_integer_actions)
    config = PrefixBranchTrainingConfig(
        branch_width=args.branch_width,
        branch_depth=args.branch_depth,
        integer_action_budget=args.integer_action_budget,
        novelty_weight=args.novelty_weight,
        neighbor_weight=args.neighbor_weight,
        required_integer_actions=required_amounts,
        max_actions_per_episode=args.max_actions_per_episode,
        random_seed=args.random_seed,
        max_workers=args.max_workers,
    )
    table_config = TableConfig(
        seats=args.seats,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        starting_stacks=[args.starting_stack] * args.seats,
        seed=args.seed,
    )
    trainer = PrefixBranchCFRTrainer(table_config=table_config, config=config)
    result = trainer.train(iterations=args.iterations)
    buffer = TorchReplayBuffer(trainer.torch_training_samples)
    if len(buffer) == 0:
        raise RuntimeError("No training samples were generated")

    first = buffer.samples[0]
    hidden = parse_hidden(args.hidden)
    metadata = TorchCheckpointMetadata(
        state_dim=len(first.state_features),
        trajectory_dim=len(first.trajectory_features),
        action_dim=len(first.action_features),
        hidden=hidden,
        dropout=args.dropout,
        table_defaults={
            "seats": args.seats,
            "small_blind": args.small_blind,
            "big_blind": args.big_blind,
            "starting_stack": args.starting_stack,
        },
        action_sampler={
            "integer_action_budget": args.integer_action_budget,
            "required_integer_actions": list(required_amounts),
        },
        training={
            "iterations": args.iterations,
            "branch_width": args.branch_width,
            "branch_depth": args.branch_depth,
            "max_workers": args.max_workers,
        },
    )
    device = _select_device(args.device, torch)
    model = ActionValueNet(input_dim=metadata.input_dim, hidden=hidden, dropout=args.dropout)
    used_data_parallel = False
    if str(device).startswith("cuda") and len(gpu_ids) > 1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        used_data_parallel = True
    stats = train_value_model(
        model,
        buffer,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    save_checkpoint(args.model_out, model, metadata, stats)
    summary: dict[str, object] = {
        "prefix_branch": True,
        "model_out": str(args.model_out),
        "device": str(device),
        "gpu_ids": list(gpu_ids),
        "data_parallel": used_data_parallel,
        "iterations": result.iterations,
        "information_sets": result.information_sets,
        "training_samples": len(buffer),
        "loss": stats["loss"],
    }
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_from_args(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
