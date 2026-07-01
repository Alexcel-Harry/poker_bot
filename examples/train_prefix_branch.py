from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import MutableMapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from poker_arena.cfr import PrefixBranchCFRTrainer, PrefixBranchTrainingConfig  # noqa: E402
from poker_arena.table import TableConfig  # noqa: E402


def parse_gpu_ids(raw: str | None) -> tuple[int, ...]:
    if raw is None:
        return ()
    text = raw.strip().lower()
    if text in {"", "none", "cpu", "off"}:
        return ()
    ids: list[int] = []
    for part in text.split(","):
        value = part.strip()
        if not value:
            continue
        try:
            gpu_id = int(value)
        except ValueError as exc:
            raise ValueError(f"GPU ids must be comma-separated integers, got {raw!r}") from exc
        if gpu_id < 0:
            raise ValueError("GPU ids must be non-negative")
        ids.append(gpu_id)
    return tuple(ids)


def parse_required_amounts(raw: str | None) -> tuple[int, ...]:
    if raw is None or not raw.strip():
        return ()
    amounts: list[int] = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        amount = int(value)
        if amount <= 0:
            raise ValueError("Required integer actions must be positive raise_to totals")
        amounts.append(amount)
    return tuple(amounts)


def configure_visible_gpus(gpu_ids: Sequence[int], env: MutableMapping[str, str] | None = None) -> None:
    target_env = os.environ if env is None else env
    target_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids) if gpu_ids else ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run integer-action prefix-branch CFR training.")
    parser.add_argument("--iterations", type=int, default=100)
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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--gpus", default="none", help="Comma-separated CUDA GPU ids, e.g. 0 or 0,1. Use 'none' for CPU.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON summary path.")
    parser.add_argument("--preview-samples", type=int, default=3)
    return parser


def run_from_args(args: argparse.Namespace, env: MutableMapping[str, str] | None = None) -> dict[str, object]:
    gpu_ids = parse_gpu_ids(args.gpus)
    configure_visible_gpus(gpu_ids, env)
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
    trainer = PrefixBranchCFRTrainer(
        table_config=TableConfig(
            seats=args.seats,
            small_blind=args.small_blind,
            big_blind=args.big_blind,
            starting_stacks=[args.starting_stack] * args.seats,
            seed=args.seed,
        ),
        config=config,
    )
    result = trainer.train(iterations=args.iterations)
    preview = [
        {
            "infoset": sample.infoset_key,
            "action": sample.action.to_dict(),
            "target_utility": sample.target_utility,
        }
        for sample in trainer.training_samples[: max(0, args.preview_samples)]
    ]
    summary: dict[str, object] = {
        "prefix_branch": True,
        "device": args.device,
        "gpu_ids": list(gpu_ids),
        "rollout_device": "cpu",
        "gpu_note": "Current prefix-branch rollouts are CPU-side; --gpus only constrains CUDA visibility for future learned models.",
        "iterations": result.iterations,
        "information_sets": result.information_sets,
        "episodes": result.episodes,
        "training_samples": len(trainer.training_samples),
        "branch_width": config.branch_width,
        "branch_depth": config.branch_depth,
        "integer_action_budget": config.integer_action_budget,
        "max_workers": config.max_workers,
        "required_integer_actions": list(required_amounts),
        "sample_preview": preview,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_from_args(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
