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

from examples.train_deep_cfr import parse_hidden  # noqa: E402
from examples.train_prefix_branch import configure_visible_gpus, parse_gpu_ids  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train level-synchronous external-sampling Deep CFR for multi-player Hold'em on one CUDA GPU."
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--traversals-per-player", type=int, default=4096)
    parser.add_argument("--parallel-traversals", type=int, default=192)
    parser.add_argument("--max-traversal-depth", type=int, default=128)
    parser.add_argument("--max-frontier-rows", type=int, default=131_072)
    parser.add_argument("--advantage-capacity", type=int, default=300_000)
    parser.add_argument("--strategy-capacity", type=int, default=300_000)
    parser.add_argument("--hidden", default="512,512,256")
    parser.add_argument("--advantage-train-steps", type=int, default=1000)
    parser.add_argument("--strategy-train-steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--seats",
        type=int,
        default=9,
        help="Maximum player count; each traversal session samples between --min-players and this value.",
    )
    parser.add_argument("--min-players", type=int, default=3, help="Minimum randomized player count; must be 3-9.")
    parser.add_argument("--small-blind", type=int, default=5)
    parser.add_argument("--big-blind", type=int, default=10)
    parser.add_argument("--starting-stack", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--gpus", default="0", help="Exactly one CUDA GPU id.")
    parser.add_argument("--resume-snapshot", type=Path, default=None)
    parser.add_argument("--snapshot-out", type=Path, default=Path("runs/cuda_deep_cfr_snapshot.pt"))
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="Atomically refresh --snapshot-out after this many new iterations; 0 saves only at the end.",
    )
    parser.add_argument("--policy-out", type=Path, default=Path("runs/cuda_deep_cfr_average_policy.pt"))
    parser.add_argument("--summary-out", type=Path, default=Path("runs/cuda_deep_cfr_summary.json"))
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def require_single_gpu(args: argparse.Namespace, env: MutableMapping[str, str]) -> tuple[int, ...]:
    gpu_ids = parse_gpu_ids(args.gpus)
    if len(gpu_ids) != 1:
        raise ValueError("CUDA Deep CFR requires exactly one GPU id")
    configure_visible_gpus(gpu_ids, env)
    return gpu_ids


def run_from_args(args: argparse.Namespace, env: MutableMapping[str, str] | None = None) -> dict[str, object]:
    if not 3 <= args.seats <= 9:
        raise ValueError("CUDA Deep CFR production training requires --seats between 3 and 9")
    if not 3 <= args.min_players <= args.seats:
        raise ValueError("--min-players must be between 3 and --seats")
    if args.snapshot_every < 0:
        raise ValueError("--snapshot-every must be non-negative")
    target_env = os.environ if env is None else env
    gpu_ids = require_single_gpu(args, target_env)

    import torch  # noqa: PLC0415

    from poker_arena.cfr.cuda_deep_cfr import CudaDeepCFRConfig, CudaDeepCFRTrainer  # noqa: PLC0415
    from poker_arena.table import TableConfig  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is false")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible CUDA GPU, found {torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    table = TableConfig(
        args.seats,
        args.small_blind,
        args.big_blind,
        [args.starting_stack] * args.seats,
    )
    config = CudaDeepCFRConfig(
        iterations=args.iterations,
        traversals_per_player=args.traversals_per_player,
        parallel_traversals=args.parallel_traversals,
        max_traversal_depth=args.max_traversal_depth,
        max_frontier_rows=args.max_frontier_rows,
        advantage_capacity=args.advantage_capacity,
        strategy_capacity=args.strategy_capacity,
        hidden=parse_hidden(args.hidden),
        advantage_train_steps=args.advantage_train_steps,
        strategy_train_steps=args.strategy_train_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
        minimum_players=args.min_players,
    )
    trainer = CudaDeepCFRTrainer(table, config, device)
    if args.resume_snapshot is not None:
        trainer.load_snapshot(args.resume_snapshot)
    starting_iteration = trainer.completed_iterations

    def progress(done: int, target: int, traversals: int) -> None:
        relative = done - starting_iteration
        if args.log_every > 0 and (done == target or relative % args.log_every == 0):
            print(f"cuda deep cfr: iteration {done}/{target}, {traversals} traversals", file=sys.stderr)
        if args.snapshot_every > 0 and relative > 0 and relative % args.snapshot_every == 0 and done != target:
            trainer.save_snapshot(args.snapshot_out)
            print(f"cuda deep cfr: refreshed resumable snapshot at iteration {done}", file=sys.stderr)

    stats = trainer.train(iterations=args.iterations, progress_callback=progress)
    trainer.save_snapshot(args.snapshot_out)
    trainer.save_inference_policy(args.policy_out)
    summary: dict[str, object] = {
        "device": str(device),
        "gpu_ids": list(gpu_ids),
        "snapshot_out": str(args.snapshot_out),
        "policy_out": str(args.policy_out),
        **stats,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_from_args(args), indent=2))


if __name__ == "__main__":
    main()
