from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_hidden(raw: str) -> tuple[int, ...]:
    hidden = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not hidden or any(width <= 0 for width in hidden):
        raise ValueError("Hidden layers must be positive comma-separated integers")
    return hidden


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the recursive external-sampling Deep CFR reference implementation."
    )
    parser.add_argument("--game", choices=("kuhn", "leduc", "holdem"), default="kuhn")
    parser.add_argument("--iterations", type=int, default=100, help="New CFR iterations to run, including after resume.")
    parser.add_argument("--traversals-per-player", type=int, default=100)
    parser.add_argument("--advantage-capacity", type=int, default=100_000)
    parser.add_argument("--strategy-capacity", type=int, default=100_000)
    parser.add_argument("--hidden", default="128,128")
    parser.add_argument("--advantage-train-steps", type=int, default=200)
    parser.add_argument("--strategy-train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--small-blind", type=int, default=5)
    parser.add_argument("--big-blind", type=int, default=10)
    parser.add_argument("--starting-stack", type=int, default=200)
    parser.add_argument("--seats", type=int, default=3, help="Hold'em player count; must be 3-9.")
    parser.add_argument("--max-history-actions", type=int, default=24)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--snapshot-out", type=Path, default=Path("runs/deep_cfr_snapshot.pt"))
    parser.add_argument("--policy-out", type=Path, default=Path("runs/deep_cfr_average_policy.pt"))
    parser.add_argument("--summary-out", type=Path, default=Path("runs/deep_cfr_summary.json"))
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def run_from_args(args: argparse.Namespace) -> dict[str, object]:
    import torch  # noqa: PLC0415

    from poker_arena.cfr.deep_cfr import DeepCFRConfig, DeepCFRTrainer, GameTreeFeatureEncoder  # noqa: PLC0415
    from poker_arena.cfr.evaluation import exploitability  # noqa: PLC0415
    from poker_arena.cfr.holdem_deep_cfr import HoldemCFRState, HoldemDeepCFRFeatureEncoder  # noqa: PLC0415
    from poker_arena.cfr.toy_games import KuhnPokerState, LeducPokerState  # noqa: PLC0415
    from poker_arena.table import TableConfig  # noqa: PLC0415

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    if args.game == "kuhn":
        root = KuhnPokerState.initial()
        encoder = GameTreeFeatureEncoder(root)
    elif args.game == "leduc":
        root = LeducPokerState.initial()
        encoder = GameTreeFeatureEncoder(root)
    else:
        if not 3 <= args.seats <= 9:
            raise ValueError("Hold'em Deep CFR training requires --seats between 3 and 9")
        root = HoldemCFRState.initial(
            TableConfig(
                args.seats,
                args.small_blind,
                args.big_blind,
                [args.starting_stack] * args.seats,
            )
        )
        encoder = HoldemDeepCFRFeatureEncoder(args.max_history_actions)

    if args.resume is not None:
        trainer = DeepCFRTrainer.load_snapshot(root, args.resume, device=device, encoder=encoder)
    else:
        config = DeepCFRConfig(
            iterations=args.iterations,
            traversals_per_player=args.traversals_per_player,
            advantage_capacity=args.advantage_capacity,
            strategy_capacity=args.strategy_capacity,
            hidden=parse_hidden(args.hidden),
            advantage_train_steps=args.advantage_train_steps,
            strategy_train_steps=args.strategy_train_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            random_seed=args.random_seed,
            device=device,
        )
        trainer = DeepCFRTrainer(root, config, encoder)

    starting_iteration = trainer.completed_iterations

    def progress(done: int, target: int, traversals: int) -> None:
        relative = done - starting_iteration
        if args.log_every > 0 and (done == target or relative % args.log_every == 0):
            print(f"deep cfr: iteration {done}/{target}, {traversals} traversals", file=sys.stderr)

    stats = trainer.train(iterations=args.iterations, progress_callback=progress)
    trainer.save_snapshot(args.snapshot_out)
    trainer.save_inference_policy(args.policy_out)
    summary: dict[str, object] = {
        "game": args.game,
        "device": device,
        "snapshot_out": str(args.snapshot_out),
        "policy_out": str(args.policy_out),
        **stats,
    }
    if args.game in {"kuhn", "leduc"}:
        result = exploitability(root, trainer.average_strategy_profile())
        summary["expected_values"] = list(result.expected_values)
        summary["best_response_values"] = list(result.best_response_values)
        summary["nash_conv"] = result.nash_conv
        summary["exploitability"] = result.exploitability
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_from_args(args), indent=2))


if __name__ == "__main__":
    main()
