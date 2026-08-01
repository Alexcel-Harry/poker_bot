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


def parse_hidden(raw: str) -> tuple[int, ...]:
    hidden = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not hidden or any(width <= 0 for width in hidden):
        raise ValueError("Hidden layers must be positive comma-separated integers")
    return hidden


def parse_fractions(raw: str) -> tuple[float, ...]:
    fractions = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not fractions or any(value <= 0 for value in fractions):
        raise ValueError("Pot fractions must be positive comma-separated numbers")
    return fractions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an action-value policy with tensorized CUDA prefix branching and rollouts."
    )
    parser.add_argument("--iterations", type=int, default=10_000, help="Total poker hands to generate.")
    parser.add_argument("--seats", type=int, default=3)
    parser.add_argument("--small-blind", type=int, default=10)
    parser.add_argument("--big-blind", type=int, default=20)
    parser.add_argument("--starting-stack", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--branch-width", type=int, default=8)
    parser.add_argument("--chance-replicas", type=int, default=4)
    parser.add_argument("--pot-fractions", default="0.3333333333,0.75,1.5")
    parser.add_argument("--parallel-hands", type=int, default=1024)
    parser.add_argument("--max-decisions-per-hand", type=int, default=64)
    parser.add_argument("--max-rollout-actions", type=int, default=128)
    parser.add_argument("--required-integer-actions", default="")
    parser.add_argument("--replay-capacity", type=int, default=250_000)
    parser.add_argument("--replay-warmup", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--optimizer-steps-per-decision", type=int, default=2)
    parser.add_argument("--final-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--evaluator-chunk-size", type=int, default=32_768)
    parser.add_argument("--log-every", type=int, default=1024, help="Report after this many completed hands; 0 disables progress output.")
    parser.add_argument("--device", choices=("cuda",), default="cuda", help="CPU fallback is intentionally unavailable.")
    parser.add_argument("--gpus", default="0", help="Exactly one CUDA GPU id; this trainer is single-GPU tensor parallel.")
    parser.add_argument("--hidden", default="512,512,256")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use CUDA automatic mixed precision.")
    parser.add_argument("--model-out", type=Path, default=Path("runs/poker_policy_gpu.pt"))
    parser.add_argument("--resume-snapshot", type=Path, default=None)
    parser.add_argument("--training-snapshot-out", type=Path, default=Path("runs/poker_policy_gpu_training.pt"))
    parser.add_argument("--summary-out", type=Path, default=Path("runs/training_summary_gpu.json"))
    return parser


def _require_single_cuda_gpu(args: argparse.Namespace, env: MutableMapping[str, str]) -> tuple[int, ...]:
    gpu_ids = parse_gpu_ids(args.gpus)
    if len(gpu_ids) != 1:
        raise ValueError("The tensorized trainer requires exactly one CUDA GPU id")
    configure_visible_gpus(gpu_ids, env)
    return gpu_ids


def run_from_args(args: argparse.Namespace, env: MutableMapping[str, str] | None = None) -> dict[str, object]:
    target_env = os.environ if env is None else env
    gpu_ids = _require_single_cuda_gpu(args, target_env)

    # CUDA visibility must be configured before importing torch or CUDA-backed modules.
    import torch  # noqa: PLC0415

    from poker_arena.cfr.gpu_prefix_branch import (  # noqa: PLC0415
        GpuPrefixBranchTrainer,
        GpuPrefixBranchTrainingConfig,
    )
    from poker_arena.cfr.prefix_branch import ActionEmbedding  # noqa: PLC0415
    from poker_arena.cfr.torch_model import (  # noqa: PLC0415
        ActionValueNet,
        StateFeatureEncoder,
        TorchCheckpointMetadata,
        save_checkpoint,
    )
    from poker_arena.table import TableConfig  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is false")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected one visible CUDA device after applying --gpus, found {torch.cuda.device_count()}"
        )

    device = torch.device("cuda:0")
    required_amounts = parse_required_amounts(args.required_integer_actions)
    hidden = parse_hidden(args.hidden)
    config = GpuPrefixBranchTrainingConfig(
        branch_width=args.branch_width,
        chance_replicas=args.chance_replicas,
        pot_fractions=parse_fractions(args.pot_fractions),
        parallel_hands=args.parallel_hands,
        max_decisions_per_hand=args.max_decisions_per_hand,
        max_rollout_actions=args.max_rollout_actions,
        replay_capacity=args.replay_capacity,
        replay_warmup=args.replay_warmup,
        batch_size=args.batch_size,
        optimizer_steps_per_decision=args.optimizer_steps_per_decision,
        final_epochs=args.final_epochs,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        required_integer_actions=required_amounts,
        random_seed=args.random_seed,
        use_amp=args.amp,
        evaluator_chunk_size=args.evaluator_chunk_size,
    )
    table_config = TableConfig(
        seats=args.seats,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        starting_stacks=[args.starting_stack] * args.seats,
    )
    metadata = TorchCheckpointMetadata(
        state_dim=StateFeatureEncoder.dimension,
        trajectory_dim=14,
        action_dim=ActionEmbedding.dimension_without_trajectory,
        hidden=hidden,
        dropout=args.dropout,
        table_defaults={
            "seats": args.seats,
            "small_blind": args.small_blind,
            "big_blind": args.big_blind,
            "starting_stack": args.starting_stack,
        },
        action_sampler={
            "integer_action_budget": args.branch_width,
            "required_integer_actions": list(required_amounts),
            "pot_fractions": list(config.pot_fractions),
        },
        training={
            "algorithm": "tensorized_cuda_prefix_branch_mc",
            "iterations": args.iterations,
            "branch_width": args.branch_width,
            "chance_replicas": args.chance_replicas,
            "pot_fractions": list(config.pot_fractions),
            "parallel_hands": args.parallel_hands,
            "max_rollout_actions": args.max_rollout_actions,
            "replay_capacity": args.replay_capacity,
            "target_unit": "starting_stack_fraction",
            "rollout_device": "cuda",
        },
    )
    model = ActionValueNet(input_dim=metadata.input_dim, hidden=hidden, dropout=args.dropout)
    trainer = GpuPrefixBranchTrainer(table_config, config, model, device)
    if args.resume_snapshot is not None:
        trainer.load_snapshot(args.resume_snapshot)
    last_report = 0

    def progress(done: int, total: int, samples: int, updates: int) -> None:
        nonlocal last_report
        if args.log_every > 0 and (done == total or done - last_report >= args.log_every):
            print(
                f"cuda rollouts: {done}/{total} hands, {samples} samples, {updates} optimizer updates",
                file=sys.stderr,
            )
            last_report = done

    stats = trainer.train(args.iterations, progress_callback=progress)
    metadata.training["iterations"] = stats["iterations"]
    save_checkpoint(args.model_out, trainer.model, metadata, stats)
    trainer.save_snapshot(args.training_snapshot_out)
    summary: dict[str, object] = {
        "prefix_branch": True,
        "algorithm": "tensorized_cuda_prefix_branch_mc",
        "model_out": str(args.model_out),
        "training_snapshot_out": str(args.training_snapshot_out),
        "device": str(device),
        "gpu_ids": list(gpu_ids),
        "rollout_device": "cuda",
        "replay_device": "cuda",
        "training_device": "cuda",
        "parallel_hands": args.parallel_hands,
        "branch_width": args.branch_width,
        "chance_replicas": args.chance_replicas,
        **stats,
    }
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_from_args(args), indent=2))


if __name__ == "__main__":
    main()
