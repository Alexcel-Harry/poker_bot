from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from poker_arena.cfr import PrefixBranchCFRTrainer, PrefixBranchTrainingConfig  # noqa: E402
from poker_arena.table import TableConfig  # noqa: E402


def run_training(iterations: int = 20) -> dict[str, object]:
    config = PrefixBranchTrainingConfig(
        branch_width=6,
        branch_depth=4,
        integer_action_budget=10,
        max_actions_per_episode=20,
        random_seed=17,
    )
    trainer = PrefixBranchCFRTrainer(
        table_config=TableConfig(seats=3, small_blind=10, big_blind=20, starting_stacks=[200, 200, 200], seed=101),
        config=config,
    )
    result = trainer.train(iterations=iterations)
    preview = [
        {
            "infoset": sample.infoset_key,
            "action": sample.action.to_dict(),
            "target_utility": sample.target_utility,
        }
        for sample in trainer.training_samples[:3]
    ]
    return {
        "prefix_branch": True,
        "iterations": result.iterations,
        "information_sets": result.information_sets,
        "training_samples": len(trainer.training_samples),
        "sample_preview": preview,
    }


def main() -> None:
    print(json.dumps(run_training(), indent=2))


if __name__ == "__main__":
    main()
