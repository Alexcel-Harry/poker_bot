from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from poker_arena import Action, Table, TableConfig  # noqa: E402


def run_demo(hands: int = 3, seed: int = 7) -> list[dict[str, object]]:
    table = Table(TableConfig(seats=3, small_blind=5, big_blind=10, starting_stacks=[200, 200, 200], seed=seed))
    histories: list[dict[str, object]] = []
    for _ in range(hands):
        state = table.start_hand()
        while not state.is_terminal:
            legal = state.legal_actions(state.current_actor)
            if legal.can_check:
                action = Action.check()
            elif legal.can_call:
                action = Action.call()
            else:
                action = Action.fold()
            state = table.apply(action)
        histories.append(
            {
                "events": [event.to_dict() for event in state.events if event.event_type != "snapshot"],
                "stacks": list(table.stacks),
                "carryover_chips": table.carryover_chips,
            }
        )
    return histories


def main() -> None:
    print(json.dumps(run_demo(), indent=2))


if __name__ == "__main__":
    main()
