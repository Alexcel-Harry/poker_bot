from __future__ import annotations

import os
from collections.abc import Callable, Mapping

from poker_arena.bots import BotPolicy


def bot_policy_factory_from_env(env: Mapping[str, str] | None = None) -> Callable[[int], BotPolicy | None] | None:
    source = os.environ if env is None else env
    model_path = source.get("POKER_BOT_MODEL")
    if not model_path:
        return None
    device = source.get("POKER_BOT_DEVICE", "auto")
    policy: BotPolicy | None = None

    def factory(_seat_id: int) -> BotPolicy | None:
        nonlocal policy
        if policy is None:
            from poker_arena.bots import TorchPolicyBot

            policy = TorchPolicyBot.from_checkpoint(model_path, device=device)
        return policy

    return factory
