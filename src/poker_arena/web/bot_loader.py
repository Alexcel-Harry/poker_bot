from __future__ import annotations

import os
from collections.abc import Callable, Mapping

from poker_arena.bots import BotPolicy


def bot_policy_factory(
    model_path: str | os.PathLike[str] | None,
    device: str = "auto",
) -> Callable[[int], BotPolicy | None] | None:
    if not model_path:
        return None
    checkpoint = os.fspath(model_path)
    policy: BotPolicy | None = None

    def factory(_seat_id: int) -> BotPolicy | None:
        nonlocal policy
        if policy is None:
            from poker_arena.bots import TorchPolicyBot

            policy = TorchPolicyBot.from_checkpoint(checkpoint, device=device)
        return policy

    return factory


def bot_policy_factory_from_env(env: Mapping[str, str] | None = None) -> Callable[[int], BotPolicy | None] | None:
    source = os.environ if env is None else env
    return bot_policy_factory(source.get("POKER_BOT_MODEL"), source.get("POKER_BOT_DEVICE", "auto"))
