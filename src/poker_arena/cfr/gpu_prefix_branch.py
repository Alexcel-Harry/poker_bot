from __future__ import annotations

from dataclasses import dataclass, fields
from itertools import combinations
import math
from typing import Any, Callable, Sequence

import torch

from poker_arena.cfr.prefix_branch import ActionEmbedding
from poker_arena.cfr.torch_model import ActionValueNet, StateFeatureEncoder
from poker_arena.table import TableConfig


FOLD = 0
CHECK = 1
CALL = 2
RAISE_TO = 3
INVALID_ACTION = -1


@dataclass(frozen=True)
class GpuPrefixBranchTrainingConfig:
    """Configuration for the fully tensorized CUDA training pipeline."""

    branch_width: int = 32
    parallel_hands: int = 1024
    max_decisions_per_hand: int = 64
    max_rollout_actions: int = 128
    replay_capacity: int = 250_000
    replay_warmup: int = 8192
    batch_size: int = 8192
    optimizer_steps_per_decision: int = 2
    final_epochs: int = 3
    learning_rate: float = 5e-4
    epsilon: float = 0.15
    required_integer_actions: tuple[int, ...] = ()
    random_seed: int = 17
    use_amp: bool = True
    evaluator_chunk_size: int = 32_768

    def __post_init__(self) -> None:
        positive = (
            "branch_width",
            "parallel_hands",
            "max_decisions_per_hand",
            "max_rollout_actions",
            "replay_capacity",
            "replay_warmup",
            "batch_size",
            "evaluator_chunk_size",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.branch_width < 3:
            raise ValueError("branch_width must be at least 3 so fold/check, call, and raise can be represented")
        if self.optimizer_steps_per_decision < 0:
            raise ValueError("optimizer_steps_per_decision must be non-negative")
        if self.final_epochs < 0:
            raise ValueError("final_epochs must be non-negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")


@dataclass(frozen=True)
class TensorLegalActions:
    actor: torch.Tensor
    actor_commitment: torch.Tensor
    actor_stack: torch.Tensor
    call_amount: torch.Tensor
    min_raise_to: torch.Tensor
    max_raise_to: torch.Tensor
    can_raise: torch.Tensor
    facing_bet: torch.Tensor


@dataclass
class TensorPokerState:
    """A batch of complete Hold'em states stored entirely as device tensors."""

    stacks: torch.Tensor
    committed: torch.Tensor
    total_committed: torch.Tensor
    folded: torch.Tensor
    all_in: torch.Tensor
    acted: torch.Tensor
    button: torch.Tensor
    street: torch.Tensor
    board: torch.Tensor
    board_count: torch.Tensor
    deck: torch.Tensor
    next_card: torch.Tensor
    hole_cards: torch.Tensor
    current_actor: torch.Tensor
    current_bet: torch.Tensor
    last_full_raise: torch.Tensor
    terminal: torch.Tensor
    trajectory_sum: torch.Tensor
    trajectory_count: torch.Tensor
    starting_stacks: torch.Tensor
    small_blind: int
    big_blind: int
    evaluator_chunk_size: int

    @property
    def batch_size(self) -> int:
        return int(self.stacks.shape[0])

    @property
    def seats(self) -> int:
        return int(self.stacks.shape[1])

    @property
    def device(self) -> torch.device:
        return self.stacks.device

    @classmethod
    def new_batch(
        cls,
        config: TableConfig,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator,
        evaluator_chunk_size: int = 32_768,
    ) -> TensorPokerState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if min(config.starting_stacks) <= config.big_blind:
            raise ValueError("GPU training requires every starting stack to exceed the big blind")

        seats = config.seats
        rows = torch.arange(batch_size, device=device)
        starting = torch.tensor(config.starting_stacks, dtype=torch.int64, device=device)
        stacks = starting.unsqueeze(0).expand(batch_size, -1).clone()
        button = torch.randint(0, seats, (batch_size,), device=device, generator=generator)
        if seats == 2:
            small_blind_seat = button
        else:
            small_blind_seat = (button + 1) % seats
        big_blind_seat = (small_blind_seat + 1) % seats

        # argsort of independent random keys is a parallel GPU deck shuffle.
        deck = torch.rand((batch_size, 52), device=device, generator=generator).argsort(dim=1)
        hole_cards = torch.empty((batch_size, seats, 2), dtype=torch.int64, device=device)
        for seat in range(seats):
            hole_cards[:, seat, 0] = deck[:, seat]
            hole_cards[:, seat, 1] = deck[:, seats + seat]

        committed = torch.zeros((batch_size, seats), dtype=torch.int64, device=device)
        total_committed = torch.zeros_like(committed)
        small_paid = torch.minimum(stacks[rows, small_blind_seat], torch.full((batch_size,), config.small_blind, dtype=torch.int64, device=device))
        stacks[rows, small_blind_seat] -= small_paid
        committed[rows, small_blind_seat] += small_paid
        total_committed[rows, small_blind_seat] += small_paid
        big_paid = torch.minimum(stacks[rows, big_blind_seat], torch.full((batch_size,), config.big_blind, dtype=torch.int64, device=device))
        stacks[rows, big_blind_seat] -= big_paid
        committed[rows, big_blind_seat] += big_paid
        total_committed[rows, big_blind_seat] += big_paid

        all_in = stacks == 0
        current_bet = committed.max(dim=1).values
        first_actor = small_blind_seat if seats == 2 else (big_blind_seat + 1) % seats

        trajectory_sum = torch.zeros((batch_size, 14), dtype=torch.float32, device=device)
        trajectory_count = torch.ones((batch_size,), dtype=torch.int64, device=device)
        trajectory_sum[:, 9] = 1.0  # hand_started, preflop
        for blind_seat, paid, pot_before in (
            (small_blind_seat, small_paid, torch.zeros_like(small_paid)),
            (big_blind_seat, big_paid, small_paid),
        ):
            # EventContextBuilder uses its 2,000-chip default until the first
            # engine snapshot, so matching that convention keeps deployment
            # and tensor-training trajectory features identical.
            actor_stack_before = torch.full_like(paid, 2000)
            vector = torch.zeros_like(trajectory_sum)
            vector[:, 8] = paid.float() / (pot_before + actor_stack_before).clamp_min(1).float()
            vector[:, 9] = 1.0
            trajectory_sum += vector
            trajectory_count += 1

        return cls(
            stacks=stacks,
            committed=committed,
            total_committed=total_committed,
            folded=torch.zeros((batch_size, seats), dtype=torch.bool, device=device),
            all_in=all_in,
            acted=torch.zeros((batch_size, seats), dtype=torch.bool, device=device),
            button=button,
            street=torch.zeros((batch_size,), dtype=torch.int64, device=device),
            board=torch.full((batch_size, 5), -1, dtype=torch.int64, device=device),
            board_count=torch.zeros((batch_size,), dtype=torch.int64, device=device),
            deck=deck,
            next_card=torch.full((batch_size,), 2 * seats, dtype=torch.int64, device=device),
            hole_cards=hole_cards,
            current_actor=first_actor,
            current_bet=current_bet,
            last_full_raise=torch.full((batch_size,), config.big_blind, dtype=torch.int64, device=device),
            terminal=torch.zeros((batch_size,), dtype=torch.bool, device=device),
            trajectory_sum=trajectory_sum,
            trajectory_count=trajectory_count,
            starting_stacks=starting,
            small_blind=config.small_blind,
            big_blind=config.big_blind,
            evaluator_chunk_size=evaluator_chunk_size,
        )

    def clone(self) -> TensorPokerState:
        values: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.clone() if isinstance(value, torch.Tensor) else value
        return TensorPokerState(**values)

    def index_select(self, indices: torch.Tensor) -> TensorPokerState:
        values: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if (
                item.name != "starting_stacks"
                and isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == self.batch_size
            ):
                values[item.name] = value.index_select(0, indices)
            else:
                values[item.name] = value
        return TensorPokerState(**values)

    def repeat_interleave(self, repeats: int) -> TensorPokerState:
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        indices = torch.arange(self.batch_size, device=self.device).repeat_interleave(repeats)
        return self.index_select(indices)

    def pot(self) -> torch.Tensor:
        return self.total_committed.sum(dim=1)

    def legal_actions(self) -> TensorLegalActions:
        rows = torch.arange(self.batch_size, device=self.device)
        actor = self.current_actor.clamp_min(0)
        actor_commitment = self.committed[rows, actor]
        actor_stack = self.stacks[rows, actor]
        outstanding = (self.current_bet - actor_commitment).clamp_min(0)
        call_amount = torch.minimum(outstanding, actor_stack)
        max_raise_to = actor_commitment + actor_stack
        can_raise = (~self.terminal) & (max_raise_to > self.current_bet)
        full_minimum = torch.where(
            self.current_bet == 0,
            torch.full_like(self.current_bet, self.big_blind),
            self.current_bet + self.last_full_raise,
        )
        min_raise_to = torch.minimum(full_minimum, max_raise_to)
        return TensorLegalActions(
            actor=actor,
            actor_commitment=actor_commitment,
            actor_stack=actor_stack,
            call_amount=call_amount,
            min_raise_to=min_raise_to,
            max_raise_to=max_raise_to,
            can_raise=can_raise,
            facing_bet=outstanding > 0,
        )

    def candidate_actions(
        self,
        width: int,
        generator: torch.Generator,
        required_amounts: Sequence[int] = (),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TensorLegalActions]:
        legal = self.legal_actions()
        batch = self.batch_size
        action_types = torch.full((batch, width), INVALID_ACTION, dtype=torch.int64, device=self.device)
        totals = torch.zeros((batch, width), dtype=torch.int64, device=self.device)
        valid = torch.zeros((batch, width), dtype=torch.bool, device=self.device)

        facing = legal.facing_bet
        action_types[:, 0] = FOLD
        valid[:, 0] = facing & ~self.terminal
        if width > 1:
            action_types[:, 1] = torch.where(facing, torch.full_like(legal.actor, CALL), torch.full_like(legal.actor, CHECK))
            valid[:, 1] = ~self.terminal

        start = torch.full_like(legal.actor, 2)
        random_values = torch.rand((batch, width), device=self.device, generator=generator)
        span = (legal.max_raise_to - legal.min_raise_to + 1).clamp_min(1)
        sampled_totals = legal.min_raise_to[:, None] + torch.floor(random_values * span[:, None].float()).to(torch.int64)
        required = tuple(dict.fromkeys(int(amount) for amount in required_amounts))

        for column in range(width):
            local = column - start
            is_raise_slot = (local >= 0) & legal.can_raise & ~self.terminal
            proposed = sampled_totals[:, column]
            proposed = torch.where(local == 0, legal.min_raise_to, proposed)
            proposed = torch.where(local == 1, legal.max_raise_to, proposed)
            for required_index, amount in enumerate(required):
                required_mask = local == required_index + 2
                amount_tensor = torch.full_like(proposed, amount)
                amount_is_legal = (amount_tensor >= legal.min_raise_to) & (amount_tensor <= legal.max_raise_to)
                proposed = torch.where(required_mask & amount_is_legal, amount_tensor, proposed)
            action_types[:, column] = torch.where(is_raise_slot, torch.full_like(legal.actor, RAISE_TO), action_types[:, column])
            totals[:, column] = torch.where(is_raise_slot, proposed, totals[:, column])
            valid[:, column] |= is_raise_slot

        raise_mask = valid & (action_types == RAISE_TO)
        same_total = totals[:, :, None] == totals[:, None, :]
        earlier = torch.tril(torch.ones((width, width), dtype=torch.bool, device=self.device), diagonal=-1)
        duplicate = (same_total & raise_mask[:, :, None] & raise_mask[:, None, :] & earlier[None, :, :]).any(dim=2)
        valid &= ~duplicate
        return action_types, totals, valid, legal

    def apply_actions(
        self,
        action_types: torch.Tensor,
        totals: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        if mask is None:
            mask = ~self.terminal
        rows = torch.nonzero(mask & ~self.terminal, as_tuple=False).squeeze(1)
        if rows.numel() == 0:
            return
        actors = self.current_actor[rows]
        selected_types = action_types[rows]
        selected_totals = totals[rows]
        row_stacks_before = self.stacks[rows, actors].clone()
        row_pot_before = self.pot()[rows]
        row_street_before = self.street[rows]
        row_commitment = self.committed[rows, actors]
        row_current_bet = self.current_bet[rows].clone()
        row_last_raise = self.last_full_raise[rows].clone()

        event = torch.zeros((rows.numel(), 14), dtype=torch.float32, device=self.device)
        event.scatter_(1, selected_types[:, None], 1.0)
        is_raise = selected_types == RAISE_TO
        raise_total_float = selected_totals.float()
        event[:, 4] = torch.where(is_raise, raise_total_float / (row_pot_before + row_stacks_before).clamp_min(1).float(), torch.zeros_like(raise_total_float))
        event[:, 5] = torch.where(is_raise, raise_total_float / row_stacks_before.clamp_min(1).float(), torch.zeros_like(raise_total_float))
        event[:, 6] = torch.where(is_raise & (row_pot_before > 0), raise_total_float / row_pot_before.clamp_min(1).float(), torch.zeros_like(raise_total_float))
        event[:, 7] = actors.float() / 8.0
        event.scatter_(1, (9 + row_street_before)[:, None], 1.0)
        self.trajectory_sum[rows] += event
        self.trajectory_count[rows] += 1

        fold_rows = rows[selected_types == FOLD]
        fold_actors = actors[selected_types == FOLD]
        self.folded[fold_rows, fold_actors] = True
        self.acted[fold_rows, fold_actors] = True

        check_rows = rows[selected_types == CHECK]
        check_actors = actors[selected_types == CHECK]
        self.acted[check_rows, check_actors] = True

        call_selector = selected_types == CALL
        call_rows = rows[call_selector]
        call_actors = actors[call_selector]
        call_paid = torch.minimum(
            (row_current_bet[call_selector] - row_commitment[call_selector]).clamp_min(0),
            row_stacks_before[call_selector],
        )
        self._commit(call_rows, call_actors, call_paid)
        self.acted[call_rows, call_actors] = True

        raise_selector = selected_types == RAISE_TO
        raise_rows = rows[raise_selector]
        raise_actors = actors[raise_selector]
        raise_totals = selected_totals[raise_selector]
        raise_paid = raise_totals - row_commitment[raise_selector]
        self._commit(raise_rows, raise_actors, raise_paid)
        raise_size = raise_totals - row_current_bet[raise_selector]
        full_raise = raise_size >= row_last_raise[raise_selector]
        self.current_bet[raise_rows] = raise_totals
        self.last_full_raise[raise_rows] = torch.where(full_raise, raise_size, row_last_raise[raise_selector])
        full_rows = raise_rows[full_raise]
        self.acted[full_rows] = False
        self.acted[raise_rows, raise_actors] = True

        self._after_action(rows, actors)

    def _commit(self, rows: torch.Tensor, actors: torch.Tensor, amount: torch.Tensor) -> None:
        if rows.numel() == 0:
            return
        paid = torch.minimum(amount.clamp_min(0), self.stacks[rows, actors])
        self.stacks[rows, actors] -= paid
        self.committed[rows, actors] += paid
        self.total_committed[rows, actors] += paid
        self.all_in[rows, actors] = self.stacks[rows, actors] == 0

    def _after_action(self, rows: torch.Tensor, previous_actors: torch.Tensor) -> None:
        live = ~self.folded
        live_count = live.sum(dim=1)
        fold_win = torch.zeros_like(self.terminal)
        fold_win[rows] = live_count[rows] == 1
        self._award_fold_wins(fold_win)

        remaining = torch.zeros_like(self.terminal)
        remaining[rows] = ~self.terminal[rows]
        active = live & ~self.all_in
        no_active = remaining & (active.sum(dim=1) == 0)
        self._showdown(no_active)

        remaining &= ~self.terminal
        matched = (self.committed == self.current_bet[:, None]) & self.acted
        round_complete = remaining & ((~active) | matched).all(dim=1)
        uncontested_betting = round_complete & (active.sum(dim=1) < 2)
        self._showdown(uncontested_betting)

        river_complete = round_complete & (self.street == 3)
        self._showdown(river_complete)

        advance = round_complete & (self.street < 3) & ~self.terminal
        self._advance_street(advance)
        self.current_actor[advance] = self._next_actor((self.button[advance] + 1) % self.seats, advance)

        continue_round = remaining & ~round_complete & ~self.terminal
        # previous_actors is aligned to rows; use a dense lookup to avoid host-side maps.
        dense_previous = torch.zeros((self.batch_size,), dtype=torch.int64, device=self.device)
        dense_previous[rows] = previous_actors
        starts = (dense_previous[continue_round] + 1) % self.seats
        self.current_actor[continue_round] = self._next_actor(starts, continue_round)

    def _next_actor(self, starts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        live_active = ~self.folded[mask] & ~self.all_in[mask]
        needs = live_active & ((self.committed[mask] != self.current_bet[mask, None]) | ~self.acted[mask])
        offsets = torch.arange(self.seats, device=self.device)
        candidates = (starts[:, None] + offsets[None, :]) % self.seats
        candidate_needs = needs.gather(1, candidates)
        first = candidate_needs.to(torch.int64).argmax(dim=1)
        return candidates.gather(1, first[:, None]).squeeze(1)

    def _advance_street(self, mask: torch.Tensor) -> None:
        rows = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if rows.numel() == 0:
            return
        old_street = self.street[rows]
        target_count = torch.where(old_street == 0, torch.full_like(old_street, 3), old_street + 3)
        self._deal_to_count(mask, target_count)
        self.street[rows] = old_street + 1
        self.current_bet[rows] = 0
        self.last_full_raise[rows] = self.big_blind
        self.committed[rows] = 0
        self.acted[rows] = False
        event = torch.zeros((rows.numel(), 14), dtype=torch.float32, device=self.device)
        event.scatter_(1, (9 + self.street[rows])[:, None], 1.0)
        self.trajectory_sum[rows] += event
        self.trajectory_count[rows] += 1

    def _deal_to_count(self, mask: torch.Tensor, target_count: torch.Tensor | int) -> None:
        rows = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if rows.numel() == 0:
            return
        if isinstance(target_count, int):
            targets = torch.full((rows.numel(),), target_count, dtype=torch.int64, device=self.device)
        else:
            targets = target_count
        for position in range(5):
            needs_card = (self.board_count[rows] <= position) & (targets > position)
            deal_rows = rows[needs_card]
            if deal_rows.numel() == 0:
                continue
            cards = self.deck[deal_rows, self.next_card[deal_rows]]
            self.board[deal_rows, position] = cards
            self.next_card[deal_rows] += 1
            self.board_count[deal_rows] += 1

    def _award_fold_wins(self, mask: torch.Tensor) -> None:
        rows = torch.nonzero(mask & ~self.terminal, as_tuple=False).squeeze(1)
        if rows.numel() == 0:
            return
        winners = (~self.folded[rows]).to(torch.int64).argmax(dim=1)
        self.stacks[rows, winners] += self.pot()[rows]
        self.terminal[rows] = True
        self.current_actor[rows] = -1

    def _showdown(self, mask: torch.Tensor) -> None:
        rows = torch.nonzero(mask & ~self.terminal, as_tuple=False).squeeze(1)
        if rows.numel() == 0:
            return
        dense_mask = torch.zeros_like(self.terminal)
        dense_mask[rows] = True
        self._deal_to_count(dense_mask, 5)
        seven = torch.cat((self.hole_cards[rows], self.board[rows, None, :].expand(-1, self.seats, -1)), dim=2)
        values = evaluate_seven_card_hands(seven, chunk_size=self.evaluator_chunk_size)
        values = values.masked_fill(self.folded[rows], -1)
        contributions = self.total_committed[rows]
        levels = contributions.sort(dim=1).values
        previous = torch.cat((torch.zeros((rows.numel(), 1), dtype=torch.int64, device=self.device), levels[:, :-1]), dim=1)
        increments = levels - previous
        contributors = contributions[:, None, :] >= levels[:, :, None]
        pot_amounts = increments * contributors.sum(dim=2)
        eligible = contributors & ~self.folded[rows, None, :]
        eligible_values = values[:, None, :].masked_fill(~eligible, -1)
        best = eligible_values.max(dim=2).values
        winners = eligible & (values[:, None, :] == best[:, :, None]) & (pot_amounts[:, :, None] > 0)
        winner_count = winners.sum(dim=2).clamp_min(1)
        shares = torch.div(pot_amounts, winner_count, rounding_mode="floor")
        awards = winners.to(torch.int64) * shares[:, :, None]
        remainder = pot_amounts - shares * winner_count
        first_winner = winners.to(torch.int64).argmax(dim=2)
        awards.scatter_add_(2, first_winner[:, :, None], remainder[:, :, None])
        self.stacks[rows] += awards.sum(dim=1)
        self.terminal[rows] = True
        self.current_actor[rows] = -1
        self.street[rows] = 4

    def rollout(self, max_actions: int, generator: torch.Generator) -> None:
        for step in range(max_actions):
            active = ~self.terminal
            legal = self.legal_actions()
            random_choice = torch.rand((self.batch_size,), device=self.device, generator=generator)
            has_raise = legal.can_raise
            facing_count = 2 + has_raise.to(torch.int64)
            checking_count = 1 + has_raise.to(torch.int64)
            choice_count = torch.where(legal.facing_bet, facing_count, checking_count)
            choice = torch.floor(random_choice * choice_count.float()).to(torch.int64)
            action_types = torch.where(
                legal.facing_bet,
                torch.where(choice == 0, torch.full_like(choice, FOLD), torch.where(choice == 1, torch.full_like(choice, CALL), torch.full_like(choice, RAISE_TO))),
                torch.where(choice == 0, torch.full_like(choice, CHECK), torch.full_like(choice, RAISE_TO)),
            )
            span = (legal.max_raise_to - legal.min_raise_to + 1).clamp_min(1)
            raise_fraction = torch.rand((self.batch_size,), device=self.device, generator=generator)
            totals = legal.min_raise_to + torch.floor(raise_fraction * span.float()).to(torch.int64)
            self.apply_actions(action_types, totals, active)
            if step % 8 == 7 and not bool((~self.terminal).any().item()):
                break

    def utilities(self) -> torch.Tensor:
        result = self.stacks.float() - self.starting_stacks[None, :].float()
        unfinished = ~self.terminal
        eligible = ~self.folded
        share = self.pot().float() / eligible.sum(dim=1).clamp_min(1).float()
        result += unfinished[:, None] * eligible * share[:, None]
        return result

    def state_features(self) -> torch.Tensor:
        legal = self.legal_actions()
        rows = torch.arange(self.batch_size, device=self.device)
        actor = legal.actor
        pot = self.pot().float()
        actor_stack = legal.actor_stack.float()
        denom = torch.maximum((pot + self.stacks.sum(dim=1).float()).clamp_min(1), (pot + actor_stack).clamp_min(1))
        features = torch.zeros((self.batch_size, StateFeatureEncoder.dimension), dtype=torch.float32, device=self.device)
        features[:, 0] = self.seats / StateFeatureEncoder.max_seats
        features[:, 1] = actor.float() / (StateFeatureEncoder.max_seats - 1)
        features[:, 2] = self.button.float() / (StateFeatureEncoder.max_seats - 1)
        features[:, 3] = pot / denom
        features[:, 4] = self.current_bet.float() / (actor_stack + self.current_bet.float()).clamp_min(1)
        features[:, 5] = actor_stack / denom
        features[:, 6] = self.board_count.float() / 5.0
        features.scatter_(1, (7 + self.street.clamp(0, 4))[:, None], 1.0)
        features[:, 12 : 12 + self.seats] = self.stacks.float() / denom[:, None]
        features[:, 21 : 21 + self.seats] = self.folded.float()
        features[:, 30] = self.all_in.sum(dim=1).float() / self.seats

        actor_cards = self.hole_cards[rows, actor]
        hole_one_hot = torch.zeros((self.batch_size, 52), dtype=torch.float32, device=self.device)
        hole_one_hot.scatter_(1, actor_cards, 1.0)
        board_one_hot = torch.zeros_like(hole_one_hot)
        valid_board = self.board >= 0
        board_one_hot.scatter_add_(1, self.board.clamp_min(0), valid_board.float())
        features[:, 31:83] = hole_one_hot
        features[:, 83:135] = board_one_hot.clamp_max(1)
        return features

    def trajectory_features(self) -> torch.Tensor:
        return self.trajectory_sum / self.trajectory_count.clamp_min(1).float()[:, None]

    def action_features(
        self,
        action_types: torch.Tensor,
        totals: torch.Tensor,
        legal: TensorLegalActions,
    ) -> torch.Tensor:
        batch, width = action_types.shape
        features = torch.zeros((batch, width, ActionEmbedding.dimension_without_trajectory), dtype=torch.float32, device=self.device)
        features.scatter_(2, action_types.clamp_min(0)[:, :, None], 1.0)
        min_raise = torch.where(legal.can_raise, legal.min_raise_to, self.current_bet)
        max_raise = torch.where(legal.can_raise, legal.max_raise_to, torch.maximum(min_raise, self.current_bet))
        span = (max_raise - min_raise).clamp_min(1).float()
        concrete_total = torch.where(action_types == RAISE_TO, totals, self.current_bet[:, None])
        added = (concrete_total - self.current_bet[:, None]).clamp_min(0)
        stack_before = (
            legal.actor_commitment + (max_raise - legal.actor_commitment).clamp_min(0)
        ).clamp_min(1).float()
        features[:, :, 4] = concrete_total.float() / max_raise.clamp_min(1).float()[:, None]
        features[:, :, 5] = (concrete_total - min_raise[:, None]).float() / span[:, None]
        features[:, :, 6] = added.float() / span[:, None]
        features[:, :, 7] = legal.call_amount.float()[:, None] / stack_before[:, None]
        features[:, :, 8] = self.current_bet.float()[:, None] / stack_before[:, None]
        features[:, :, 9] = legal.actor_commitment.float()[:, None] / stack_before[:, None]
        features[:, :, 10] = min_raise.float()[:, None] / stack_before[:, None]
        features[:, :, 11] = max_raise.float()[:, None] / stack_before[:, None]
        return features


_FIVE_CARD_COMBINATIONS = tuple(combinations(range(7), 5))


def evaluate_seven_card_hands(cards: torch.Tensor, chunk_size: int = 32_768) -> torch.Tensor:
    """Return sortable exact hand values for ``[..., 7]`` card-id tensors."""

    if cards.shape[-1] != 7:
        raise ValueError("evaluate_seven_card_hands requires seven cards per hand")
    leading = cards.shape[:-1]
    flat = cards.reshape(-1, 7)
    combo_indices = torch.tensor(_FIVE_CARD_COMBINATIONS, dtype=torch.int64, device=cards.device)
    outputs: list[torch.Tensor] = []
    for start in range(0, flat.shape[0], chunk_size):
        chunk = flat[start : start + chunk_size]
        five = chunk[:, combo_indices]
        outputs.append(_evaluate_five_card_combinations(five).max(dim=1).values)
    return torch.cat(outputs).reshape(leading)


def _evaluate_five_card_combinations(cards: torch.Tensor) -> torch.Tensor:
    ranks = cards.remainder(13) + 2
    suits = torch.div(cards, 13, rounding_mode="floor")
    ranks = ranks.sort(dim=-1, descending=True).values
    flush = (suits == suits[..., :1]).all(dim=-1)
    normal_straight = ((ranks[..., :-1] - ranks[..., 1:]) == 1).all(dim=-1)
    wheel = (ranks == torch.tensor([14, 5, 4, 3, 2], dtype=torch.int64, device=cards.device)).all(dim=-1)
    straight_high = torch.where(normal_straight, ranks[..., 0], torch.where(wheel, torch.full_like(ranks[..., 0], 5), torch.zeros_like(ranks[..., 0])))

    rank_values = torch.arange(15, device=cards.device)
    counts = (ranks[..., :, None] == rank_values).sum(dim=-2)
    desc_ranks = torch.arange(14, -1, -1, device=cards.device)
    desc_counts = counts.flip(-1)
    singles = torch.where(desc_counts == 1, desc_ranks, torch.zeros_like(desc_ranks)).sort(dim=-1, descending=True).values
    pairs = torch.where(desc_counts == 2, desc_ranks, torch.zeros_like(desc_ranks)).sort(dim=-1, descending=True).values
    trips = torch.where(desc_counts == 3, desc_ranks, torch.zeros_like(desc_ranks)).max(dim=-1).values
    quads = torch.where(desc_counts == 4, desc_ranks, torch.zeros_like(desc_ranks)).max(dim=-1).values
    pair_count = (counts == 2).sum(dim=-1)
    has_trip = trips > 0
    has_quad = quads > 0
    is_straight = straight_high > 0

    category = torch.zeros_like(straight_high)
    category = torch.where(pair_count == 1, torch.ones_like(category), category)
    category = torch.where(pair_count >= 2, torch.full_like(category, 2), category)
    category = torch.where(has_trip, torch.full_like(category, 3), category)
    category = torch.where(is_straight, torch.full_like(category, 4), category)
    category = torch.where(flush, torch.full_like(category, 5), category)
    category = torch.where(has_trip & (pair_count >= 1), torch.full_like(category, 6), category)
    category = torch.where(has_quad, torch.full_like(category, 7), category)
    category = torch.where(flush & is_straight, torch.full_like(category, 8), category)

    tiebreak = ranks.clone()
    tiebreak = torch.where((category == 1)[..., None], torch.stack((pairs[..., 0], singles[..., 0], singles[..., 1], singles[..., 2], torch.zeros_like(category)), dim=-1), tiebreak)
    tiebreak = torch.where((category == 2)[..., None], torch.stack((pairs[..., 0], pairs[..., 1], singles[..., 0], torch.zeros_like(category), torch.zeros_like(category)), dim=-1), tiebreak)
    tiebreak = torch.where((category == 3)[..., None], torch.stack((trips, singles[..., 0], singles[..., 1], torch.zeros_like(category), torch.zeros_like(category)), dim=-1), tiebreak)
    straight_break = torch.stack((straight_high, torch.zeros_like(category), torch.zeros_like(category), torch.zeros_like(category), torch.zeros_like(category)), dim=-1)
    tiebreak = torch.where(((category == 4) | (category == 8))[..., None], straight_break, tiebreak)
    full_house_break = torch.stack((trips, pairs[..., 0], torch.zeros_like(category), torch.zeros_like(category), torch.zeros_like(category)), dim=-1)
    tiebreak = torch.where((category == 6)[..., None], full_house_break, tiebreak)
    quad_break = torch.stack((quads, singles[..., 0], torch.zeros_like(category), torch.zeros_like(category), torch.zeros_like(category)), dim=-1)
    tiebreak = torch.where((category == 7)[..., None], quad_break, tiebreak)
    powers = torch.tensor([15**4, 15**3, 15**2, 15, 1], dtype=torch.int64, device=cards.device)
    return category * (15**5) + (tiebreak * powers).sum(dim=-1)


class CudaReplayBuffer:
    """Fixed-size replay tensors that never leave the CUDA device."""

    def __init__(self, capacity: int, input_dim: int, device: torch.device) -> None:
        self.capacity = capacity
        self.features = torch.empty((capacity, input_dim), dtype=torch.float32, device=device)
        self.targets = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        self.size = 0
        self.cursor = 0
        self.samples_seen = 0

    def add(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        features = features.detach()
        targets = targets.detach().reshape(-1, 1)
        if features.shape[0] != targets.shape[0]:
            raise ValueError("features and targets must contain the same number of rows")
        if features.shape[0] > self.capacity:
            features = features[-self.capacity :]
            targets = targets[-self.capacity :]
        count = int(features.shape[0])
        first = min(count, self.capacity - self.cursor)
        self.features[self.cursor : self.cursor + first].copy_(features[:first])
        self.targets[self.cursor : self.cursor + first].copy_(targets[:first])
        remainder = count - first
        if remainder:
            self.features[:remainder].copy_(features[first:])
            self.targets[:remainder].copy_(targets[first:])
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.capacity, self.size + count)
        self.samples_seen += count

    def sample(self, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        if self.size == 0:
            raise ValueError("Cannot sample an empty replay buffer")
        indices = torch.randint(0, self.size, (min(batch_size, self.size),), device=self.features.device, generator=generator)
        return self.features[indices], self.targets[indices]


class GpuPrefixBranchTrainer:
    """Batched prefix branching, rollout, replay, and learning on one CUDA GPU."""

    def __init__(
        self,
        table_config: TableConfig,
        config: GpuPrefixBranchTrainingConfig,
        model: ActionValueNet,
        device: torch.device,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("GpuPrefixBranchTrainer requires a CUDA device; CPU fallback is intentionally disabled")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.table_config = table_config
        self.config = config
        self.device = device
        self.model = model.to(device)
        self.model.train()
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        torch.set_float32_matmul_precision("high")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.amp_enabled = bool(config.use_amp)
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled and self.amp_dtype == torch.float16)
        input_dim = StateFeatureEncoder.dimension + 14 + ActionEmbedding.dimension_without_trajectory
        self.replay = CudaReplayBuffer(config.replay_capacity, input_dim, device)
        self.loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        self.optimizer_updates = 0
        self.decision_checkpoints = 0

    def train(
        self,
        iterations: int,
        progress_callback: Callable[[int, int, int, int], None] | None = None,
    ) -> dict[str, Any]:
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        completed = 0
        while completed < iterations:
            batch_hands = min(self.config.parallel_hands, iterations - completed)
            state = TensorPokerState.new_batch(
                self.table_config,
                batch_hands,
                self.device,
                self.generator,
                evaluator_chunk_size=self.config.evaluator_chunk_size,
            )
            self._train_hand_batch(state)
            completed += batch_hands
            if progress_callback is not None:
                progress_callback(completed, iterations, self.replay.samples_seen, self.optimizer_updates)

        final_steps = self.config.final_epochs * max(1, math.ceil(self.replay.size / self.config.batch_size))
        for _ in range(final_steps):
            self._optimizer_step()
        return {
            "iterations": iterations,
            "decision_checkpoints": self.decision_checkpoints,
            "generated_samples": self.replay.samples_seen,
            "training_samples": self.replay.size,
            "optimizer_updates": self.optimizer_updates,
            "loss": [float((self.loss_sum / max(1, self.optimizer_updates)).item())],
            "amp": self.amp_enabled,
            "amp_dtype": str(self.amp_dtype).removeprefix("torch.") if self.amp_enabled else None,
        }

    def _train_hand_batch(self, state: TensorPokerState) -> None:
        width = self.config.branch_width
        for _ in range(self.config.max_decisions_per_hand):
            active_indices = torch.nonzero(~state.terminal, as_tuple=False).squeeze(1)
            if active_indices.numel() == 0:
                break
            state = state.index_select(active_indices)
            actors = state.current_actor.clone()
            action_types, totals, valid, legal = state.candidate_actions(
                width,
                self.generator,
                self.config.required_integer_actions,
            )
            state_features = state.state_features()
            trajectory_features = state.trajectory_features()
            action_features = state.action_features(action_types, totals, legal)
            features = torch.cat(
                (
                    state_features[:, None, :].expand(-1, width, -1),
                    trajectory_features[:, None, :].expand(-1, width, -1),
                    action_features,
                ),
                dim=2,
            )

            branch_states = state.repeat_interleave(width)
            flat_valid = valid.reshape(-1)
            branch_states.terminal[~flat_valid] = True
            branch_states.current_actor[~flat_valid] = -1
            branch_states.apply_actions(action_types.reshape(-1), totals.reshape(-1), flat_valid)
            immediate_states = branch_states.clone()
            branch_states.rollout(self.config.max_rollout_actions, self.generator)
            utilities = branch_states.utilities()
            repeated_actors = actors.repeat_interleave(width)
            rows = torch.arange(branch_states.batch_size, device=self.device)
            targets = utilities[rows, repeated_actors].reshape(state.batch_size, width)
            targets = targets / state.starting_stacks[actors].float()[:, None]

            self.replay.add(features[valid], targets[valid])
            self.decision_checkpoints += state.batch_size
            if self.replay.size >= self.config.replay_warmup:
                for _ in range(self.config.optimizer_steps_per_decision):
                    self._optimizer_step()

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                    policy_scores = self.model(features.reshape(-1, features.shape[-1])).reshape(state.batch_size, width)
                policy_scores = policy_scores.masked_fill(~valid, -torch.inf)
                random_scores = torch.rand(policy_scores.shape, device=self.device, generator=self.generator).masked_fill(~valid, -1.0)
                random_choice = random_scores.argmax(dim=1)
                greedy_choice = policy_scores.argmax(dim=1)
                explore = torch.rand((state.batch_size,), device=self.device, generator=self.generator) < self.config.epsilon
                chosen = torch.where(explore, random_choice, greedy_choice)
            flat_choice = torch.arange(state.batch_size, device=self.device) * width + chosen
            state = immediate_states.index_select(flat_choice)

    def _optimizer_step(self) -> None:
        if self.replay.size == 0:
            return
        features, targets = self.replay.sample(self.config.batch_size, self.generator)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
            prediction = self.model(features)
            loss = torch.nn.functional.smooth_l1_loss(prediction, targets)
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        self.optimizer_updates += 1
        self.loss_sum += loss.detach()
