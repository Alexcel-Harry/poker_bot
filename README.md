# Poker Arena

`poker_arena` is an event-sourced No Limit Texas Hold'em engine, CUDA training pipeline, and local browser table for poker-bot experiments.

The project currently targets one Windows workstation with an NVIDIA RTX 5070 Ti and the conda environment at `C:\conda_envs\poker_ai_env`.

## What is implemented

### Poker engine

- Two to nine seats with fixed display-seat identities.
- Integer chip stacks, 10/20 blinds in the web room, and arbitrary legal `raise_to(total_commitment)` actions.
- Preflop, flop, turn, river, showdown, folds, checks, calls, raises, all-ins, and automatic board runout when no further betting is possible.
- Main-pot and side-pot construction from each player's total contribution.
- Uncalled all-in excess is returned with an `uncalled_bet_returned` event instead of being reported as another pot win.
- Equal winning hands split a pot. If an integer split leaves an odd chip, the current engine records `odd_chip_carryover` and adds that chip to the next hand's first pot.
- Internal five-card and best-of-seven-card evaluator.
- Event history, deterministic state snapshots, replay support, public views, and player-specific views that hide opponents' cards from bot policies.
- Legal-action validation, including minimum raises, short all-ins, and the rule that a player cannot fold when checking is free.
- Chip conservation and side-pot behavior covered by automated tests.

### Bots

- `CheckCallBot`: checks when possible and otherwise calls.
- `RandomLegalBot`: samples from currently legal actions.
- `TorchPolicyBot`: loads a self-describing `.pt` checkpoint, scores legal concrete actions, and chooses the highest-scoring action.
- One loaded Torch policy is shared by all reserved bot seats in a web-room process.
- Debug card reveal changes only the human-facing snapshot; bots still receive player views with opponents' hole cards hidden.

### Browser table

- Nine persistent display seats, human seat claims, host-controlled bot reservation, blinds, stacks, board cards, pot, action controls, and hand history.
- Fixed display-seat IDs remain stable when a player busts. For example, Seat 3 remains Seat 3 after Seat 2 is eliminated even though the engine compacts active players internally.
- Human raises are entered as `raise_by`; the browser converts them to the engine's `raise_to` representation.
- Min, half-pot, pot, all-in, custom raise, fold, check, call, and explicit next-hand controls.
- Finished hands remain visible until a human starts the next hand.
- Debug mode reveals every active player's hole cards and is enabled by default for the current development workflow.
- Host and guest URLs are printed at startup. Loopback host actions are authorized locally, while remote LAN host actions require the host session credential.
- Host-session cookies recover authorization across refreshes and reject expired seat sessions cleanly.
- Downloadable JSON session logs include boards, all players' hole cards, actions, payouts, stable display-seat IDs, and final stacks.
- Payout events distinguish main pots, side pots, split shares, uncontested wins, and uncalled-bet returns.

## Windows and CUDA setup

Python 3.11 or newer is required. Python 3.12 is the configured and recommended version.

From PowerShell in the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows.ps1
```

The script creates or updates `C:\conda_envs\poker_ai_env`, installs the editable project and test dependencies, verifies PyTorch CUDA support, and checks that exactly one RTX 50-series-capable GPU is visible. The configured environment uses PyTorch 2.8.0 with CUDA 12.9 wheels.

Override the environment path if needed:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows.ps1 `
  -EnvPath C:\path\to\another_env
```

## Run the browser table

The recommended launcher passes the checkpoint, device, and reveal setting directly to Python. It does not set or leave `POKER_BOT_MODEL`, `POKER_BOT_DEVICE`, or `POKER_ARENA_REVEAL_CARDS` in the PowerShell environment.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_poker_arena.ps1
```

Defaults:

- environment: `C:\conda_envs\poker_ai_env`
- checkpoint: `runs\poker_policy_gpu.pt`
- inference device: `cuda`
- port: `8000`
- debug card reveal: on
- bind address: `127.0.0.1`

Examples:

```powershell
# Hide opponents' cards.
.\scripts\run_poker_arena.ps1 -HideCards

# Use another checkpoint and port.
.\scripts\run_poker_arena.ps1 `
  -ModelPath runs\poker_policy_gpu_100k.pt `
  -Port 8100

# Listen on the trusted local network.
.\scripts\run_poker_arena.ps1 -Lan
```

The equivalent direct CLI is:

```powershell
C:\conda_envs\poker_ai_env\python.exe -B -m poker_arena.web `
  --model .\runs\poker_policy_gpu.pt `
  --device cuda `
  --reveal-cards `
  --port 8000
```

CLI options:

```text
--model PATH
--device {cuda,cpu,auto}
--reveal-cards / --no-reveal-cards
--port PORT
--lan
```

The launcher validates both the conda Python executable and checkpoint before starting. The checkpoint is loaded lazily when the first model-backed bot is reserved.

The server prints a private host URL and a guest URL. Open the newly printed host URL after every restart. LAN mode uses plain HTTP and is intended only for trusted local networks; it is not a public deployment configuration.

## Playing and logs

1. Start the server with `scripts\run_poker_arena.ps1`.
2. Open the printed host URL.
3. Claim a human seat with the displayed room code and a nickname.
4. Reserve one or more bots in unoccupied seats.
5. Play the hand and use **Next Hand** after reviewing a completed result.
6. Use **Download Log** to save the session JSON.

Session logs use zero-based numeric IDs in JSON but explicitly declare:

```json
{
  "seat_id_space": "display"
}
```

The GUI displays `seat_id + 1`. A player shown as Seat 3 therefore has JSON `seat_id: 2`. Hand-finished events also include `stacks_by_seat`, so inactive seats cannot cause later seats to be renumbered.

New logs include all dealt hole cards for debugging. Treat downloaded logs as private if real players use the table.

## Headless engine example

```powershell
C:\conda_envs\poker_ai_env\python.exe -B .\examples\headless_demo.py
```

The primary API is available from `poker_arena`:

```python
from poker_arena import Action, Table, TableConfig

table = Table(TableConfig(2, 10, 20, [2000, 2000], seed=7))
state = table.start_hand()
state = table.apply(Action.call())
```

## RTX 5070 Ti training

The production training path is CUDA-only and intentionally has no silent CPU fallback. It represents complete batches of poker states as PyTorch tensors and currently processes 1,024 independent hands in parallel by default.

At each decision checkpoint it:

1. Generates concrete legal fold/check/call/raise candidates.
2. Repeats each tensor state across the branch width.
3. Applies each candidate from the shared prefix.
4. Rolls branches to terminal states on CUDA.
5. Uses the acting player's terminal utility as the action-value target.
6. Stores features and targets in a CUDA-resident replay buffer.
7. Updates an `ActionValueNet` with AdamW and mixed precision.
8. Continues the training trajectory with epsilon-greedy action selection.

State transitions, legal actions, shuffled deals, street dealing, best-of-seven evaluation, terminal settlement, feature construction, replay sampling, inference, and optimization remain on the GPU. Python coordinates batches and writes checkpoint and summary files.

### 10k demonstration checkpoint

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_rtx5070ti.ps1 `
  -Iterations 10000 `
  -ModelOut runs\poker_policy_gpu.pt
```

The current 10k run recorded:

- 10,000 generated hands
- 83,849 decision checkpoints
- 2,171,567 generated branch samples
- 250,000 retained replay samples
- 779 optimizer updates
- bfloat16 automatic mixed precision

These values are recorded in `runs\training_summary_gpu.json`.

### 100k comparison experiment

A 100k run is a useful next experiment, but write it to a separate checkpoint so the 10k baseline remains available:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_rtx5070ti.ps1 `
  -Iterations 100000 `
  -ModelOut runs\poker_policy_gpu_100k.pt
```

The script currently writes its summary to `runs\training_summary_gpu.json`, so copy or rename the 10k summary before starting if both summaries must be retained. Training starts from a fresh model; checkpoint resume is not currently implemented.

The equivalent fully configurable command is:

```powershell
C:\conda_envs\poker_ai_env\python.exe -B .\examples\train_gpu_prefix_branch.py `
  --device cuda `
  --gpus 0 `
  --iterations 100000 `
  --parallel-hands 1024 `
  --branch-width 32 `
  --max-decisions-per-hand 64 `
  --max-rollout-actions 128 `
  --replay-capacity 250000 `
  --replay-warmup 8192 `
  --batch-size 8192 `
  --optimizer-steps-per-decision 2 `
  --final-epochs 3 `
  --learning-rate 0.0005 `
  --epsilon 0.15 `
  --model-out runs\poker_policy_gpu_100k.pt `
  --summary-out runs\training_summary_gpu_100k.json
```

Important controls:

- `--parallel-hands`: main GPU memory/throughput control.
- `--branch-width`: number of concrete candidate actions evaluated per state.
- `--max-rollout-actions`: safety bound for a rollout.
- `--replay-capacity`: number of action-value samples retained on CUDA.
- `--epsilon`: probability of a random continuation choice during training only.
- `--required-integer-actions`: exact raise totals that must be considered when legal.
- `--evaluator-chunk-size`: controls seven-card evaluator memory use.

## What 100k training can and cannot fix

Increasing 10k to 100k should provide broader deal coverage, approximately an order of magnitude more optimizer work, repeated turnover of the fixed replay buffer, and lower-variance action-value estimates. It may improve basic hand valuation, bet sizing, and some obvious over-folding.

It is not sufficient by itself to create professional bluffing or bluff-catching:

- `TorchPolicyBot` deploys a deterministic `argmax`; it never samples a learned mixed strategy.
- Training epsilon produces exploration data but is not used during inference.
- Branch targets use Monte Carlo random continuations, not counterfactual values against an equilibrium opponent.
- The trajectory encoder is a compact summary rather than a full learned opponent-range model.
- The trainer optimizes action values, not the average regret-matched strategy normally used to make poker policies difficult to exploit.

A sharper action-value model can even become more consistently "honest" if the current training distribution rewards value betting and folding. Therefore the 100k checkpoint should be treated as an empirical baseline, not as the final architecture.

Recommended evaluation compares the 10k and 100k checkpoints over many duplicate deals with seats rotated and records:

- big blinds won per 100 hands
- fold-to-bet and fold-to-raise by street
- call efficiency and river bluff-catch frequency
- aggression frequency and bet-size distribution
- showdown rate and showdown profit
- bets made with low estimated equity (bluffs) versus draws with improving equity (semi-bluffs)
- performance against always-aggressive, tight, check/call, random, and previous-checkpoint opponents

A principled stronger bot should eventually learn and deploy a probability distribution over actions from counterfactual regrets or an average self-play policy. Sampling that distribution makes bluffs and bluff-catches emerge as parts of balanced ranges. Adding a fixed "bluff percentage" or merely sampling low-value actions would add noise, not poker intelligence.

## Other training implementations

The repository retains three useful paths:

- `examples\train_cfr.py`: small CPU sampled-CFR reference implementation.
- `examples\train_prefix_branch.py`: CPU engine-connected prefix-branch experimentation and replay generation.
- `examples\train_gpu_prefix_branch.py`: production tensorized CUDA Monte Carlo action-value trainer.

The older finite abstraction contains fold, check, call, minimum raise, half-pot, pot, and all-in labels. The CUDA path instead keeps concrete legal integer `raise_to(total)` values as first-class actions and encodes their pot/stack ratios continuously.

The project also includes deterministic `TrajectoryEncoder` features and a trainable trajectory encoder for CPU experiments.

## Checkpoints

GPU training writes a self-describing Torch checkpoint containing:

- checkpoint format version
- model weights and hidden-layer configuration
- state, trajectory, and action dimensions
- table defaults
- concrete-action sampler metadata
- training metadata and statistics
- normalization metadata

Programmatic loading:

```python
from poker_arena import TorchPolicyBot

bot = TorchPolicyBot.from_checkpoint(
    "runs/poker_policy_gpu.pt",
    device="cuda",
)
```

## Tests

Run the complete suite without writing Python bytecode or pytest cache files:

```powershell
C:\conda_envs\poker_ai_env\python.exe -B -m pytest -p no:cacheprovider -q
```

The suite covers cards, evaluator rankings, betting rules, all-in runouts, side pots, uncalled returns, stable web-seat IDs, session logs, host authentication, browser assets, bots, CPU CFR components, Torch checkpoints, CUDA CLI configuration, and tensor/reference-engine agreement.

## Project layout

```text
src/poker_arena/                 rules engine, views, events, bots
src/poker_arena/cfr/             CPU CFR, feature encoders, Torch model, CUDA trainer
src/poker_arena/web/             FastAPI room/server and browser assets
examples/                        headless and training entry points
scripts/bootstrap_windows.ps1    conda/CUDA environment setup
scripts/run_poker_arena.ps1      environment-isolated web launcher
scripts/train_rtx5070ti.ps1      single-GPU training launcher
tests/                           automated test suite
runs/                            checkpoints and training summaries
```

## Current limitations

- The trained web bot is deterministic and does not yet deploy a balanced mixed strategy.
- The CUDA trainer is an action-value Monte Carlo prefix-branch method, not a converged Deep CFR implementation.
- No checkpoint-resume or league-training workflow is implemented.
- No automated exploitability or best-response benchmark is implemented.
- The web room is in-memory; restarting the process creates a new room and host token.
- Downloaded logs expose all hole cards for debugging.
- LAN mode has no TLS and must remain on a trusted network.
- Odd split-pot chips currently carry into the next hand rather than following a casino-specific odd-chip assignment rule.
