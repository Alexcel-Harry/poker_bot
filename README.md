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
- `DeepCFRAveragePolicyBot`: loads a multi-player Deep CFR average-strategy checkpoint and samples the learned legal-action distribution for its acting seat.
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

# Reproduce the invalidated 500-iteration policy for diagnostics only.
.\scripts\run_poker_arena.ps1 `
  -ModelPath runs\cuda_deep_cfr_3p_500_policy.pt `
  -Device cuda

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

For a valid Deep CFR run, use its small inference policy with the GUI, not its multi-gigabyte resumable training snapshot. The checkpoint loader detects the Deep CFR policy format automatically; no inference code or GUI configuration change is required. The existing `cuda_deep_cfr_3p_100_*` and `cuda_deep_cfr_3p_500_*` artifacts are invalidated training outputs and must not be used as production opponents.

The server prints a private host URL and a guest URL. Open the newly printed host URL after every restart. LAN mode uses plain HTTP and is intended only for trusted local networks; it is not a public deployment configuration.

## Playing and logs

1. Start the server with `scripts\run_poker_arena.ps1`.
2. Open the printed host URL.
3. Claim a human seat with the displayed room code and a nickname.
4. Reserve one or more bots in unoccupied seats.
5. Press **Start Game** after every intended player and bot has occupied a seat. Joining or reserving seats does not deal a hand.
6. Play the hand and use **Next Hand** after reviewing a completed result.
7. Use **Download Log** to save the session JSON.

The host can start with any supported table size of at least two occupied seats. Joining humans or reserving bots only prepares the lobby: the backend cannot create the first table or deal cards until the authenticated host presses **Start Game**. The button remains disabled until at least two seats are occupied and becomes disabled permanently after the first hand begins.

For a future mixed-count checkpoint, prepare any supported table size from three through nine before **Start Game** is pressed. For solo play, claim one human seat, reserve the intended number of bot seats, and then start the game. All bot seats share one loaded policy object, while the policy selects the strategy network corresponding to the acting engine seat.

Mixed-count policies record every player count encountered in training. Tables from three through nine use the corresponding trained seat network directly with no mismatch warning. A two-player table remains allowed by the GUI, but it uses the explicitly warned ensemble fallback because two-player play is outside the production training distribution. Legacy fixed-count checkpoints retain their earlier fallback behavior at mismatched table sizes.

The browser table remains at 2,000 chips with 10/20 blinds (100 BB). The invalidated runs used 200 chips with 5/10 blinds (20 BB). The next formal run is planned for 1,000 chips with 5/10 blinds (100 BB), and its policy will be selectable through the same `-ModelPath` option without code changes.

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

## Deep CFR training

The primary learning path is now mixed three-to-nine-player external-sampling Deep CFR. Before every traversal session, the CUDA RNG independently samples a valid player count. Sessions with the same count are grouped into CUDA batches, preserving parallel throughput without forcing every environment in a batch to use one permanent table size. It addresses the main limitations of the original 10k action-value checkpoint:

- A traverser decision recursively branches over every legal abstract action. Descendant traverser decisions branch again, so this is branches-over-branches rather than one flat branch followed by a random rollout.
- Opponent and chance nodes are sampled independently. Separate traversals receive independently shuffled undealt cards while preserving the public/private prefix already observed.
- A projected frontier larger than the configured row budget is no longer fatal. The trainer splits its parent states into contiguous CUDA chunks, traverses each chunk independently at the same logical depth, concatenates their values in original order, and then continues normal back-propagation. `frontier_chunk_splits` and `maximum_projected_frontier_rows` expose how often this path was needed.
- The current policy is derived from positive predicted counterfactual regrets. Deployment uses a separately learned average strategy and samples its action distribution; it is not deterministic `argmax` and does not hardcode bluff frequencies.
- Bet sizing uses a compact legal abstraction: fold/check/call, minimum raise, one-third-pot, three-quarter-pot, 1.5x-pot, and all-in. Duplicate concrete raise totals are removed for short stacks.
- Every possible seat up to the configured maximum has its own advantage network, average-strategy network, and two reservoir memories. A seat is trained only in sampled table sizes where it exists, and the fixed-width state features explicitly encode the current player count. Early data is not simply overwritten by recent play.

The implementation has two paths that share the same policy format:

- `examples\train_deep_cfr.py` is the readable recursive reference trainer. It supports two-player Kuhn/Leduc validation and three-to-nine-player Hold'em experiments.
- `examples\train_cuda_deep_cfr.py` is the production single-GPU trainer. It executes traversals as level-synchronous CUDA frontiers, expands every traverser action, samples one action at each non-traverser node, and back-propagates the selected traverser's terminal value through the recorded frontier.

The exact Nash-convergence guarantee associated with CFR is specific to two-player zero-sum games. Multi-player poker remains constant-sum, but independent regret minimization is not guaranteed to converge to a Nash equilibrium. Mixed three-to-nine-player training is therefore an intentional practical extension whose checkpoints must be judged empirically, not labeled solved from training loss alone.

### Validation gates

Kuhn and Leduc include exact game trees, exact expected value, an infoset-consistent exact best response, NashConv/exploitability, and a tabular CFR baseline. Use them before changing traversal, regret, or averaging logic:

```powershell
C:\conda_envs\poker_ai_env\python.exe -B .\examples\train_deep_cfr.py `
  --game kuhn `
  --iterations 100 `
  --snapshot-out runs\kuhn_snapshot.pt `
  --policy-out runs\kuhn_policy.pt `
  --summary-out runs\kuhn_summary.json
```

The toy-game summary records expected values, best-response values, NashConv, and exploitability. Exact exploitability is deliberately not claimed for full Hold'em.

### Production RTX 5070 Ti run

Start with a short smoke run, inspect the summary, and only then scale the iteration and traversal counts:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_cuda_deep_cfr.ps1 `
  -Iterations 5 `
  -TraversalsPerPlayer 256 `
  -ParallelTraversals 64 `
  -MinPlayers 3 `
  -Seats 9 `
  -SnapshotOut runs\cuda_deep_cfr_smoke_snapshot.pt `
  -PolicyOut runs\cuda_deep_cfr_smoke_policy.pt
```

For a longer run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_cuda_deep_cfr.ps1 `
  -Iterations 100 `
  -TraversalsPerPlayer 4096 `
  -ParallelTraversals 192 `
  -MinPlayers 3 `
  -Seats 9 `
  -StartingStack 1000 `
  -SnapshotOut runs\cuda_deep_cfr_snapshot.pt `
  -PolicyOut runs\cuda_deep_cfr_average_policy.pt
```

`-MinPlayers` and `-Seats` define the inclusive randomized range; setting both to the same value preserves fixed-count training. `TraversalsPerPlayer` is performed for every possible seat up to `-Seats`, so a 3–9 run performs nine times that number of sessions per iteration. The sampled sessions are grouped by player count before tensor traversal.

Reservoir capacities are per possible seat. The mixed-count production defaults use 300,000 advantage and 300,000 strategy samples per seat, putting aggregate reservoir memory about 10% below the former three-seat configuration of one million samples per seat. Production parallelism defaults to 192 traversals and materialized frontier chunks target at most 131,072 rows. Larger projected trees are split rather than terminating the run. These settings leave more headroom for the desktop and nine-seat tensor states; raising the capacities or returning to 256-512 parallel traversals can still exhaust VRAM or make the machine intermittently unresponsive.

Resume adds the requested number of new iterations and restores every seat's networks and reservoirs, player-count sampling counters, and random-number-generator states. Both ends of the player-count range must match the snapshot. Runtime-only controls such as iteration count, parallel traversal count, and frontier safety limit may change safely. Resume only from a post-fix snapshot whose summary passes the reservoir checks below; the invalidated 100/500 snapshots are rejected by the loader:

```powershell
.\scripts\train_cuda_deep_cfr.ps1 `
  -Iterations 400 `
  -MinPlayers 3 `
  -Seats 9 `
  -ParallelTraversals 192 `
  -SnapshotEvery 100 `
  -ResumeSnapshot runs\cuda_deep_cfr_mixed_3to9_fresh_100_snapshot.pt `
  -SnapshotOut runs\cuda_deep_cfr_mixed_3to9_fresh_500_snapshot.pt `
  -PolicyOut runs\cuda_deep_cfr_mixed_3to9_fresh_500_policy.pt
```

`--snapshot-every` atomically refreshes the resumable snapshot after that many new iterations; an interrupted write cannot replace the last complete snapshot. Important production controls are `--parallel-traversals` and `--max-frontier-rows` for peak GPU memory, reservoir capacities for retained training data, and the per-network training-step counts. A completed sound traversal should report `depth_limit_rollouts: 0`, `latest_iteration_retained: true`, and an `advantage_iteration_ranges` and `strategy_iteration_ranges` entry for every seat whose maximum is the completed iteration. A failed freshness check invalidates the run regardless of counters or loss. A nonzero depth-limit count also requires investigation before trusting the run.

Deep CFR iteration counts are not comparable to the old checkpoint's generated-hand count. Each iteration performs many player-specific traversals and recursively evaluates multiple actions.

### Invalidated local runs and root cause (2026-08-01)

The original 100- and 500-iteration runs completed mechanically, but their learned policies are invalid. `CudaReservoirBuffer.add` attempted batched replacement with advanced-indexed `.copy_()` and `.fill_()` calls. PyTorch advanced indexing returned temporary tensors, so the underlying reservoir rows stopped changing after the initial one-million-row fill even though `samples_seen`, traversal counts, losses, and snapshots continued to advance. The 500-iteration run inherited the frozen memory from the 100-iteration snapshot.

| Checkpoint | Training path | Traversals | Status |
| --- | --- | ---: | --- |
| 100 iterations | Fresh three-player, 20 BB | 1,228,800 | Invalid; do not deploy or resume |
| 500 iterations | Resume at 100, add 400 | 6,144,000 | Invalid; do not deploy or resume |

Forensic inspection of the 500-iteration snapshot found newest retained advantage iterations `[8, 6, 6]` and newest retained strategy iterations `[3, 5, 3]` instead of 500. A duplicate-deal audit also found that forcing a pre-flop all-in with 75o lost about 2.63 BB per hand against the deployed average opponents and about 4.05 BB against final regret-matched opponents. This confirms destructive policy drift rather than healthy learned aggression.

The replacement path now uses `index_copy_` and `index_fill_` on the original tensors. A deterministic regression test fills a reservoir, adds thousands of later samples, and verifies that features, masks, targets, and iteration ages are replaced together. Training summaries now expose retained iteration ranges, and snapshot loading rejects the severe frozen-memory signature.

The previous throughput benchmark remains useful only for sizing: at 4,096 traversals per player, increasing parallel traversals from 256 to 512 reduced elapsed time from 27.76 seconds to 14.92 seconds (about 1.86x) while peak allocated GPU memory rose from about 4.1 GB to 5.1 GB. It does not validate policy quality.

No corrected formal checkpoint exists yet. The previously requested launch was stopped before iteration output or a checkpoint was written. The next steps are:

1. On explicit approval, start a fresh mixed 3–9-player 100-BB run; never resume either invalidated snapshot.
2. At every saved checkpoint, require coverage in `player_count_traversals`, fresh reservoir ranges, zero depth-limit rollouts, finite networks, and legal-action inference smoke tests at every supported table size.
3. Before GUI deployment, audit weak offsuit pre-flop all-ins and evaluate at least 10,000 rotated duplicate deals per matchup and player count.
4. Compare confidence intervals, seat, table-size, and street telemetry plus representative hand histories before deciding whether to add iterations or model capacity.

### Evaluation and deployment

`play_rotating_match` evaluates every shuffled deck once for each placement of the tracked policy. At three seats, the tracked checkpoint plays each deal three times—once in every seat—against two instances of the opponent. It reports a grouped 95% confidence interval and action telemetry by street, action type, facing-bet status, and raise-to-pot ratio:

```python
from poker_arena import TorchPolicyBot, play_rotating_match

result = play_rotating_match(
    lambda: TorchPolicyBot.from_checkpoint("runs/cuda_deep_cfr_average_policy.pt", device="cuda"),
    lambda: TorchPolicyBot.from_checkpoint("runs/poker_policy_gpu.pt", device="cuda"),
    deals=10_000,
    seats=3,
)
print(result.to_dict())
```

The average-strategy checkpoint is self-describing. The existing loader recognizes it and returns a stochastic `DeepCFRAveragePolicyBot`:

```python
from poker_arena import TorchPolicyBot

bot = TorchPolicyBot.from_checkpoint(
    "runs/cuda_deep_cfr_average_policy.pt",
    device="cuda",
)
```

The policy records its supported player-count range. A supported table selects the corresponding seat-specific strategy network; an unsupported two-to-nine-player table uses the explicitly warned ensemble fallback described above. Preserve the original 10k checkpoint as a baseline and compare checkpoints on rotated duplicate deals rather than judging a few manually selected hands. Bluffing is expected to emerge as part of the learned mixed strategy, but it still needs empirical validation; the architecture does not guarantee a strong policy after a short run.

## Improved prefix-branch baseline

The original CUDA action-value trainer remains available as a controlled baseline. It now uses the same compact pot-sized candidates, adds explicit `added / pot` and `total / (pot + call)` action features, averages each sibling action over independently reshuffled future-card replicas, records candidate/selection telemetry, and writes a fully resumable training snapshot. The loader retains the 12-feature path required by existing checkpoints:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_rtx5070ti.ps1 `
  -Iterations 10000 `
  -BranchWidth 8 `
  -ChanceReplicas 4 `
  -ModelOut runs\poker_policy_gpu_v2.pt `
  -TrainingSnapshotOut runs\poker_policy_gpu_v2_training.pt
```

Resume with `-ResumeSnapshot`. This baseline is still Monte Carlo action-value learning with epsilon-greedy continuations and deterministic inference, so more steps alone should not be expected to solve balanced bluffing.

## Training artifacts

Training snapshots and inference policies are intentionally separate:

- snapshots contain optimizer/training state, networks, replay or reservoir contents, RNG states, counters, and telemetry needed for an exact continuation;
- inference policies contain only the schema, encoders, learned policy networks, and training metadata required to play.

Other retained implementations are `examples\train_cfr.py` (small sampled-CFR reference), `examples\train_prefix_branch.py` (CPU engine-connected prefix branching), and `examples\train_gpu_prefix_branch.py` (CUDA Monte Carlo baseline).

## Tests

Run the complete suite without writing Python bytecode or pytest cache files:

```powershell
C:\conda_envs\poker_ai_env\python.exe -B -m pytest -p no:cacheprovider -q
```

The suite covers cards, evaluator rankings, betting rules, all-in runouts, side pots, uncalled returns, stable web-seat IDs, session logs, host authentication, browser assets, bots, exact Kuhn/Leduc exploitability, recursive and mixed-count CUDA Deep CFR, reservoir sampling, resumable snapshots, seat-rotated duplicate-deal evaluation, CUDA CLI configuration, and tensor/reference-engine agreement.

## Project layout

```text
src/poker_arena/                 rules engine, views, events, bots
src/poker_arena/cfr/             toy games, exact evaluation, Deep CFR, CUDA trainers
src/poker_arena/evaluation_arena.py  duplicate/rotated-deal policy evaluation and telemetry
src/poker_arena/web/             FastAPI room/server and browser assets
examples/                        headless and training entry points
scripts/bootstrap_windows.ps1    conda/CUDA environment setup
scripts/run_poker_arena.ps1      environment-isolated web launcher
scripts/train_rtx5070ti.ps1      single-GPU training launcher
scripts/train_cuda_deep_cfr.ps1  production 3-9 player Deep CFR launcher
tests/                           automated test suite
runs/                            checkpoints and training summaries
```

## Current limitations

- Production Deep CFR samples table sizes from a configured 3-9 range and stores a separate policy/memory set for every possible seat up to the maximum. Higher seat IDs necessarily receive data only from larger tables, so coverage and strength must be measured separately by player count. Two-player inference remains an out-of-distribution ensemble fallback.
- Multi-player CFR does not inherit the two-player zero-sum Nash-convergence guarantee, so rotated empirical evaluation is mandatory.
- Full Hold'em is evaluated empirically; exact exploitability and exact best response are available only for the finite Kuhn/Leduc validation games.
- The compact action abstraction cannot represent every legal integer bet size.
- No league-training workflow is implemented.
- A trained average strategy is not assumed converged merely because the pipeline completed; compare it against baselines with duplicate deals and confidence intervals.
- The web room is in-memory; restarting the process creates a new room and host token.
- Downloaded logs expose all hole cards for debugging.
- LAN mode has no TLS and must remain on a trusted network.
- Odd split-pot chips currently carry into the next hand rather than following a casino-specific odd-chip assignment rule.

## Acknowledgment

Great thanks to Codex and ChatGPT for implementing the codebase.
