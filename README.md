# Poker Arena

`poker_arena` is a small Python rules engine for No Limit Hold'em experiments.

The first version focuses on a deterministic, testable engine:

- 2-9 seats
- integer chips
- arbitrary legal `raise_to(total_commitment)` actions
- standard blinds, betting rounds, all-ins, side pots, and showdown
- internal 5/7-card evaluator
- event history with public and player-specific views
- simple bot policy hooks and a headless demo

## Windows setup

This project requires Python 3.11 or newer. Python 3.12 is the recommended
version for the Windows/CUDA environment. From PowerShell in the repository
root, update or create the configured environment and verify CUDA with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_windows.ps1
```

The default environment location is `C:\conda_envs\poker_ai_env`. Override it
with `-EnvPath` if needed. The environment uses the PyTorch CUDA 12.9 wheel
index, which supports the RTX 50-series GPU used for this project.

Run the complete test suite:

```powershell
conda run --prefix C:\conda_envs\poker_ai_env python -B -m pytest
```

Run the headless demo or local web table:

```powershell
conda run --prefix C:\conda_envs\poker_ai_env python -B examples/headless_demo.py
conda run --prefix C:\conda_envs\poker_ai_env python -B -m poker_arena.web --port 8000
```

Run the web table with a trained `.pt` bot policy:

```powershell
$env:POKER_BOT_MODEL = "runs/poker_policy_gpu.pt"
$env:POKER_BOT_DEVICE = "auto"
conda run --prefix C:\conda_envs\poker_ai_env python -B -m poker_arena.web --port 8000
```

If `POKER_BOT_MODEL` is not set, reserved bot seats keep the old placeholder
behavior and the table pauses when an unavailable bot is asked to act.

Use `--lan` to listen on your LAN:

```powershell
conda run --prefix C:\conda_envs\poker_ai_env python -B -m poker_arena.web --port 8000 --lan
```

The server prints a private host URL/token and a guest room code. LAN mode uses plain HTTP and is not safe for public Internet exposure. For remote play outside a trusted local network, put the app behind HTTPS with a real deployment or a tunnel that provides access controls.

Run a small engine-connected prefix-branch CFR training smoke test:

```powershell
conda run --prefix C:\conda_envs\poker_ai_env python -B examples/train_cfr.py
```

## RTX 5070 Ti training

The production trainer is CUDA-only. It represents complete poker states as
PyTorch tensors and processes 1,024 independent hands concurrently. At each
decision it checkpoints the tensor batch, expands every sampled action into a
`hands x branches` batch, rolls those branches to terminal states on the GPU,
and learns continuously from a CUDA-resident replay buffer. The command fails
fast if CUDA is unavailable; there is no silent CPU fallback.

The convenience script is configured for a 10,000-hand run on CUDA device 0:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_rtx5070ti.ps1 -Iterations 10000
```

The equivalent configurable command is:

```powershell
conda run --prefix C:\conda_envs\poker_ai_env python -B examples/train_gpu_prefix_branch.py `
  --device cuda `
  --gpus 0 `
  --iterations 10000 `
  --parallel-hands 1024 `
  --branch-width 32 `
  --max-rollout-actions 128 `
  --batch-size 8192 `
  --model-out runs/poker_policy_gpu.pt `
  --summary-out runs/training_summary_gpu.json
```

The tensors include stacks, commitments, betting-round flags, actors, shuffled
decks, hole cards, board cards, trajectory summaries, and side-pot inputs. Legal
action generation, checkpoint expansion, state transitions, street dealing,
terminal rollouts, seven-card evaluation, side-pot settlement, replay sampling,
model inference, and optimization all execute on CUDA. Python only coordinates
large batches and writes progress/checkpoint files.

`--parallel-hands` is the main memory-throughput control. Lower it if CUDA runs
out of memory; raise it if the GPU still has ample memory. `--branch-width`
controls how many concrete actions share each prefix. `--max-rollout-actions`
is only a safety bound—unfinished branches at that bound receive a neutral
zero-sum pot-share estimate. The fixed-size replay buffer defaults to 250,000
GPU-resident samples.

The trainer writes model weights and self-describing checkpoint metadata. Its
state features include the acting player's hole cards and all public board
cards. Load the result as a bot policy:

```python
from poker_arena import TorchPolicyBot

bot = TorchPolicyBot.from_checkpoint("runs/poker_policy_gpu.pt", device="auto")
```

The project also retains the original CPU reference CFR implementation for
small correctness tests and experimentation. It is not used by the RTX script.

The CUDA prefix-branch path samples concrete legal integer actions, expands
multiple branches from each tensor checkpoint, and records action-conditioned
training targets without converting the replay data into Python objects.
Integer `raise_to(total)` amounts remain first-class actions; pot and stack
ratios are continuous features in the action embedding, not fixed labels.

```python
from poker_arena.cfr.gpu_prefix_branch import (
    GpuPrefixBranchTrainer,
    GpuPrefixBranchTrainingConfig,
)

# The CLI constructs the CUDA model, trainer, metadata, and checkpoint together;
# use examples/train_gpu_prefix_branch.py as the supported programmatic template.
```

The older sampled CFR baseline uses a finite action abstraction:

- `fold`
- `check`
- `call`
- `min_raise`
- `half_pot`
- `pot`
- `all_in`

The trajectory encoder turns event histories into deterministic numeric
vectors. Raise sizes are encoded with pot and stack ratios, so nearby integer
bet amounts produce nearby vectors. This is the starting point for replacing
tabular abstractions with learned action/trajectory representations later.

For learned trajectory embeddings, use `TrainableTrajectoryEncoder`. It builds
fixed-size vectors from ordered event histories, reconstructs pot/current-bet
context from snapshots when available, and can still infer public betting
context from snapshot-free logs.

```python
from poker_arena import TrainableTrajectoryEncoder

encoder = TrainableTrajectoryEncoder(embedding_dim=8, random_seed=7)
encoder.fit(list_of_hand_event_sequences, epochs=20, learning_rate=0.03)
embedding = encoder.transform(one_hand_events)
```

The CFR information-set encoder can also use these vectors instead of raw
action-history text:

```python
from poker_arena import CFRTrainer, InformationSetEncoder, TableConfig, TrajectoryEncoder

trainer = CFRTrainer(
    table_config=TableConfig(3, 10, 20, [2000, 2000, 2000], seed=13),
    infoset_encoder=InformationSetEncoder(trajectory_encoder=TrajectoryEncoder()),
)
```

The browser table displays raises as `raise_by` for humans. The engine and wire
format still use `raise_to(total_commitment)`, so the web client converts the
displayed raise amount into the backend total before submitting the action.
