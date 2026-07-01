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

Run the demo from this folder:

```bash
PYTHONPATH=src python3.13 examples/headless_demo.py
```

Run the local web table:

```bash
PYTHONPATH=src python3.13 -m poker_arena.web --port 8000
```

Run the web table with a trained `.pt` bot policy:

```bash
POKER_BOT_MODEL=runs/poker_policy.pt \
POKER_BOT_DEVICE=auto \
PYTHONPATH=src python3.13 -m poker_arena.web --port 8000
```

If `POKER_BOT_MODEL` is not set, reserved bot seats keep the old placeholder
behavior and the table pauses when an unavailable bot is asked to act.

Use `--lan` to listen on your LAN:

```bash
PYTHONPATH=src python3.13 -m poker_arena.web --port 8000 --lan
```

The server prints a private host URL/token and a guest room code. LAN mode uses plain HTTP and is not safe for public Internet exposure. For remote play outside a trusted local network, put the app behind HTTPS with a real deployment or a tunnel that provides access controls.

Run the tests with pytest if installed:

```bash
PYTHONPATH=src python3.13 -m pytest
```

The tests are also compatible with the standard library runner:

```bash
PYTHONPATH=src python3.13 -m unittest discover -s tests -v
```

Run a small engine-connected prefix-branch CFR training smoke test:

```bash
PYTHONPATH=src python3.13 examples/train_cfr.py
```

Run configurable prefix-branch training:

```bash
PYTHONPATH=src python3.13 examples/train_prefix_branch.py \
  --iterations 1000 \
  --branch-width 32 \
  --branch-depth 8 \
  --integer-action-budget 32 \
  --max-workers 8 \
  --output runs/prefix_branch_summary.json
```

The prefix-branch rollout engine is CPU-side. Use `--max-workers` to
parallelize branch expansion while keeping arbitrary integer raises in the
sample set.

Install the optional Torch training stack to train a deployable action-value
policy:

```bash
python3.13 -m pip install -e ".[train]"
```

Depending on the machine, CUDA support may require installing the PyTorch wheel
from the PyTorch CUDA index URL instead of the default Python package index.

Train a `.pt` checkpoint with the GPU-aware prefix-branch pipeline:

```bash
# CPU only
PYTHONPATH=src python3.13 examples/train_gpu_prefix_branch.py \
  --device cpu \
  --gpus none \
  --model-out runs/poker_policy.pt

# One visible GPU
PYTHONPATH=src python3.13 examples/train_gpu_prefix_branch.py \
  --device cuda \
  --gpus 0 \
  --model-out runs/poker_policy.pt

# Multiple visible GPUs
PYTHONPATH=src python3.13 examples/train_gpu_prefix_branch.py \
  --device cuda \
  --gpus 0,1 \
  --model-out runs/poker_policy.pt
```

The GPU trainer keeps rollouts on CPU, batches action-conditioned samples for
Torch, uses `torch.nn.DataParallel` when multiple visible GPUs are requested,
and writes checkpoint metadata beside the model weights. Load the result as a
bot policy:

```python
from poker_arena import TorchPolicyBot

bot = TorchPolicyBot.from_checkpoint("runs/poker_policy.pt", device="auto")
```

The CFR layer is not a toy Kuhn/Leduc implementation. It connects to the
No Limit Hold'em engine and supports two training paths.

The newer prefix-branch path samples concrete legal integer actions from the
engine, expands multiple branches from the same table prefix, and records
action-conditioned training samples. Integer `raise_to(total)` amounts remain
first-class actions; pot and stack ratios are continuous features in the action
embedding, not fixed action labels.

```python
from poker_arena import PrefixBranchCFRTrainer, PrefixBranchTrainingConfig, TableConfig

trainer = PrefixBranchCFRTrainer(
    table_config=TableConfig(3, 10, 20, [2000, 2000, 2000], seed=13),
    config=PrefixBranchTrainingConfig(branch_width=16, integer_action_budget=32),
)
result = trainer.train(iterations=20)
samples = trainer.training_samples
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
