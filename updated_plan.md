# Updated Plan and Implementation Summary

## Context

The training discussion started from `train_poker_v7.ipynb`, which is a standalone RLCard/PyTorch Deep CFR prototype. The notebook improves throughput by running 64 independent games in parallel, batching neural-network inference by player, using replay buffers, and training with mixed precision. It is useful as a batching and GPU-utilization reference, but it uses a limited ratio-based action abstraction and does not implement same-prefix branch expansion.

The project requirement was clarified as follows:

- Preserve arbitrary integer `raise_to(total)` actions as first-class actions.
- Do not reduce training or inference to fixed pot-ratio choices.
- Build training around prefix branch expansion: sample many concrete legal actions from the same table snapshot, clone branches, and evaluate those branches.
- Use trajectory and action embeddings to guide exploration and shape action-conditioned training samples.

## What Changed

### Integer-Action Sampling

Added `IntegerActionSampler`, which samples concrete legal actions from the engine:

- Includes fold, check, and call when legal.
- Samples arbitrary legal integer `raise_to(total)` amounts.
- Preserves required/user-provided integer raise amounts when they are legal.
- Deduplicates concrete actions by action type and total amount.
- Can use embedding novelty to prefer less-covered action regions.

### Action and Trajectory Embeddings

Added `ActionEmbedding` and `EmbeddingCoverageIndex`.

`ActionEmbedding` encodes concrete actions with continuous numeric features, including:

- action type,
- raise total,
- position within legal raise interval,
- amount added over current bet,
- call/current-bet/min/max context,
- optional trajectory embedding appended to the action vector.

This means nearby integer bet amounts, such as `raise_to(123)` and `raise_to(124)`, produce nearby embeddings without turning them into fixed labels.

`EmbeddingCoverageIndex` records explored embeddings and assigns higher novelty to less-covered regions. This currently guides exploration; it does not replace exact information-set identity.

### Prefix Branch Expansion

Added `PrefixBranchExplorer`.

It takes one live table prefix, clones the table for each selected concrete action, applies that action in the clone, rolls forward for a bounded depth, and returns `BranchResult` objects containing:

- the concrete action,
- utility vector,
- terminal flag,
- rollout step count,
- trajectory embedding,
- action embedding.

The original table snapshot is not mutated by branch expansion.

### Prefix-Branch CFR Trainer

Added `PrefixBranchCFRTrainer` and `PrefixBranchTrainingConfig`.

The trainer:

- samples legal integer actions at each decision point,
- expands branches from the same prefix,
- computes exact branch rollout utilities as the primary target,
- updates per-information-set regrets for the expanded concrete actions,
- records action-conditioned training samples for future learned inference,
- tracks embedding coverage for novelty-guided exploration.

`neighbor_weight` exists in the config and defaults to `0.0`, so neighbor smoothing is disabled by default. This keeps exact branch rollout outcomes as the main training signal.

### Public Interfaces

The following interfaces were added and exported from `poker_arena.cfr` and top-level `poker_arena`:

- `ActionEmbedding`
- `ActionTrainingSample`
- `BranchResult`
- `EmbeddingCoverageIndex`
- `IntegerActionSampler`
- `PrefixBranchCFRTrainer`
- `PrefixBranchExplorer`
- `PrefixBranchTrainingConfig`

### Example and Docs

Updated `examples/train_cfr.py` so the formal training example now uses the prefix-branch pipeline and reports:

- `prefix_branch: true`,
- iteration count,
- information-set count,
- number of action-conditioned training samples,
- a small sample preview.

Updated the README CFR section to state that integer `raise_to(total)` actions are first-class actions in the new prefix-branch path, while pot and stack ratios are continuous embedding features rather than fixed action labels.

## Verification

Added `tests/test_prefix_branch_training.py` with coverage for:

- arbitrary legal integer raises are sampled and preserved,
- required/user-entered legal integer raises survive even with a small sample budget,
- nearby integer bets produce nearby action embeddings,
- prefix branch expansion does not mutate the original table,
- embedding coverage marks repeated regions as less novel,
- neighbor smoothing is disabled by default,
- trainer nodes only track actually expanded branches,
- the training example uses the prefix-branch pipeline.

Verification commands run successfully:

```bash
python3.13 -B -c 'import sys, unittest; sys.path.insert(0, "/Users/sunyihao/Desktop/poker"); sys.path.insert(0, "/Users/sunyihao/Desktop/poker/src"); suite=unittest.defaultTestLoader.discover("/Users/sunyihao/Desktop/poker/tests"); result=unittest.TextTestRunner(verbosity=1).run(suite); raise SystemExit(0 if result.wasSuccessful() else 1)'
```

Result: `46 tests OK`.

```bash
python3.13 -m compileall /Users/sunyihao/Desktop/poker/src /Users/sunyihao/Desktop/poker/examples /Users/sunyihao/Desktop/poker/tests
```

Result: compile check passed.

```bash
python3.13 -B /Users/sunyihao/Desktop/poker/examples/train_cfr.py
```

Result: example ran successfully and produced prefix-branch training samples.

## Current Boundary

The implemented work adds the integer-action prefix-branch training scaffold and action-conditioned sample generation. It does not yet add a Torch/GPU value or regret network. The new `ActionTrainingSample` records are the bridge to that next Deep CFR-style learned inference step.

## Recommended Next Step

Add the learned action-conditioned model that consumes `(state_features, trajectory_embedding, action_embedding)` and trains on the collected `ActionTrainingSample` targets. That is the point where GPU batching should become central: branch generation remains engine-side, while scoring and training many concrete integer actions can run as large tensor batches.
