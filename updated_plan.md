# GPU Training Rewrite

## Goal

Make same-prefix branch expansion the unit of GPU parallelism instead of using
the GPU only for a short neural-network fit after CPU rollouts.

## Implemented Architecture

The supported RTX entry point is now `examples/train_gpu_prefix_branch.py`.
It is CUDA-only and fails fast when CUDA or exactly one requested GPU is not
available. The legacy Python engine/CFR path remains as a small reference
implementation, but the RTX script does not call it.

`TensorPokerState` stores a batch of complete Hold'em states in device tensors:

- stacks, street and total commitments, folds, all-ins, acted flags, and actors;
- independently shuffled 52-card decks, private cards, boards, and deal cursors;
- current bets and minimum full-raise state;
- tensor-native trajectory aggregates needed by the deployed model.

For every live decision checkpoint, the trainer:

1. generates concrete legal fold/check/call/integer-raise candidates on CUDA;
2. repeats each checkpoint into a fixed `hands x branches` tensor batch;
3. applies every valid candidate without mutating the original prefix;
4. performs all random rollout state transitions and card dealing on CUDA;
5. evaluates seven-card hands and settles main/side pots on CUDA;
6. writes action-conditioned targets into a CUDA-resident ring replay buffer;
7. performs online mixed-precision optimizer steps throughout generation;
8. chooses a policy/epsilon action and continues from its immediate checkpoint.

The main throughput control is `--parallel-hands` (default 1,024). The default
branch width is 32, so a full checkpoint can expose roughly 32,768 branch rows
to the GPU before terminal rollouts compact naturally through masks.

## Correctness and Compatibility

The generated checkpoint retains the existing `StateFeatureEncoder`,
`ActionEmbedding`, trajectory dimension, and `TorchPolicyBot` input contract.
The model therefore remains loadable by the web table and existing bot API.

The tensor simulator includes exact integer betting rules, short all-ins,
street transitions, exact best-five-of-seven ranking, unequal-stack side pots,
split pots, and deterministic odd-chip assignment within an independent hand.
If the rollout safety limit is reached before a terminal state, the unfinished
pot is assigned as a neutral zero-sum share among live players.

## Verification Performed Without Launching Training

- Tensor evaluator ordering matched the Python reference evaluator across
  randomized seven-card hands.
- A three-player unequal-stack all-in/side-pot hand matched the reference engine
  after every action and at terminal settlement.
- Batched random rollouts terminated and conserved chips.
- Checkpoint branch tensors did not mutate their source prefix.
- CUDA-only CLI validation, configuration validation, and help paths passed.
- The complete 79-test project suite passed.
- `git diff --check` passed.

No rewritten CUDA training run was launched during this pass.

## Supported 10k Command

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_rtx5070ti.ps1 -Iterations 10000
```

The script defaults to `runs/poker_policy_gpu.pt` and
`runs/training_summary_gpu.json`, so it does not overwrite the prior CPU-rollout
checkpoint unless an existing output path is explicitly supplied.
