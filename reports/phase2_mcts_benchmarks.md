# Phase 2 MCTS Benchmark Notes

Date: 2026-05-13  
Machine: `canada-l.cs.ucl.ac.uk`  
GPU: NVIDIA GeForce RTX 3090 Ti, 24GB

## Summary

The rollout MCTS implementation is useful as a baseline and diagnostic tool,
but it should not be treated as the final AlphaZero data engine. Large leaf
batches maximize GPU throughput by evaluating many random rollouts together,
but they make UCT less adaptive because many leaves are selected before values
are backed up. Small leaf batches produce much stronger single-root play, but
GPU utilization and data throughput drop sharply.

## Deep Rollout MCTS Runs

| Mode | Batch | Sims/Root | Leaf Batch | Rollouts/Leaf | Chunk | Roots/sec | Visits/sec | Est. Rollout Games/sec | Peak CUDA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Deep balanced | 1024 | 128 | 16,384 | 128 | 2,097,152 | 12.67 | 1,621.9 | 207.6k | 7.0GB |
| More data/weaker | 1024 | 64 | 16,384 | 128 | 2,097,152 | 24.98 | 1,598.8 | 204.6k | 7.0GB |
| Heavier rollout | 1024 | 128 | 16,384 | 256 | 4,194,304 | 9.29 | 1,189.5 | 304.5k | 14.1GB |
| Terminal good-play config | 1 | 8192 | 16 | 256 | 65,536 | n/a | n/a | ~2.1M rollouts/move | ~50% util observed |

## Interpretation

The terminal good-play command:

```csh
python3 -m connect4_zero.scripts.play_terminal \
  --device cuda \
  --simulations-per-root 8192 \
  --max-leaf-batch-size 16 \
  --rollouts-per-leaf 256 \
  --max-rollouts-per-chunk 65536
```

plays much better because it restores the adaptive UCT loop:

```text
select -> evaluate -> backprop -> select again
```

The high-throughput settings instead approximate:

```text
select many leaves with stale statistics -> evaluate batch -> backprop later
```

That is good for rollout throughput, but weak for tactical validation. The
AlphaZero path should therefore use neural PUCT as the serious search backend:
CPU/control code manages the tree, while the GPU batches ResNet policy/value
inference.

## Decision

Keep rollout MCTS for diagnostics, benchmarks, and comparison. Use the new PUCT
stack for AlphaZero self-play and training data once the ResNet evaluator and
tactical PUCT tests pass.
