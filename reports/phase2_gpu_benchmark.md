# Deprecated Phase 2 Shallow Search GPU Benchmark Report

> Deprecated: this report measured the old root/action evaluator. That search
> only evaluated root actions explicitly and is not suitable as final AlphaZero
> seed data. Keep these numbers only as historical CUDA rollout-throughput
> measurements.

Date: 2026-05-12  
Machine: `canada-l.cs.ucl.ac.uk`  
GPU: NVIDIA GeForce RTX 3090 Ti, 24GB  
Code commit: `2f51501`

## Summary

The batched root/action rollout path is working well on CUDA. The most important
optimization was increasing `max_rollouts_per_chunk` from `65,536` to `262,144`,
which raised estimated rollout throughput from about `451k` to `535k` rollout
games/sec.

Historical best shallow config:

```text
batch_size=1024
rollouts_per_leaf=64
num_selection_waves=4
leaves_per_root=4
max_rollouts_per_chunk=262144
```

This keeps the same throughput as the larger `2048` batch run while giving
cleaner progress, lower CPU-side accumulation, and safer chunk boundaries.

## Benchmarks

All benchmark rows used CUDA on `canada-l`.

| Run | Batch | Rollouts/Leaf | Waves | Leaves/Root | Max Rollouts/Chunk | Roots/sec | Visits/sec | Est. Rollout Games/sec | Peak CUDA Allocated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 1024 | 64 | 8 | 4 | 65,536 | 146.74 | 7,043.4 | 450,776.9 | 223.6 MB |
| Bigger chunk | 1024 | 64 | 8 | 4 | 262,144 | 174.07 | 8,355.5 | 534,751.6 | 883.0 MB |
| Recommended | 1024 | 64 | 4 | 4 | 262,144 | 261.28 | 8,360.8 | 535,092.5 | 883.0 MB |
| Faster/noisier | 1024 | 32 | 4 | 4 | 262,144 | 507.28 | 16,232.8 | 519,450.0 | 883.3 MB |
| Larger batch | 2048 | 64 | 4 | 4 | 262,144 | 261.57 | 8,370.3 | 535,700.5 | 886.8 MB |

## Interpretation

The larger chunk size is the clear win. With `rollouts_per_leaf=64`, moving from
`65,536` to `262,144` rollouts per chunk reduces Python/GPU chunk overhead:

```text
65,536 / 64 = 1,024 leaf states per chunk
262,144 / 64 = 4,096 leaf states per chunk
```

For the recommended config:

```text
visits per root = 16 initial actions + 4 waves * 4 leaves = 32
rollout games per root = 32 * 64 = 2,048
```

At about `535k` rollout games/sec:

```text
535,092 / 2,048 ~= 261 roots/sec
```

The `2048` batch run did not materially improve throughput. It doubled the work
per iteration and roughly doubled the iteration time, leaving rollout-games/sec
basically unchanged. For the first real dataset run, `1024` is cleaner and safer.

The `rollouts_per_leaf=32` run is much faster in roots/sec, but it produces
noisier action-value estimates. It may be useful for quick exploratory data, but
`64` is a better first seed-data setting.

## Deprecated Command

```csh
cd /tmp/thakwani/connect4-zero
source /opt/Python/Python-3.11.5_Setup.csh
setenv PYTHONPATH /tmp/thakwani/connect4-zero/src
setenv RL_DATA /tmp/thakwani/rl-data
setenv RL_RUNS /tmp/thakwani/rl-runs
setenv PIP_CACHE_DIR /tmp/thakwani/pip-cache
setenv XDG_CACHE_HOME /tmp/thakwani/cache
setenv TORCH_HOME /tmp/thakwani/torch-cache

python3 -m connect4_zero.scripts.generate_selfplay \
  --device cuda \
  --games 10000 \
  --batch-size 1024 \
  --games-per-write 1024 \
  --rollouts-per-leaf 64 \
  --num-selection-waves 4 \
  --leaves-per-root 4 \
  --max-rollouts-per-chunk 262144 \
  --samples-per-shard 32768 \
  --out /tmp/thakwani/rl-data/seed-v1
```

## Expected Runtime

The benchmark measures root-search throughput, not full self-play runtime. A
rough estimate for the recommended config is:

```text
~261 root positions/sec
```

If games average:

```text
40 plies -> about 6.5 games/sec -> 10k games in ~26 minutes
45 plies -> about 5.8 games/sec -> 10k games in ~29 minutes
50 plies -> about 5.2 games/sec -> 10k games in ~32 minutes
```

The real generator should be run once on a smaller smoke dataset first because
active-game counts, shard writing, and terminal positions change the exact
runtime.

## Next Step

Run a small real generation smoke test before the full dataset:

```csh
python3 -m connect4_zero.scripts.generate_selfplay \
  --device cuda \
  --games 128 \
  --batch-size 128 \
  --games-per-write 128 \
  --rollouts-per-leaf 64 \
  --num-selection-waves 4 \
  --leaves-per-root 4 \
  --max-rollouts-per-chunk 262144 \
  --samples-per-shard 4096 \
  --out /tmp/thakwani/rl-data/smoke-v1
```

If that writes and verifies cleanly, proceed with the `10,000` game seed run.
