# Connect4 Zero

AlphaZero-style reinforcement learning for gravity-based 4x4x4 Connect Four.

The project currently has a clean, vectorized PyTorch game engine, a reference
classical MCTS implementation, and a batched root/action search path for
generating AlphaZero-style seed data. Neural network training and remote GPU
experiment scripts come later.

## Local Setup

Use `uv` with Python 3.11:

```bash
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"
uv run pytest
```

The local Mac setup is for correctness tests and small smoke checks. Heavy
self-play and training runs should happen on the remote NVIDIA GPU machines.

## Visual Docs

A static visual walkthrough lives at:

```text
docs/site/index.html
```

Open that file in a browser to see the board model, engine tensors, object
relationships, MCTS flow, value signs, and current test coverage.

## Play Against MCTS

Run the small local web app:

```bash
uv run python play_web/server.py
```

Then open:

```text
http://127.0.0.1:8787
```

The app shows a rotatable 3D board and lets you play blue stones against the
red MCTS bot.

## Game Rules

- Board size: `4 x 4 x 4`.
- Gravity-based action space: players choose one of 16 `(x, y)` columns.
- A piece lands at the next open `z` index in that column.
- Coordinates are ordered as `(x, y, z)`.
- `z = 0` is the bottom of a column.
- A player wins by completing any length-4 line in 3D.

## Engine State

The main engine class is `Connect4x4x4Batch`.

```python
from connect4_zero.game import Connect4x4x4Batch

game = Connect4x4x4Batch(batch_size=1024)
```

It stores many independent games at once:

- `board`: shape `(B, 4, 4, 4)`, dtype `torch.int8`
- `heights`: shape `(B, 4, 4)`, dtype `torch.int8`
- `done`: shape `(B,)`, dtype `torch.bool`
- `outcome`: shape `(B,)`, dtype `torch.int8`

Board values:

- `1`: player to move
- `-1`: opponent
- `0`: empty

The board is canonicalized after every legal non-terminal move: the engine
multiplies that game's board by `-1`, so the next player also sees themselves
as `1`.

## Action Mapping

There are 16 actions:

```text
action = 4 * x + y
```

Examples:

```text
0  -> (x=0, y=0)
15 -> (x=3, y=3)
```

## Step Results And Outcome Convention

`game.step(actions)` applies one action per game and returns a `StepResult`.

```python
result = game.step(actions)
```

Fields:

- `legal`: whether each attempted move was legal
- `won`: whether each legal move won immediately
- `draw`: whether each legal move filled the board without a win
- `done`: whether each game is now terminal
- `outcome`: terminal result stored by the engine

Important convention:

```text
outcome = 1 means the player who just moved won.
outcome = 0 means unfinished or draw.
```

It does **not** mean "the current player after the step won." This distinction
matters because non-terminal states are perspective-flipped after each move.

## Tests

Run:

```bash
uv run pytest
```

The current tests cover:

- action mapping
- all 76 generated winning lines
- symmetry permutations
- gravity and full-column legality
- canonical perspective flips
- win/draw detection
- mixed batched game behavior
- cloning and device movement

## MCTS

The Phase 2 search code lives in `connect4_zero.search`.

```python
from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.search import MCTS, MCTSConfig

game = Connect4x4x4Batch(batch_size=1)
mcts = MCTS(MCTSConfig(num_simulations=100, rollout_batch_size=128))
result = mcts.search(game)
```

`result.policy` is a length-16 visit-count policy suitable as an AlphaZero-style
training target. Illegal actions have probability `0`.

Value convention:

- Node values are from the player-to-move perspective at that node.
- Child values are negated when viewed from the parent.
- A terminal child where the previous mover won has value `-1` from the child
  perspective.

The first `NodeStore` implementation is a plain tree. MCTS uses the store
boundary so a future Zobrist-backed DAG can replace tree storage without
rewriting selection, evaluation, or backpropagation.

## Batched Data Generation

Phase 2.5 adds a separate high-throughput search path for seed data:

```python
from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.search import BatchedRootActionConfig, BatchedRootActionMCTS

roots = Connect4x4x4Batch(batch_size=1024)
search = BatchedRootActionMCTS(
    BatchedRootActionConfig(
        num_selection_waves=8,
        leaves_per_root=4,
        rollouts_per_leaf=64,
        rollout_device="cuda",
    )
)
result = search.search_batch(roots)
```

This combines root parallelism and leaf/action parallelism:

```text
B roots x selected actions x random rollout continuations
```

The self-play data utilities live in `connect4_zero.data`. Generated samples are
stored as safetensor shards plus a JSONL manifest:

```text
boards, policies, values, visit_counts, q_values, legal_masks, actions, plies
```

The neural network only needs `boards`, `policies`, and `values`; the extra
search stats are kept for debugging, filtering, and analysis.

### Verbose Generation Scripts

Benchmark the batched search path:

```bash
connect4-benchmark-search \
  --device cuda \
  --batch-size 1024 \
  --iterations 20 \
  --warmup 3 \
  --rollouts-per-leaf 64 \
  --num-selection-waves 8 \
  --log-dir /tmp/thakwani/rl-runs/benchmarks
```

Generate seed self-play shards:

```bash
connect4-generate-selfplay \
  --device cuda \
  --games 10000 \
  --batch-size 1024 \
  --games-per-write 1024 \
  --rollouts-per-leaf 64 \
  --num-selection-waves 8 \
  --samples-per-shard 32768 \
  --out /tmp/thakwani/rl-data/seed-v1
```

Both commands log live progress to stdout and write detailed timestamped logs
under the selected output/log directory. On UCL `csh` machines, set caches and
outputs away from home quota first:

```csh
setenv RL_DATA /tmp/thakwani/rl-data
setenv RL_RUNS /tmp/thakwani/rl-runs
setenv UV_CACHE_DIR /tmp/thakwani/uv-cache
setenv PIP_CACHE_DIR /tmp/thakwani/pip-cache
setenv TORCH_HOME /tmp/thakwani/torch-cache
setenv XDG_CACHE_HOME /tmp/thakwani/cache
```
