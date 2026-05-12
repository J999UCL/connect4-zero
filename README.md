# Connect4 Zero

AlphaZero-style reinforcement learning for gravity-based 4x4x4 Connect Four.

The project is currently in Phase 1: a clean, vectorized PyTorch game engine.
MCTS, neural network training, self-play, and remote GPU runs come later.

## Local Setup

Use `uv` with Python 3.11:

```bash
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"
uv run pytest
```

The local Mac setup is for correctness tests and small smoke checks. Heavy
self-play and training runs should happen on the remote NVIDIA GPU machines.

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
