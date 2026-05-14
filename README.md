# Connect4 AlphaZero

Clean-room AlphaZero for gravity-based 4x4x4 Connect Four.

The active implementation lives in `src/c4az`. Legacy Python code, old reports,
and old tests are archived under `archive/legacy_python` for reference only.
The active path does not import the legacy `connect4_zero` package.

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"
uv run pytest
```

## Active Commands

```bash
c4az-selfplay
c4az-train
c4az-loop
c4az-arena
c4az-inspect-checkpoint
c4az-generate-diagrams
```

Use `--help` on any command for arguments.

## Algorithm

The active stack is paper-faithful AlphaZero:

- bitboard 4x4x4 Connect Four game core
- canonical side-to-move perspective
- neural policy/value model
- PUCT MCTS with no random rollouts
- exact terminal values
- self-play policy targets from root visit counts
- final outcome value targets
- loss: policy cross entropy + value MSE + optimizer weight decay
- root Dirichlet noise during self-play only
- deterministic arena/eval by default

See `docs/alpha_zero_spec.md` for the implementation contract and verification
strategy.

## Visual Architecture

Generate the local visual walkthrough:

```bash
uv run c4az-generate-diagrams
```

Then open:

```text
docs/c4az_visual/index.html
```

The generator inspects only `src/c4az` and emits Mermaid diagrams for the
system flow, module dependencies, class UML, PUCT search, ResNet architecture,
and data lifecycle.

## Model Presets

| Preset | Blocks | Channels | Params | Use |
|---|---:|---:|---:|---|
| `tiny` | 3 | 16 | 46,791 | tests and smoke runs |
| `small` | 4 | 32 | 229,879 | default serious model |
| `medium` | 6 | 64 | 1,338,711 | ablation |

The default is `small`, a much leaner 3D ResNet than the old 1.34M parameter
starting point.

## Verification

Run:

```bash
uv run pytest
```

The new suite checks game rules, all 76 win masks, symmetries, model shapes and
parameter counts, PUCT ledger behavior, an independent reference PUCT
differential, tactical win/block behavior, data round trips, checkpointing, and
a tiny end-to-end self-play/train/arena smoke.
