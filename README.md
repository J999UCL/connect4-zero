# C4Zero

Clean AlphaZero for gravity-based 4x4x4 Connect Four.

The active runtime is split intentionally:

- C++ owns game rules, PUCT MCTS, neural arena, heuristic bot baselines, self-play, and
  TorchScript inference during search.
- Python/PyTorch owns model definition, training, checkpointing, replay loading,
  and TorchScript export.
- Python inspection tools live under `src/python/c4zero_tools`; training lives
  under `src/python/c4zero_train`.

Previous implementations are archived for reference only:

- `archive/legacy_python`
- `archive/python_c4az_2026_05_14`
- `archive/rust_prototype_2026_05_14`

Active code must not import or link against archived code.

## C++ Build

The C++ build uses CMake, C++17, and libtorch:

```bash
uv sync
cmake -S src/cpp -B build/c4zero \
  -DCMAKE_PREFIX_PATH="$PWD/.venv/lib/python3.11/site-packages/torch/share/cmake"
cmake --build build/c4zero
ctest --test-dir build/c4zero --output-on-failure
```

This machine currently has PyTorch CMake files in the venv, but local `cmake`
may need to be installed before the full build can run.

## CLI Shape

After building:

```bash
build/c4zero/c4zero version --json
build/c4zero/c4zero bots
build/c4zero/c4zero botmatch --bot-a center --bot-b tactical --games 20
build/c4zero/c4zero arena --model-a checkpoints/a/inference.ts --model-b checkpoints/b/inference.ts --games 20 --simulations 800
build/c4zero/c4zero selfplay --model checkpoints/current/inference.ts --games 2 --simulations 32 --out runs/c4zero-smoke
```

## Python Training And Tools

```bash
PYTHONPATH=src/python python -m c4zero_tools.version --json
PYTHONPATH=src/python python -m c4zero_tools.datasets runs/c4zero-smoke/manifest.json
PYTHONPATH=src/python python -m c4zero_train.cli --preset tiny --manifest runs/c4zero-smoke/manifest.json --steps 1 --out checkpoints/smoke
```

## Algorithm Contract

The active stack follows AlphaZero mechanics:

- bitboard 4x4x4 Connect Four game core
- canonical side-to-move perspective
- PyTorch 3D ResNet policy/value network
- TorchScript export loaded by C++ for MCTS inference
- PUCT MCTS with neural leaf evaluation
- no random rollouts in the active AlphaZero path
- exact terminal values
- self-play targets from root visit counts
- final outcome value targets
- replay-window sampling by recent games
- SGD with momentum and L2 weight decay
- root Dirichlet noise during self-play only
- deterministic checkpoint arena/eval by default

Version compatibility is centralized in `version_manifest.json`. Datasets,
checkpoints, and run manifests must carry the version snapshot so schema,
encoder, action mapping, and game-rule drift fails loudly.
