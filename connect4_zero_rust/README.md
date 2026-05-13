# connect4_zero_rust

Rust self-play and evaluation engine for 4x4x4 Connect Four.

This workspace is a clean restart of the scaling-critical parts of the Python project. Python remains the training environment for PyTorch. Rust owns fast game simulation, PUCT tree search, arena evaluation, and dataset writing.

## Workspace Layout

```text
connect4_zero_rust/
  Cargo.toml
  docs/
    architecture.md
  crates/
    c4-core/      # Game state, actions, geometry, win masks, symmetries.
    c4-search/    # PUCT tree, node arena, search configs, search diagnostics.
    c4-infer/     # Neural evaluator trait and model backends.
    c4-data/      # Self-play sample schema and shard writing/reading.
    c4-arena/     # Checkpoint-vs-checkpoint match orchestration.
    c4-cli/       # Thin command entrypoints: selfplay, arena, inspect.
```

## Boundary Decisions

- `c4-core` has no ML concepts. It should be deterministic, tiny, and heavily tested.
- `c4-search` depends on `c4-core` and `c4-infer`, but not on data writing or CLIs.
- `c4-infer` hides whether evaluation runs on ONNX CPU, ONNX CUDA, or a test stub.
- `c4-data` owns the file format. Search should return values; data code decides how to serialize them.
- `c4-cli` contains no core logic. Binaries should parse flags, build configs, and call library crates.

## Implemented Baseline

The workspace now has an end-to-end baseline:

- bitboard game state with exact 76-line win detection
- canonical current-player perspective
- PUCT search with tree reuse for self-play
- uniform evaluator for tests
- ONNX Runtime CPU evaluator for exported PyTorch models
- Rust self-play CLI writing custom zstd-compressed binary shards
- Rust arena CLI with paired random openings
- Python loader for Rust shards

## Smoke Commands

```bash
cargo test --workspace
cargo run --bin c4-selfplay -- --uniform --games 4 --workers 2 --simulations-per-move 16 --out /tmp/c4-rust-smoke
cargo run --bin c4-arena -- --uniform --games 4 --workers 2 --simulations-per-move 8 --opening-plies 2 --paired-openings
```

Export a Python model and run Rust self-play:

```bash
uv run python -m connect4_zero.scripts.export_onnx --out /tmp/connect4.onnx --device cpu
cd connect4_zero_rust
cargo run --bin c4-selfplay -- --model /tmp/connect4.onnx --games 4 --workers 2 --simulations-per-move 32 --out /tmp/c4-rust-onnx
```
