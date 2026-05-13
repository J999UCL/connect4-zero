# Rust Architecture

## Why This Exists

The Python prototype proved the rules, model shape, data schema, and arena workflow. It also showed the scaling bottleneck: self-play spends too much time in Python tree management. The Rust workspace restarts that part with simpler ownership and predictable performance.

## Crates

### `c4-core`

Owns only the game:

- board representation
- action mapping
- legal move generation
- stepping a move
- terminal detection
- symmetry tables

This crate should stay dependency-light and serve as the test oracle for everything above it.

### `c4-infer`

Owns policy/value inference behind a trait:

```rust
Evaluator::evaluate(&[Board]) -> Vec<Evaluation>
```

Implemented backends:

- uniform/random evaluator for tests
- ONNX Runtime CPU evaluator for real self-play

CUDA inference can be added later without changing search.

### `c4-search`

Owns PUCT:

- compact node arena
- tree reuse after moves
- root noise
- action temperature
- search diagnostics

It should not write files and should not know about training epochs.

### `c4-data`

Owns self-play samples:

- encoded board planes
- visit-count policy
- value target
- legal mask
- metadata

Rust writes a JSON manifest plus zstd-compressed binary shards. Python reads this through `RustBinarySelfPlayDataset`.

### `c4-arena`

Owns evaluation matches:

- paired random openings
- deterministic or noisy search
- alternating starts
- summary metrics

The arena should be strict about separating strength evaluation from stochastic training self-play.

### `c4-cli`

Owns command-line entrypoints only:

- `c4-selfplay`
- `c4-arena`
- `c4-inspect`

If a function is hard to test because it lives in `c4-cli`, it belongs in another crate.
