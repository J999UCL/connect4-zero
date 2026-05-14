# C4Zero AlphaZero Specification

The active implementation is split across C++ and Python/PyTorch. C++ owns game
rules, PUCT MCTS, self-play, arena, and TorchScript inference. Python/PyTorch
owns the canonical model definition, training loop, checkpointing, replay
loading, and TorchScript export. Archived Python/Rust prototypes are reference
material only. This project follows the published AlphaZero/AlphaGo Zero
algorithm for 4x4x4 Connect Four; it is not a copy of DeepMind production code.

## State

- The game is represented as bitboards: `current` and `opponent`.
- The game is represented canonically: `current` is always the side to move.
- A terminal value is always from the side-to-move perspective.
- If a move wins, the child has `terminal_value = -1.0` because the next side to
  move has lost.

## Search

- MCTS uses neural policy/value inference only. There are no random rollouts.
- Production selection maximizes:

```text
Q(s,a) + C(s) * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
C(s) = log((1 + N(s) + 19652) / 19652) + 1.25
```

- `Q(s,a)` is from the parent perspective, so child mean values are sign-flipped.
- Terminal children bypass neural inference and back up exact values.
- Self-play adds Dirichlet noise at the root only.

## Training

- The policy target is generated from root visit counts.
- The value target is the final game outcome from each sampled position's
  side-to-move perspective.
- Python samples uniformly over positions from the most recent replay-window
  games written by C++.
- Training uses SGD with momentum and L2 regularization. The loss is:

```text
(z - v)^2 - pi^T log(p) + 1e-4 ||theta||^2
```

## Inference Boundary

- Python exports `inference.ts` from the PyTorch model in `eval()` mode.
- C++ loads `inference.ts` with libtorch JIT once, then runs inference inside
  MCTS without calling Python.
- TorchScript input is `float32[B,2,4,4,4]`.
- TorchScript output is `(policy_logits float32[B,16], value float32[B])`.
- C++ masks illegal actions and softmaxes logits after inference.

## Versioning

- `version_manifest.json` is the canonical schema/rules/model version source.
- C++ embeds the manifest into `c4zero::version`.
- Python inspection tools read the same manifest.
- Dataset and checkpoint readers reject incompatible game rules, encoder,
  action mapping, model config, or major schema versions.

## Verification

The test suite includes game legality/win tests, PUCT ledger tests, independent
reference-search differential tests, target-sign tests, dataset round trips, and
tiny end-to-end self-play/train/arena smoke coverage.
