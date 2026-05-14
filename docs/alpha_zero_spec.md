# Clean AlphaZero Specification

This implementation follows the published AlphaZero/AlphaGo Zero algorithm for
4x4x4 Connect Four. It is not a copy of DeepMind production code.

## State

- The game is represented canonically: `current` is always the side to move.
- A terminal value is always from the side-to-move perspective.
- If a move wins, the child has `terminal_value = -1.0` because the next side to
  move has lost.

## Search

- MCTS uses neural policy/value inference only. There are no random rollouts.
- Selection maximizes:

```text
Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

- `Q(s,a)` is from the parent perspective, so child mean values are sign-flipped.
- Terminal children bypass neural inference and back up exact values.
- Self-play adds Dirichlet noise at the root only.

## Training

- The policy target is generated from root visit counts.
- The value target is the final game outcome from each sampled position's
  side-to-move perspective.
- The loss is:

```text
cross_entropy(pi, p) + mse(z, v) + optimizer weight decay
```

## Verification

The test suite includes game legality/win tests, PUCT ledger tests, independent
reference-search differential tests, target-sign tests, dataset round trips, and
tiny end-to-end self-play/train/arena smoke coverage.
