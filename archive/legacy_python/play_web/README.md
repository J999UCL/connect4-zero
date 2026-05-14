# Play Web App

Small local web app for playing 4x4x4 Connect Four against the current MCTS bot.

Run from the repo root:

```bash
uv run python play_web/server.py
```

Then open:

```text
http://127.0.0.1:8787
```

Useful options:

```bash
uv run python play_web/server.py --device mps --simulations 32 --rollout-batch-size 32
uv run python play_web/server.py --device cpu --simulations 64 --rollout-batch-size 64
```

The board is shown from the human perspective:

- blue stones are human stones
- red stones are bot stones
- action buttons and base squares choose `(x, y)` gravity columns
