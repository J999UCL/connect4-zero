# Stage 0 Supervised Training

This package is for policy-only supervised pretraining on the synthetic Stage 0
curriculum datasets.

It is deliberately separate from the AlphaZero replay trainer:

- Stage 0 uses explicit train/validation/test manifests.
- Training iterates shuffled epochs instead of sampling with replacement.
- The forward pass uses only the shared trunk and policy head.
- The value loss is not used, and the value head is frozen by default.

Example remote command shape:

```bash
python -m c4zero_supervised.stage0_train \
  --preset small \
  --train-manifest /tmp/thakwani/rl-data/curriculum/stage0-v1/manifest.json \
  --val-manifest /tmp/thakwani/rl-data/curriculum/stage0-v1-val/manifest.json \
  --batch-size 2048 \
  --epochs 3 \
  --log-every-steps 10 \
  --eval-every-steps 50 \
  --device cuda \
  --out /tmp/thakwani/rl-runs/stage0-small
```

Do not pass the test split to training. Test is reserved for a final explicit
evaluation after checkpoint/model-selection decisions are frozen:

```bash
python -m c4zero_supervised.stage0_eval \
  --checkpoint /tmp/thakwani/rl-runs/stage0-small \
  --manifest /tmp/thakwani/rl-data/curriculum/stage0-v1-test/manifest.json \
  --batch-size 2048 \
  --device cuda \
  --out /tmp/thakwani/rl-runs/stage0-small/test-final.json
```
