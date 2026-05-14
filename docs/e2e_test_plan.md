# C4Zero Operational E2E Test Plan

This plan documents end-to-end operational suites for the active C4Zero runtime without adding production test harness code. Commands are shaped for an operator or CI job to run from the repository root.

The active package and CLI names used here are:

- C++ binary: `build/c4zero/c4zero`
- Python module path: `PYTHONPATH=src/python`
- Python console scripts from `pyproject.toml`: `c4zero-tools-version`, `c4zero-validate-dataset`, `c4zero-train`, `c4zero-export`, `c4zero-inspect-model`

Do not run the GPU nightly suite on a laptop or remote host until the operator has checked machine availability, free GPU, storage, and the intended working directory. The commands below do not assume a current remote machine, GPU id, dataset id, run id, or result.

## Common Setup

Use these setup commands before any suite that needs a fresh build or Python environment:

```bash
uv sync
cmake -S src/cpp -B build/c4zero \
  -DCMAKE_PREFIX_PATH="$PWD/.venv/lib/python3.11/site-packages/torch/share/cmake"
cmake --build build/c4zero
```

Expected setup artifacts:

- `build/c4zero/c4zero`
- CTest executables under `build/c4zero/`
- Python packages importable from `src/python`

## Suite 1: CPU Smoke

Purpose: prove that the current C++ binary, Python tools, self-play writer, dataset reader, trainer, exporter, and arena can form one local loop on CPU with tiny settings.

```bash
rm -rf runs/e2e/cpu-smoke checkpoints/e2e/cpu-smoke
mkdir -p runs/e2e checkpoints/e2e

build/c4zero/c4zero version --json
build/c4zero/c4zero bots
build/c4zero/c4zero botmatch --bot-a center --bot-b tactical --games 2

uv run c4zero-tools-version --json
uv run c4zero-export --preset tiny --out checkpoints/e2e/cpu-smoke/bootstrap.ts --device cpu

build/c4zero/c4zero selfplay \
  --model checkpoints/e2e/cpu-smoke/bootstrap.ts \
  --games 1 \
  --simulations 4 \
  --seed 101 \
  --device cpu \
  --out runs/e2e/cpu-smoke/selfplay

uv run c4zero-validate-dataset runs/e2e/cpu-smoke/selfplay/manifest.json

uv run c4zero-train \
  --preset tiny \
  --manifest runs/e2e/cpu-smoke/selfplay/manifest.json \
  --steps 1 \
  --seed 101 \
  --device cpu \
  --out checkpoints/e2e/cpu-smoke/trained

uv run c4zero-export \
  --checkpoint checkpoints/e2e/cpu-smoke/trained \
  --out checkpoints/e2e/cpu-smoke/trained/inference-reexport.ts \
  --device cpu

build/c4zero/c4zero arena \
  --model-a checkpoints/e2e/cpu-smoke/trained/inference.ts \
  --model-b checkpoints/e2e/cpu-smoke/trained/inference-reexport.ts \
  --games 2 \
  --simulations 4 \
  --seed 101 \
  --device cpu
```

Acceptance:

- Every command exits with status `0`.
- `c4zero-validate-dataset` prints JSON with `num_games: 1` and a positive `num_samples`.
- Training creates `checkpoints/e2e/cpu-smoke/trained/model_state.pt`, `metadata.json`, and `inference.ts`.
- Arena prints a match summary and does not throw a TorchScript or version compatibility error.

## Suite 2: Artifact Contract

Purpose: assert that self-play, checkpoint, and TorchScript artifacts carry the fields downstream jobs depend on.

```bash
test -f runs/e2e/cpu-smoke/selfplay/manifest.json
test -f runs/e2e/cpu-smoke/selfplay/shards/shard-000000.c4az
test -f checkpoints/e2e/cpu-smoke/trained/model_state.pt
test -f checkpoints/e2e/cpu-smoke/trained/metadata.json
test -f checkpoints/e2e/cpu-smoke/trained/inference.ts

uv run python - <<'PY'
import json
import struct
from pathlib import Path

manifest_path = Path("runs/e2e/cpu-smoke/selfplay/manifest.json")
manifest = json.loads(manifest_path.read_text())
for key in ["schema_version", "num_games", "num_samples", "model_checkpoint", "shard_paths", "config", "version"]:
    assert key in manifest, key
for key in ["dataset_schema_version", "game_rules_version", "encoder_version", "action_mapping_version", "symmetry_version"]:
    assert key in manifest["version"], key

shard_path = manifest_path.parent / manifest["shard_paths"][0]
magic, major, minor, sample_count = struct.unpack_from("<8sIIQ", shard_path.read_bytes(), 0)
assert magic == b"C4AZSP01", magic
assert major == 1, major
assert minor == 0, minor
assert sample_count == manifest["num_samples"], (sample_count, manifest["num_samples"])

metadata = json.loads(Path("checkpoints/e2e/cpu-smoke/trained/metadata.json").read_text())
for key in ["version", "model_config", "model_config_hash", "step", "epoch", "replay_manifests", "metrics", "export_schema"]:
    assert key in metadata, key
assert metadata["export_schema"]["input"] == "float32[B,2,4,4,4]"
assert metadata["export_schema"]["policy_logits"] == "float32[B,16]"
assert metadata["export_schema"]["value"] == "float32[B]"
print("artifact contract ok")
PY

uv run python - <<'PY'
import torch
model = torch.jit.load("checkpoints/e2e/cpu-smoke/trained/inference.ts", map_location="cpu")
policy, value = model(torch.zeros((1, 2, 4, 4, 4), dtype=torch.float32))
assert tuple(policy.shape) == (1, 16), tuple(policy.shape)
assert tuple(value.shape) == (1,), tuple(value.shape)
print("torchscript contract ok")
PY
```

Acceptance:

- Required files exist at the expected paths.
- Manifest fields match the active shard and version contract.
- TorchScript accepts `float32[1,2,4,4,4]` and returns policy logits `float32[1,16]` plus value `float32[1]`.

## Suite 3: Deterministic CPU

Purpose: catch nondeterministic self-play output for fixed evaluator, seed, game count, and simulation count.

```bash
rm -rf runs/e2e/deterministic
mkdir -p runs/e2e/deterministic

for run in a b; do
  build/c4zero/c4zero selfplay \
    --games 2 \
    --simulations 8 \
    --seed 2026 \
    --device cpu \
    --out "runs/e2e/deterministic/$run"
  uv run c4zero-validate-dataset "runs/e2e/deterministic/$run/manifest.json"
done

shasum -a 256 runs/e2e/deterministic/a/shards/shard-000000.c4az \
  | sed 's#runs/e2e/deterministic/a#RUN#' \
  > runs/e2e/deterministic/a.sha256
shasum -a 256 runs/e2e/deterministic/b/shards/shard-000000.c4az \
  | sed 's#runs/e2e/deterministic/b#RUN#' \
  > runs/e2e/deterministic/b.sha256
diff -u runs/e2e/deterministic/a.sha256 runs/e2e/deterministic/b.sha256

uv run python - <<'PY'
import json
from pathlib import Path

a = json.loads(Path("runs/e2e/deterministic/a/manifest.json").read_text())
b = json.loads(Path("runs/e2e/deterministic/b/manifest.json").read_text())
for key in ["schema_version", "num_games", "num_samples", "model_checkpoint", "config", "version"]:
    assert a[key] == b[key], key
print("deterministic manifest contract ok")
PY
```

Acceptance:

- Both generated manifests validate.
- Shard hashes match exactly after normalizing only the run directory in the checksum text.
- Manifest semantic fields match exactly.

## Suite 4: Adversarial Contracts

Purpose: confirm operational failure modes are loud when artifacts are missing, corrupted, or version-incompatible.

```bash
rm -rf runs/e2e/adversarial
mkdir -p runs/e2e/adversarial
cp -R runs/e2e/cpu-smoke/selfplay runs/e2e/adversarial/good

uv run python - <<'PY'
import json
from pathlib import Path

root = Path("runs/e2e/adversarial")
good = json.loads((root / "good/manifest.json").read_text())

missing = dict(good)
missing.pop("version")
(root / "missing-version.json").write_text(json.dumps(missing, indent=2))

bad_schema = dict(good)
bad_schema["schema_version"] = "999.0"
(root / "bad-schema.json").write_text(json.dumps(bad_schema, indent=2))

bad_shard = root / "bad-shard.c4az"
bad_shard.write_bytes(b"not-a-c4zero-shard")
bad_shard_manifest = dict(good)
bad_shard_manifest["shard_paths"] = ["bad-shard.c4az"]
bad_shard_manifest["num_samples"] = 1
(root / "bad-shard.json").write_text(json.dumps(bad_shard_manifest, indent=2))
PY

! uv run c4zero-validate-dataset runs/e2e/adversarial/missing-version.json
! uv run c4zero-validate-dataset runs/e2e/adversarial/bad-schema.json
! uv run c4zero-validate-dataset runs/e2e/adversarial/bad-shard.json

! uv run c4zero-export \
  --checkpoint runs/e2e/adversarial/does-not-exist \
  --out runs/e2e/adversarial/should-not-exist.ts

! build/c4zero/c4zero arena \
  --model-a runs/e2e/adversarial/does-not-exist-a.ts \
  --model-b runs/e2e/adversarial/does-not-exist-b.ts \
  --games 1 \
  --simulations 1 \
  --device cpu
```

Acceptance:

- Every command prefixed with `!` exits non-zero.
- Failures identify the missing or invalid contract, rather than succeeding with empty or fallback data.
- No command writes outside `runs/e2e/adversarial`.

## Suite 5: GPU Nightly

Purpose: run the same operational path at a size more representative of nightly training, using CUDA for model export, self-play inference, training, and arena evaluation.

Operator preflight, to run manually on the selected GPU host:

```bash
pwd
git status --short
nvidia-smi
df -h .
uv sync
cmake -S src/cpp -B build/c4zero \
  -DCMAKE_PREFIX_PATH="$PWD/.venv/lib/python3.11/site-packages/torch/share/cmake"
cmake --build build/c4zero
```

GPU nightly command sequence:

```bash
rm -rf runs/e2e/gpu-nightly checkpoints/e2e/gpu-nightly
mkdir -p runs/e2e/gpu-nightly checkpoints/e2e/gpu-nightly

uv run c4zero-export \
  --preset small \
  --out checkpoints/e2e/gpu-nightly/bootstrap.ts \
  --device cuda

build/c4zero/c4zero selfplay \
  --model checkpoints/e2e/gpu-nightly/bootstrap.ts \
  --games 32 \
  --simulations 128 \
  --seed 4242 \
  --device cuda \
  --out runs/e2e/gpu-nightly/selfplay

uv run c4zero-validate-dataset runs/e2e/gpu-nightly/selfplay/manifest.json

uv run c4zero-train \
  --preset small \
  --manifest runs/e2e/gpu-nightly/selfplay/manifest.json \
  --steps 100 \
  --seed 4242 \
  --device cuda \
  --out checkpoints/e2e/gpu-nightly/trained

uv run c4zero-export \
  --checkpoint checkpoints/e2e/gpu-nightly/trained \
  --out checkpoints/e2e/gpu-nightly/trained/inference-reexport.ts \
  --device cuda

build/c4zero/c4zero arena \
  --model-a checkpoints/e2e/gpu-nightly/bootstrap.ts \
  --model-b checkpoints/e2e/gpu-nightly/trained/inference.ts \
  --games 64 \
  --simulations 128 \
  --seed 4242 \
  --device cuda
```

Acceptance:

- Preflight records the selected working directory, git status, GPU visibility, and available filesystem space before the run.
- Self-play writes one manifest and at least one shard under `runs/e2e/gpu-nightly/selfplay`.
- Training writes `model_state.pt`, `metadata.json`, and `inference.ts` under `checkpoints/e2e/gpu-nightly/trained`.
- Arena completes with no CUDA, TorchScript, or version compatibility error.
- The operator records real runtime, GPU, storage, and result details in the project notebook after execution.

## Suite Ownership

| Suite | Cadence | Owner | Runs GPU or remote work |
| --- | --- | --- | --- |
| CPU Smoke | per PR or before handoff | local operator or CI | no |
| Artifact Contract | per PR or release candidate | local operator or CI | no |
| Deterministic CPU | per PR touching search/data/version behavior | local operator or CI | no |
| Adversarial Contracts | per PR touching readers/loaders/export paths | local operator or CI | no |
| GPU Nightly | nightly or before long training cycles | Jeet/operator | yes, manual only |
