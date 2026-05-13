## Project Goal

Implement Alphazero style RL for a 3d Connect 4 game (4 by 4 by 4)

## Core Rules
- Answer questions directly, if there is a question in users response do not make any actions you may view and read files but only answer.
- Inspection is allowed without extra permission: read files, search the repo, inspect notebook history, and read docs.
- Do not edit files, write scripts, or execute project scripts, data-processing jobs, training jobs, remote sync commands, or experiment runs until Jeet explicitly says to.
- Jeet runs final experiment commands unless he explicitly delegates that work.
- Never invent metrics, results, dataset IDs, run metadata, or remote machine state.
- Never do a git Push without permission.

## How To Work Here
- For open-ended implementation work, Jeet decides the approach. For direct factual questions, answer first and teach second.
- Use `$gpu-run-prep` whenever work involves remote machines, GPUs, storage checks, syncing files, remote paths, or preparing experiment commands.
- Use the lab notebook as the project bookkeeping system: check history before resuming, log real processing/datasets/runs/takeaways after they happen, and use mirror files only as read-only fallback.
- Keep code simple and easy to explain. Avoid unnecessary abstractions.

## Verification
- Use the smallest relevant verification that matches the change.
- Docs-only changes do not need code checks.
- When you change Python files but cannot run the real project scripts under the repo rules, prefer non-execution checks such as `python -m py_compile` on the touched files.
- There is no general lint/build harness checked into this repo right now. Do not invent one.
- In the final response, say what you verified and what you did not run.

## Done Means
- The user’s request is satisfied.
- The repo constraints above were respected.
- Relevant checks were run when allowed, or skipped with a clear reason.

## Reference Docs
- `markdown/project_context.md`: project scope, repo map, and notebook memory layout
