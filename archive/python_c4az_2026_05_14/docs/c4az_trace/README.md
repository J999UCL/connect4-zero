# c4az Debug Trace Views

This folder holds debugger-style visualizations for the clean-room `src/c4az`
AlphaZero implementation.

Generate both views:

```bash
uv run c4az-generate-debug-traces
```

Open:

- `docs/c4az_trace/index.html` for the landing page.
- `docs/c4az_trace/viztracer/puct_demo.html` for the real execution timeline.
- `docs/c4az_trace/lectrace_site/index.html` for the step-through lecture view.

For live lectrace stepping while editing:

```bash
uv run lectrace serve docs/c4az_trace/lectures/01_puct_walkthrough.py --port 7000
```

The demo is intentionally tiny and deterministic. It exists to explain how
`Position`, `PUCTMCTS`, `DemoEvaluator`, and the visit-count policy target move
together, not to benchmark training throughput.
