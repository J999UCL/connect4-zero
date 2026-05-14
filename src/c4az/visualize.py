from __future__ import annotations

import argparse
import ast
import html
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from c4az.network import count_parameters, create_model


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = Path("docs/c4az_visual")
MODULE_ORDER = ("game", "network", "mcts", "selfplay", "data", "train", "arena", "cli")


@dataclass(frozen=True, slots=True)
class ClassInfo:
    module: str
    name: str
    bases: tuple[str, ...]
    fields: tuple[str, ...]
    methods: tuple[str, ...]
    annotations: tuple[str, ...]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate visual docs for the clean c4az AlphaZero stack.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args(argv)
    generate_visual_docs(args.out)
    return 0


def generate_visual_docs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = collect_classes(PACKAGE_ROOT)
    diagrams = {
        "system_flow.mmd": system_flow_diagram(),
        "module_dependencies.mmd": module_dependency_diagram(PACKAGE_ROOT),
        "class_uml.mmd": class_uml_diagram(classes),
        "puct_sequence.mmd": puct_sequence_diagram(),
        "resnet_architecture.mmd": resnet_architecture_diagram(),
        "data_lifecycle.mmd": data_lifecycle_diagram(),
    }
    for filename, diagram in diagrams.items():
        (output_dir / filename).write_text(diagram + "\n", encoding="utf-8")
    model_summary = model_parameter_summary()
    (output_dir / "model_summary.json").write_text(json.dumps(model_summary, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(render_html(diagrams, model_summary), encoding="utf-8")


def collect_classes(package_root: Path) -> list[ClassInfo]:
    classes: list[ClassInfo] = []
    for path in sorted(package_root.glob("*.py")):
        if path.name == "__init__.py":
            continue
        module = path.stem
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            fields: list[str] = []
            methods: list[str] = []
            annotations: list[str] = []
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    annotation = ast.unparse(item.annotation)
                    fields.append(f"{item.target.id}: {annotation}")
                    annotations.append(annotation)
                elif isinstance(item, ast.FunctionDef):
                    methods.append(_format_method(item))
            classes.append(
                ClassInfo(
                    module=module,
                    name=node.name,
                    bases=tuple(_base_name(base) for base in node.bases),
                    fields=tuple(fields),
                    methods=tuple(methods),
                    annotations=tuple(annotations),
                )
            )
    return classes


def module_dependency_diagram(package_root: Path) -> str:
    edges: set[tuple[str, str]] = set()
    modules = {path.stem for path in package_root.glob("*.py") if path.name != "__init__.py"}
    for path in package_root.glob("*.py"):
        source = path.stem
        if source == "__init__":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("c4az."):
                target = node.module.split(".")[1]
                if target in modules and target != source:
                    edges.add((source, target))
    lines = [
        "flowchart LR",
        "  classDef core fill:#edf7ff,stroke:#3778bf,color:#0f2742;",
        "  classDef search fill:#fff6dc,stroke:#c48a00,color:#4a3100;",
        "  classDef ml fill:#effbec,stroke:#4b9444,color:#173d14;",
        "  classDef data fill:#f7edff,stroke:#8a5cc2,color:#2c1747;",
    ]
    for module in sorted(modules, key=_module_sort_key):
        lines.append(f"  {module}[\"c4az.{module}\"]:::{_module_class(module)}")
    for source, target in sorted(edges):
        lines.append(f"  {source} --> {target}")
    return "\n".join(lines)


def class_uml_diagram(classes: list[ClassInfo]) -> str:
    known = {info.name for info in classes}
    lines = ["classDiagram", "  direction LR"]
    edges: set[str] = set()
    for info in sorted(classes, key=lambda item: (_module_sort_key(item.module), item.name)):
        lines.append(f"  class {info.name} {{")
        lines.append(f"    <<c4az.{info.module}>>")
        for field in info.fields[:8]:
            lines.append(f"    +{_clean_member(field)}")
        for method in info.methods[:8]:
            lines.append(f"    +{_clean_member(method)}")
        lines.append("  }")
        for base in info.bases:
            if base in known:
                edges.add(f"  {base} <|-- {info.name}")
        for annotation in info.annotations:
            for candidate in known:
                if candidate != info.name and candidate in annotation:
                    edges.add(f"  {info.name} --> {candidate}")
    lines.extend(sorted(edges))
    return "\n".join(lines)


def system_flow_diagram() -> str:
    return """flowchart TD
  Position["Position\nbitboards, heights, canonical side-to-move"]
  Model["AlphaZeroNet\npolicy logits + value"]
  Evaluator["TorchEvaluator\nbatched neural inference boundary"]
  MCTS["PUCTMCTS\nselection, expansion, backup"]
  Policy["SearchResult\nvisit-count policy target"]
  SelfPlay["Self-play game\nsample actions from pi"]
  Dataset["SelfPlayDataset\nclean npz shards + manifest"]
  Train["train_step\nCE(pi,p) + MSE(z,v)"]
  Arena["Arena\ncheckpoint-vs-checkpoint eval"]

  Position --> MCTS
  MCTS --> Evaluator
  Evaluator --> Model
  Model --> Evaluator
  Evaluator --> MCTS
  MCTS --> Policy
  Policy --> SelfPlay
  SelfPlay --> Dataset
  Dataset --> Train
  Train --> Model
  Model --> Arena
"""


def puct_sequence_diagram() -> str:
    return """sequenceDiagram
  participant Root as Root Node
  participant Tree as PUCTMCTS
  participant Leaf as Leaf Node
  participant Net as TorchEvaluator / AlphaZeroNet
  participant Backup as Backup

  Tree->>Root: ensure root expanded
  loop simulations_per_move
    Tree->>Tree: select argmax Q + c_puct * P * sqrt(N) / (1 + N_a)
    Tree->>Leaf: create child if edge is unexpanded
    alt terminal leaf
      Leaf-->>Tree: exact terminal_value
    else non-terminal leaf
      Tree->>Net: evaluate Position
      Net-->>Tree: priors P(s,·), value V(s)
      Tree->>Leaf: initialize legal priors
    end
    Tree->>Backup: backup value, flipping sign each edge
  end
  Tree-->>Root: pi from root visit counts
"""


def resnet_architecture_diagram() -> str:
    summary = model_parameter_summary()
    rows = "\\n".join(f"{item['preset']}: {item['parameters']:,} params" for item in summary)
    return f"""flowchart LR
  Input["Input\\n[B, 2, 4, 4, 4]\\ncurrent/opponent bitboard planes"]
  Stem["Stem\\nConv3d 3x3x3 + BN + ReLU"]
  Tower["Residual Tower\\n3/4/6 residual blocks\\n16/32/64 channels"]
  PolicyConv["Policy Head\\n1x1x1 Conv3d -> BN -> ReLU"]
  PolicyOut["Policy logits\\n[B, 16]"]
  ValueConv["Value Head\\n1x1x1 Conv3d -> BN -> ReLU"]
  ValueOut["Value\\n[B] in [-1, 1]"]
  Params["Preset sizes\\n{rows}"]

  Input --> Stem --> Tower
  Tower --> PolicyConv --> PolicyOut
  Tower --> ValueConv --> ValueOut
  Tower -.-> Params
"""


def data_lifecycle_diagram() -> str:
    return """flowchart LR
  Game["Self-play game"]
  Search["PUCT search per ply"]
  Sample["SelfPlaySample\ncurrent_bits, opponent_bits, pi, z, visits"]
  Shard["NPZ shard\ncolumnar arrays"]
  Manifest["manifest.json\nschema + shard records"]
  Loader["SelfPlayDataset"]
  Augment["8 symmetries\napplied at load time"]
  Batch["Training batch\ninput, policy, value"]

  Game --> Search --> Sample --> Shard
  Sample --> Manifest
  Shard --> Loader
  Manifest --> Loader
  Loader --> Augment --> Batch
"""


def model_parameter_summary() -> list[dict[str, int | str]]:
    return [
        {"preset": preset, "parameters": count_parameters(create_model(preset))}
        for preset in ("tiny", "small", "medium")
    ]


def render_html(diagrams: dict[str, str], model_summary: list[dict[str, int | str]]) -> str:
    generated = datetime.now(timezone.utc).isoformat()
    cards = []
    labels = {
        "system_flow.mmd": "System Flow",
        "module_dependencies.mmd": "Module Dependencies",
        "class_uml.mmd": "Class UML",
        "puct_sequence.mmd": "PUCT Sequence",
        "resnet_architecture.mmd": "3D ResNet Architecture",
        "data_lifecycle.mmd": "Data Lifecycle",
    }
    for filename, diagram in diagrams.items():
        cards.append(
            f"""
      <section class="diagram-block" id="{filename.removesuffix('.mmd')}">
        <div class="diagram-heading">
          <h2>{html.escape(labels[filename])}</h2>
          <a href="{html.escape(filename)}">Mermaid source</a>
        </div>
        <pre class="mermaid">{html.escape(diagram)}</pre>
      </section>
"""
        )
    model_rows = "\n".join(
        f"<tr><td>{item['preset']}</td><td>{item['parameters']:,}</td></tr>" for item in model_summary
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>c4az Visual Architecture</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #5c6b78;
      --line: #d9e0e7;
      --panel: #ffffff;
      --bg: #f5f7fa;
      --accent: #285e9e;
    }}
    body {{
      margin: 0;
      font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      padding: 28px 32px 18px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }}
    h1, h2 {{
      margin: 0;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    h1 {{
      font-size: 28px;
    }}
    h2 {{
      font-size: 19px;
    }}
    p {{
      max-width: 940px;
      color: var(--muted);
    }}
    main {{
      padding: 24px 32px 40px;
      display: grid;
      gap: 18px;
    }}
    nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 18px;
    }}
    nav a, .diagram-heading a {{
      color: var(--accent);
      text-decoration: none;
      border: 1px solid var(--line);
      background: #fff;
      padding: 6px 9px;
      border-radius: 6px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }}
    .panel, .diagram-block {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .diagram-heading {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin-top: 8px;
    }}
    th, td {{
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 6px 8px;
    }}
    .mermaid {{
      overflow: auto;
      margin: 0;
      background: #fff;
    }}
  </style>
</head>
<body>
  <header>
    <h1>c4az Visual Architecture</h1>
    <p>
      Generated from the clean-room AlphaZero package only. Legacy archived code
      and Rust experiments are intentionally excluded.
    </p>
    <nav>
      <a href="#system_flow">System</a>
      <a href="#module_dependencies">Modules</a>
      <a href="#class_uml">UML</a>
      <a href="#puct_sequence">PUCT</a>
      <a href="#resnet_architecture">ResNet</a>
      <a href="#data_lifecycle">Data</a>
    </nav>
  </header>
  <main>
    <section class="summary">
      <div class="panel">
        <h2>Scope</h2>
        <p>Only files under <code>src/c4az</code> are inspected.</p>
      </div>
      <div class="panel">
        <h2>Generated</h2>
        <p>{html.escape(generated)}</p>
      </div>
      <div class="panel">
        <h2>Model Sizes</h2>
        <table><tbody>{model_rows}</tbody></table>
      </div>
    </section>
    {''.join(cards)}
  </main>
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: true, theme: 'default', securityLevel: 'loose' }});
  </script>
</body>
</html>
"""


def _format_method(node: ast.FunctionDef) -> str:
    args = [arg.arg for arg in node.args.args if arg.arg != "self"]
    suffix = ""
    if node.returns is not None:
        suffix = f" -> {ast.unparse(node.returns)}"
    return f"{node.name}({', '.join(args)}){suffix}"


def _base_name(base: ast.expr) -> str:
    text = ast.unparse(base)
    return text.split(".")[-1]


def _clean_member(value: str) -> str:
    return (
        value.replace("->", ":")
        .replace("[", "(")
        .replace("]", ")")
        .replace("<", "")
        .replace(">", "")
        .replace("|", " or ")
        .replace(":", " :")
    )


def _module_class(module: str) -> str:
    if module == "game":
        return "core"
    if module in {"mcts", "selfplay", "arena"}:
        return "search"
    if module in {"network", "train"}:
        return "ml"
    return "data"


def _module_sort_key(module: str) -> tuple[int, str]:
    try:
        return MODULE_ORDER.index(module), module
    except ValueError:
        return len(MODULE_ORDER), module


if __name__ == "__main__":
    raise SystemExit(main())
