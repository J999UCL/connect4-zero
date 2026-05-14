from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "version_manifest.json").exists():
            return parent
    raise FileNotFoundError("could not find repository version_manifest.json")


def manifest_path() -> Path:
    packaged = Path(__file__).with_name("version_manifest.json")
    if packaged.exists():
        return packaged
    return repo_root() / "version_manifest.json"


def current_version_info() -> dict[str, Any]:
    with manifest_path().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print c4zero tool/schema versions.")
    parser.add_argument("--json", action="store_true", help="Print the canonical version manifest as JSON.")
    args = parser.parse_args(argv)
    info = current_version_info()
    if args.json:
        print(json.dumps(info, indent=2, sort_keys=True))
    else:
        print(f"c4zero-tools {info['python_tools_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
