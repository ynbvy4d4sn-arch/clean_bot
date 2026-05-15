from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(".").resolve()
ENTRYPOINTS = [
    Path("daily_bot.py"),
    Path("main.py"),
    Path("smoke_test.py"),
]

IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "outputs",
    "logs",
    "data",
    "docs",
    "notebooks",
}

all_py = sorted(
    p for p in ROOT.glob("*.py")
    if p.is_file()
)

module_to_file = {p.stem: p for p in all_py}


def imports_in_file(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return set()

    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])

    return imports


imports_by_file = {p: imports_in_file(p) for p in all_py}

reachable: set[Path] = set()
frontier = [p for p in ENTRYPOINTS if p.exists()]

while frontier:
    current = frontier.pop()
    if current in reachable:
        continue
    reachable.add(current)

    for mod in imports_by_file.get(current, set()):
        target = module_to_file.get(mod)
        if target and target not in reachable:
            frontier.append(target)

unused = [p for p in all_py if p not in reachable]

out = Path("docs/ACTIVE_CODE_MAP.md")
out.parent.mkdir(exist_ok=True)

lines = []
lines.append("# Active Code Map")
lines.append("")
lines.append("## Entrypoints")
lines.append("")
for p in ENTRYPOINTS:
    lines.append(f"- `{p}` {'OK' if p.exists() else 'MISSING'}")

lines.append("")
lines.append("## Reachable local Python files")
lines.append("")
for p in sorted(reachable):
    lines.append(f"- `{p}`")

lines.append("")
lines.append("## Possibly unused top-level Python files")
lines.append("")
for p in unused:
    lines.append(f"- `{p}`")

lines.append("")
lines.append("## Import edges")
lines.append("")
for p in sorted(reachable):
    local_imports = sorted(
        mod for mod in imports_by_file.get(p, set())
        if mod in module_to_file
    )
    if local_imports:
        lines.append(f"### `{p}`")
        for mod in local_imports:
            lines.append(f"- `{mod}` -> `{module_to_file[mod]}`")
        lines.append("")

out.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {out}")
print(f"Reachable files: {len(reachable)}")
print(f"Possibly unused files: {len(unused)}")
