"""Run metadata and audit helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Any

import json


def _git_commit_hash(base_dir: str | Path | None = None) -> str:
    """Return the current git commit hash when available."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode == 0:
            return completed.stdout.strip()
    except Exception:
        pass
    return ""


def create_run_metadata(
    params: dict[str, Any],
    active_tickers: list[str],
    mode: str,
    *,
    removed_tickers: list[str] | None = None,
    data_start: str | None = None,
    data_end: str | None = None,
    execution_mode: str = "order_preview_only",
) -> dict[str, Any]:
    """Build a compact run-metadata payload."""

    return {
        "run_id": f"audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "model_version": "v1",
        "asset_universe_version": "registry_v1",
        "git_commit_hash": _git_commit_hash(),
        "random_seed": int(params.get("random_seed", 42)),
        "active_tickers": list(active_tickers),
        "removed_tickers": list(removed_tickers or []),
        "parameter_json": {key: value for key, value in params.items() if key not in {"investopedia_password"}},
        "data_start": data_start,
        "data_end": data_end,
        "execution_mode": execution_mode,
        "dry_run": bool(params.get("dry_run", True)),
    }


def write_audit_metadata(metadata: dict[str, Any], output_path: str | Path) -> Path:
    """Write audit metadata to JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path
