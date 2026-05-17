"""Research artifact snapshot utilities."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run_git(repo_root: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def collect_git_snapshot(repo_root: Path) -> dict[str, Any]:
    """Collect git metadata and a human-readable status snapshot."""

    try:
        commit = _run_git(repo_root, ["rev-parse", "HEAD"])
        branch = _run_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
        status_short = _run_git(repo_root, ["status", "--short"])
        status_full = _run_git(repo_root, ["status", "--short", "--branch"])
        dirty = bool(status_short)
        return {
            "git_commit": commit,
            "git_branch": branch,
            "is_git_dirty": dirty,
            "git_snapshot_text": status_full,
        }
    except Exception:
        return {
            "git_commit": None,
            "git_branch": None,
            "is_git_dirty": True,
            "git_snapshot_text": "git metadata unavailable",
        }


def _copy_files(paths: list[Path], destination_dir: Path) -> list[str]:
    copied: list[str] = []
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source in paths:
        src = source.resolve()
        if not src.exists():
            raise FileNotFoundError(f"snapshot source path does not exist: {src}")
        if src.is_dir():
            raise ValueError(f"snapshot source must be a file: {src}")
        target = destination_dir / src.name
        shutil.copy2(src, target)
        copied.append(str(target))
    return copied


def snapshot_research_artifacts(
    run_id: str,
    reports: list[Path],
    results: list[Path],
    configs: list[Path],
    notes: str = "",
    command: str = "",
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    workspace_root: Path = Path("."),
    force: bool = False,
) -> Path:
    """Create a research artifact snapshot under research/artifacts/runs/{run_id}."""

    workspace = workspace_root.resolve()
    runs_root = workspace / "research" / "artifacts" / "runs"
    run_dir = runs_root / run_id
    if run_dir.exists():
        if not force:
            raise FileExistsError(f"snapshot run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    git_data = collect_git_snapshot(workspace)
    copied_reports = _copy_files(reports, run_dir / "reports")
    copied_results = _copy_files(results, run_dir / "results")
    copied_configs = _copy_files(configs, run_dir / "config_snapshot")

    artifacts_copied = copied_reports + copied_results + copied_configs
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_data["git_commit"],
        "git_branch": git_data["git_branch"],
        "is_git_dirty": bool(git_data["is_git_dirty"]),
        "command": command,
        "models": models or [],
        "datasets": datasets or [],
        "artifacts_copied": artifacts_copied,
        "notes": notes,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (run_dir / "git_snapshot.txt").write_text(str(git_data["git_snapshot_text"]) + "\n", encoding="utf-8")
    (run_dir / "notes.md").write_text((notes or "") + "\n", encoding="utf-8")
    return run_dir
