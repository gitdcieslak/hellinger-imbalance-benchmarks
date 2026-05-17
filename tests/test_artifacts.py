import json
import subprocess
from pathlib import Path

import pytest

from hib.artifacts import snapshot_research_artifacts


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True, capture_output=True, text=True)
    (path / "README.tmp").write_text("init\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.tmp"], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True, capture_output=True, text=True)


def _make_file(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_snapshot_creates_layout_and_copies_files(tmp_path):
    _init_git_repo(tmp_path)
    report = _make_file(tmp_path / "reports" / "a.md", "report")
    result = _make_file(tmp_path / "results" / "a.jsonl", "{}\n")
    config = _make_file(tmp_path / "configs" / "m.yaml", "models: {}\n")

    run_dir = snapshot_research_artifacts(
        run_id="run-001",
        reports=[report],
        results=[result],
        configs=[config],
        notes="hello",
        command="python demo",
        workspace_root=tmp_path,
    )

    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "git_snapshot.txt").exists()
    assert (run_dir / "config_snapshot" / "m.yaml").exists()
    assert (run_dir / "reports" / "a.md").exists()
    assert (run_dir / "results" / "a.jsonl").exists()
    assert (run_dir / "notes.md").read_text(encoding="utf-8").strip() == "hello"


def test_manifest_contains_git_fields(tmp_path):
    _init_git_repo(tmp_path)
    report = _make_file(tmp_path / "reports" / "a.md", "report")
    run_dir = snapshot_research_artifacts(
        run_id="run-002",
        reports=[report],
        results=[],
        configs=[],
        workspace_root=tmp_path,
    )
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert "git_commit" in manifest
    assert "git_branch" in manifest
    assert "is_git_dirty" in manifest


def test_duplicate_run_id_fails_without_force(tmp_path):
    _init_git_repo(tmp_path)
    report = _make_file(tmp_path / "reports" / "a.md", "report")
    snapshot_research_artifacts("run-003", [report], [], [], workspace_root=tmp_path)
    with pytest.raises(FileExistsError):
        snapshot_research_artifacts("run-003", [report], [], [], workspace_root=tmp_path)


def test_force_overwrites_existing_run(tmp_path):
    _init_git_repo(tmp_path)
    report1 = _make_file(tmp_path / "reports" / "a.md", "first")
    run_dir = snapshot_research_artifacts("run-004", [report1], [], [], workspace_root=tmp_path)
    assert (run_dir / "reports" / "a.md").read_text(encoding="utf-8") == "first"

    report2 = _make_file(tmp_path / "reports" / "a.md", "second")
    run_dir = snapshot_research_artifacts("run-004", [report2], [], [], workspace_root=tmp_path, force=True)
    assert (run_dir / "reports" / "a.md").read_text(encoding="utf-8") == "second"
