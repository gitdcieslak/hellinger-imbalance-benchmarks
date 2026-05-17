# Tasks

## 1. Research Docs

- [x] Add `research/README.md`.
- [x] Add `research/artifacts/README.md`.
- [x] Add paper outline scaffold.
- [x] Add initial backlog entries.

## 2. Snapshot Utility

- [x] Add `src/hib/artifacts.py` with snapshot logic.
- [x] Add `scripts/snapshot_research_artifacts.py` CLI.
- [x] Capture git branch/commit/dirty state and write `git_snapshot.txt`.
- [x] Copy reports/results/configs into run snapshot layout.
- [x] Enforce duplicate run-id failure unless `--force`.

## 3. Tests and Verification

- [x] Add tests for layout, manifest git fields, copying, force behavior, and notes writing.
- [x] Run `python -m pytest`.
- [x] Run snapshot verification CLI command.
