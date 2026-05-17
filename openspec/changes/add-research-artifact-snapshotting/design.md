# Design

## Snapshot Layout

Each run snapshot is written to:

`research/artifacts/runs/{run_id}/`

with files:

- `run_manifest.json`
- `git_snapshot.txt`
- `config_snapshot/`
- `reports/`
- `results/`
- `notes.md`

## Manifest Contents

`run_manifest.json` includes:

- `run_id`
- `created_at`
- `git_commit`
- `git_branch`
- `is_git_dirty`
- `command`
- `models`
- `datasets`
- `artifacts_copied`
- `notes`

## CLI Behavior

`scripts/snapshot_research_artifacts.py`:

- requires `--run-id`
- accepts zero-or-more files for `--reports`, `--results`, `--configs`
- accepts optional `--notes`
- fails if run directory exists unless `--force`

## Git Hygiene

Snapshots are gitignored by default via:

- `research/artifacts/runs/**`
- `!research/artifacts/README.md`
