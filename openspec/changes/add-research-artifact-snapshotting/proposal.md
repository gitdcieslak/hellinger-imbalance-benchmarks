# Add Research Artifact Snapshotting

## Summary

Add a lightweight snapshot system to preserve benchmark outputs with run metadata, git state, copied configs, and notes for reproducible research tracking.

## Motivation

As benchmark scope expands, we need a structured way to freeze evidence for specific runs without committing generated outputs.

## Scope

- Add research workspace docs (`research/README.md`, backlog/journal/paper scaffolding).
- Add artifact snapshot utility module and CLI.
- Snapshot reports/results/configs into `research/artifacts/runs/{run_id}/`.
- Record run manifest and git snapshot metadata.
- Add tests for layout, overwrite behavior, copied files, and notes.

## Non-Goals

- No Supabase integration.
- No model checkpoint storage.
- No benchmark execution changes.
