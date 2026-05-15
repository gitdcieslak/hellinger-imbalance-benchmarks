# Experiment Runner Specification

## Requirements

### Requirement: Config-driven execution

Experiment runs SHALL be defined by configuration files rather than hard-coded scripts.

Experiment configurations SHALL identify:

- experiment id
- dataset ids or synthetic dataset definitions
- model ids
- split strategy
- seeds
- metrics
- output location

### Requirement: Smoke-first workflow

The runner SHALL support small smoke-test runs before large benchmark campaigns.

Smoke runs SHOULD use minimal data sizes, limited seeds, and fast models where possible.

### Requirement: No implicit downloads

The runner SHALL NOT download datasets unless a user explicitly invokes a future acquisition workflow.
