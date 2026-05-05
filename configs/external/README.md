# External Configs

This folder contains configuration files for external chess tooling, not neural-network training.

- `cli.config.toml`: local engine/CRTK-style CLI defaults.
- `default.engine.toml`: referenced by `cli.config.toml` if an external engine protocol file is added locally.

Training configs live under `configs/benchmarks/`, `configs/suites/`, and `ideas/i###_*/config.yaml`.
