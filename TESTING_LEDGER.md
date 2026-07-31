# TESTING_LEDGER.md — Every model configuration and strategy variant ever evaluated

**Append-only. Never delete or edit a row (directive §9.7).** This ledger is the trial
count *N* for the Deflated Sharpe Ratio (P10.3). An incomplete ledger makes DSR a lie.

Rules:
- Every training run, hyperparameter set, feature-set variant, label-horizon choice,
  or strategy configuration that is *evaluated* gets a row — including failures,
  abandoned ideas, and runs whose results were never looked at twice.
- Rows are appended automatically by the training entrypoint (P8.4); manual
  experiments get manual rows before the result is inspected.
- `config_hash` = SHA-256 of the canonical JSON config. `data_version` = DVC rev.

| # | timestamp (UTC) | phase/task | experiment id | config_hash | data_version | git commit | seed | CV scheme | metric(s) | result | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|

*(no model configurations evaluated yet — modelling begins Phase 8)*
