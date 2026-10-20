# Engineering rules

This repository is an independent research implementation. The neighbouring
`MultiAgents-main` repository is a reference, not a dependency.

1. Keep domain and algorithm modules free of Ray, Hydra, provider SDKs and file/network I/O.
   `torch`, `verl` and `agentlightning` are imported only in `src/orchestrator_research/training/`.
2. Depend on small `Protocol` interfaces; put concrete API/GPU/storage code in `adapters/`.
3. Every mathematical change needs a formula, a source note and invariant-based unit tests.
4. Every experiment changes one declared variable and writes immutable configuration + results.
5. Do not add a dependency when the standard library is sufficient. Record necessary architectural
   decisions in `docs/adrs/`.
6. Train rows 0-599 never go into `val_test` (600-799). `val_full` (0-799) overlaps train by design
   for parity with the previous project; never report it as held-out (ADR 0005).
7. Одна гипотеза = один каталог `experiments/NNN_name/` с `config.toml` и `README.md`; `[experiment].name` = имя каталога.
   Это соглашение, код его не проверяет.

