# Validation and limitations

This repository preserves experiments; it does not present backtest outputs as investable performance.

Before publishing a metric, each experiment should provide:

1. A versioned data contract and a small synthetic/sample dataset.
2. Session-aware features and labels that cannot cross market close.
3. Purged, chronological walk-forward validation and a final untouched holdout.
4. Thresholds selected only on validation data, never the final test period.
5. A portfolio simulator that tracks concurrent positions, capital, fills, costs, slippage, and ambiguous intrabar stop/target events explicitly.
6. A command that regenerates the report from a frozen configuration.

Known historical modules may contain hard-coded local paths, incomplete dependency declarations, unfinished interfaces, or research shortcuts. They are retained to document development, not to claim production readiness.

Broker credentials and access tokens must never be stored in source control.
