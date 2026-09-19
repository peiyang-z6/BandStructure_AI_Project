# Historical research source inspection snapshot

This directory preserves the local import closure of the guarded train_ssl.py and
finetune_supervised.py entry points, with source hashes. It is **not** imported by
the MCP or copied into its Docker image. It does not include datasets, weights,
credentials or a promise of bitwise reproduction of the archived run.

The copied current source contains later provenance/acceptance gates; it must not
be represented as the exact code used by every earlier experiment. Review
[training notes](../../docs/TRAINING.md) before use. Install research libraries in a
separate environment and supply newly reviewed data contracts; do not disable guards.
