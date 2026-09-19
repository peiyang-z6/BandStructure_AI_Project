# Scientific acceptance is separate from software verification

The current local corpus contains 1000 AI-assisted figure records from 839 documents
and approximately 2471 candidate boxes. Independent human-approved count is **zero**.
Capacity, Computer Use and a `human_confirmed` flag cannot change this. This corpus
and development receipts are not distributed with the source release.

Before reporting scientific benchmarks:

1. Verify rights, paper/figure provenance, document/material grouping and exact/near
   duplicates, including cross-document reuse and third-party figures.
2. Record development exposure. Previously debugged samples are not unseen blind
   tests. Freeze a new document/material-disjoint set after code/prompts/settings.
3. Independent humans must correct axes, panels, curves, references and quantitative
   targets using predefined tolerances, external authentication and hash-bound evidence.
4. Compare AI-only and AI+MCP with the same model, attachments, budgets and settings.
   Predefine error, dangerous false acceptance, refusal, coverage, latency and cost;
   retain failed/timeout cases rather than dropping them.
5. Claims of UQ/OOD require independent calibration and risk-coverage/shift experiments.
   Current nulls mean unknown, not safe or in-distribution.
6. Establish novelty through current related work and measured gains. A wrapper or
   integration test does not establish exclusivity or superiority.

Legacy learned models, 60k retrieval and active DFT remain disabled. Restoring them
requires separate provenance, baselines and acceptance. MCP does not recover unseen
branches or uniquely infer structure/synthesis from bands. Author data or new physical
calculations may be necessary. Do not train on evaluation-reserved data to pass tests.
