# Current development context

The maintained delivery is the local MCP-first assistant described in
[README](../README.md) and [architecture](../docs/ARCHITECTURE.md).
The host AI owns document perception; MCP owns bounded checks and evidence-aware
measurements. The current version is defined in `mcp_server/version.py`.

Historical development narratives and generated caches were moved to a private,
hash-verified archive outside this Git root. They are not release inputs. Scientific
reports, frozen datasets, model assets and their provenance were preserved locally.

Use [repair status](../docs/REPAIR_STATUS.md) and [release checklist](../docs/RELEASING.md)
for current boundaries. Do not interpret historical roadmap text as authorization
to train models, alter frozen evaluation data, publish credentials or grant AI
human-review approval. A runnable MCP is not independent scientific acceptance.
