# MCP implementation

The public entry point is `bandstructure-mcp` (`mcp_server.cli:serve`). Direct
`python -B mcp_server/server.py` remains compatible with local configs. Both supported
entry points default to isolation and reject an explicit non-isolated launch.
Operator attachment preparation also uses a resource-bounded child. This does not
provide filesystem/network confinement; direct imported parser functions are for
trusted development, not untrusted-upload handling.

See the [root README](../README.md) and [contracts](../docs/ARCHITECTURE.md).
Version comes from `version.py`; tests check `config.json` inventory. The startup
build hash covers source/config/reference files, not dependencies or Git history;
`bandstructure-admin doctor` reports installed dependencies. Reconnect after updates.
Generating a client template is not evidence of a successful native client run.
