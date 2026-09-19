# Client verification matrix

Version 2 adds authenticated Streamable HTTP and Docker-exec stdio. A config
template is not a native-client acceptance test. See [deployment](DEPLOYMENT.md)
and the current [verification record](V2_VERIFICATION.md).

| Surface | Evidence boundary |
| --- | --- |
| Python MCP SDK 1.30 / Windows | Real initialize/list/tools/resources/prompts tested |
| VS Code native MCP / Copilot Chat | 0.7 native status and real-image/448-band mesh calls verified with a project-shared store; actual invocation receipts kept privately |
| Claude Desktop, Cursor, Codex | Templates provided; no new native 0.7 acceptance claim |
| Linux Docker | Version-2 runtime and protocol checks are recorded separately in V2_VERIFICATION.md |
| Remote/cloud HTTP | Single-owner bearer or OAuth JWT resource server; operator HTTPS/issuer and account-level acceptance are still required; not multi-tenant |

For installed packages configure the absolute `bandstructure-mcp` executable as
stdio command. Source-checkout configs: `python scripts/generate_mcp_client_configs.py
--help`. Generated configs contain local paths and must remain private. Adapter and
server must share `BAND_MCP_STORE_DIR`. Native verification records host/model/version,
build SHA, attachment SHA, actual tool invocation/structured outputs and latency,
not just an AI summary. GUI automation is never independent human review.

## Store visibility and long results

In this Windows environment the adapter and native VS Code process initially saw
different contents despite reporting the same AppData path hash. The default-path
fallback alone did not resolve that case; the underlying host/profile visibility
mechanism was not established. Registering bytes in an explicit project-shared
private directory, with that exact `BAND_MCP_STORE_DIR` in both processes, resolved
the observed failure without changing security permissions.

`get_service_status.artifact_store` reports a path fingerprint and attachment count.
Matching path text alone does not prove a shared filesystem namespace. Perform an
actual `inspect_attachment` roundtrip first. Keep the active store private, outside
the public snapshot, and do not remove it during backup cleanup. IDs cannot be used
in a different empty store.

Some hosts page long JSON results into a local resource. Read the complete result
before declaring a field absent. In the 448-band case an initial host summary missed
gap fields after reading only 300 lines; the actual returned JSON contained them.
Scalar measurements now precede long band-ID arrays, without dropping IDs.
