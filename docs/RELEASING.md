# Release checklist

1. Review original-code MIT ownership and third-party notices. Papers, author data,
   models, the local 1000-figure corpus and development logs are not source assets.
2. Run MCP contracts and real stdio demo. Separately record real attachments and
   native client receipts. Never label these tests scientific accuracy benchmarks.
3. Run `python scripts/prepare_mcp_release.py --output-dir /new/outside/repo/release`.
   It creates an allowlisted `source/`, deterministic `source.zip` and SHA inventory.
   It excludes Git history, private paths/configs, models, data, reports and logs;
   suspicious source patterns fail with filenames/line numbers, never secret values.
4. In that source snapshot, build with `python -m build`; install the wheel in a
   separate environment and test from outside the checkout. Retain upstream fixture
   license. Review the complete manifest and dependency inventory before publishing.
5. The existing research worktree may contain unrelated preexisting changes and
   sensitive history. A clean snapshot does not sanitize that history. Do not use
   blanket `git add .` or force-push; review a deliberate source-only commit or import
   into a new repository under the user's chosen publication workflow.
6. Run hosted CI after pushing with explicit authorization. Keep Linux/other-host
   native acceptance marked pending until actual runs finish. No release script
   uploads anything or modifies branches/remotes.

Local cleanup is recoverable: archive development logs and caches outside the Git
root with relative-path/hash manifests. Preserve datasets, models, original papers,
frozen review records and scientific evidence. Backups/logs must never be uploaded.

The helper's heuristic source scan is not a complete security or Git-history scan.
Local subprocess timeouts are not a sandbox; public multi-user/remote deployment
requires a separate authenticated and resource-confined design and acceptance run.
