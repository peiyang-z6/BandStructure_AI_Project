# 0.7.1 integration and security follow-up

This release remains local, single-user research software. Passing integration
tests does not establish scientific accuracy, calibrated uncertainty or safety for
public remote deployment. The earlier 0.7.0 snapshot and its receipts are historical.

## Responsibility boundary

| Actor | Work performed | Not established by that work |
| --- | --- | --- |
| Operator / attachment adapter | Select files; bound and register bytes; SHA and coordinate transforms; obtain opaque IDs | Permission for the model to browse arbitrary local paths or publish files |
| Host AI agent | Read original pages with native vision/OCR; identify plot semantics; select ROIs; transcribe axes; research primary sources; separate reported facts from inference | Independent human approval; experimental success; truth of every quote or a unique inverse structure |
| MCP | Validate schemas, identities and declared coordinates; apply budgets; optional local PDF/OCR fallback; extract visible ink; calculate supplied-path/mesh quantities; review evidence contracts; export traceable results | An autonomous literature researcher, unrestricted filesystem browser, chemistry predictor or DFT solver |
| Independent scientist | Verify source interpretation, experimental conditions, data rights, ground truth and scientific acceptance | AI-generated labels are not a substitute for this role |

MCP tools have no URL-fetch or arbitrary local-path interface. Download and file
selection happen outside MCP, under operator/host authority. PDF text is untrusted
data, not instructions. A prompt, figure or tool result cannot grant approval.
Host-model behavior is outside the server's authority: a host may have other tools,
and the server cannot guarantee that host will never use them.

Five tools can write local cache/results and advertise that fact:
`analyze_visual_observations`, `extract_band_from_image`, `analyze_attachment`,
`export_attachment`, `validate_axis_calibration`. They are not described as
read-only/idempotent. Other MCP tools are read-only at the advertised boundary.

## Engineering changes

- Formula-like text cells are neutralized in all observation/review CSV branches;
  original labels remain in JSON. Negative numeric energies remain numbers.
- Supported stdio entry points enforce parser isolation. Workers and operator
  attachment preparation install a fail-closed 1 GiB ceiling before native imports.
  Windows Job Object enforcement and a bounded allocation refusal were tested.
  POSIX RLIMIT_AS is implemented but requires platform-specific acceptance.
- Worker pipes have bounded capture, timeout and cleanup, including thread-start
  failure. This is not filesystem/network confinement or a native parser audit.
- Batch admission bounds record/image sizes and aggregate metadata, streams hashes,
  retains paths rather than all image bytes, and rechecks bytes when consumed.
- Connected-component filtering uses a label-indexed mask rather than one full
  image comparison per component.
- Symbolic k labels can bind to relative plotted positions without inventing
  reciprocal-space units. Invalid physical-unit or inconsistent anchor claims fail.
- PDF text flags suspicious control/private-use/replacement glyphs and preserves
  raw text; it does not silently repair scientific equations or certify clean OCR.
- Valid numerical line-mode results include band and k-point counts, including
  metallic early-return paths, consistently with attachment/export consumers.

## Measured functional scope

Six operator-supplied papers (69 pages) were passed through actual MCP stdio.
The host identified electron-band panels in three papers; the other three include
phonon, ionic-transport or defect-level diagrams, not interchangeable E(k) inputs.
Seven electronic panels were exported as visible-ink CSV/SVG/PNG. Text, markers,
unresolved crossings and hidden branches remain unverified.

Three author VASP meshes and three POSCARs were imported and exported with SHA
checks. Two additional author QE line-mode datasets were checked through MCP and
drawn without dropping bands: 43 x 176 and 65 x 232 supplied samples. These are
author-dataset expansions, not proof that a later dataset revision exactly matches
every plotted calculation in the paper. Mesh data are not turned into line paths.

The local VS Code native MCP client was also exercised and actual serialized tool
receipts were exported. Private receipts, downloaded papers, author data, test logs
and the 1000-figure corpus are excluded from the public source snapshot. The
six-paper cases are exposed integration cases, not a blind benchmark or six
independent scientific validations. No model was trained for this release.

## Remaining acceptance work

Independent human labels, matched AI-only comparisons, calibrated UQ/OOD, full
per-band ground truth, cross-host/native-client testing and hosted Linux CI remain
external acceptance tasks. Public hosting would additionally require authenticating
clients and confining filesystem, network and process resources. The current store
is single-user, not multi-tenant. Dependency internals and Git history were not
exhaustively audited. No GitHub push or publication is performed by release scripts.
