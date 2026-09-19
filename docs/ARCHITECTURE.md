# Architecture and contracts

Version 2 adds Docker and authenticated Streamable HTTP alongside stdio; see
[deployment](DEPLOYMENT.md). Docker enforces an additional process/filesystem
boundary, while the Python parser's `os_sandboxed` field remains false: it does not
authenticate or introspect the surrounding container configuration. Network egress
is not completely disabled. HTTP instances are single-owner, never multi-tenant.

## Responsibility boundary

The operator selects attachments; the deterministic adapter registers bounded bytes
and digests. The host AI interprets originals and proposes observations. MCP checks
types, budgets, references, coordinates and physical consistency. Independent human
review is external. Paper/OCR contents are data, never executable instructions.

## Scientific inputs

`band://schemas` and `tools/list` publish nested observation v2 and scientific-data
contracts. Incomplete observations can be sent to `request_missing_evidence` for
actionable questions. Invalid types are errors. Unversioned Fermi inputs remain
supported. v2 supports explicit Fermi/VBM/arbitrary references, nullable curve
samples, source labels, segments and optional physical metadata.

A continuous branch crossing known EF within one segment makes a sampled path
metallic. Numerical and dense v2 inputs share the same kernel. Explicit band roles
with VBM/arbitrary references permit edge separation without inventing EF. Ambiguous
or incomplete observations cannot certify a gap. Directional effective mass requires
isolated, straight physical-k samples and stable fits, not just an image parabola.

`analyze_electronic_data` requires `data_kind`. Uniform mesh also needs occupancies,
their full-occupation convention, fractional k, positive normalized weights and
lattice. It reports `sampled_mesh_gap_eV`, never a line-mode or certified global gap.
Partial occupations may be smearing, not proof of metallicity. VASP imports preserve
channels/global band IDs within 4096 bands/262144 cells; unsupported paths are refused,
not converted from point indices. POSCAR supports positive scalar scale and explicit
species with Direct/Cartesian positions; symmetry is not guessed.

## Source and geometry

An attachment ID plus matching SHA verifies registered bytes only.
`validate_axis_calibration` checks declared eV/y and k/x anchors, strict monotonicity,
raw/corrected numeric-label agreement and half-pixel linear-fit residual. Its ID can
bind v2 `pixel_points` samples to transforms. It does not independently authenticate
OCR, captions or branch identity. Unregistered observations remain caller-declared.
Corrections require raw text and evidence; minus signs/EF are not invented.

The host must crop electronic bands separately from DOS and declare semantics.
Candidates are geometry aids. Mixed-colour extraction retains neutral/coloured ink
and may retain labels/markers. Coverage describes observed ink or supplied samples,
not all physical bands. Reject non-electronic panels.

`review_material_evidence` separates composition/dopants, crystal/lattice,
experimental/cited synthesis and properties. Reported/inferred claims need
document/page/excerpt/origin; inferences also need reasons. A computational paper
cannot be represented as a reported experimental SOP. Source quotes still need
independent checking; contract success is not verified chemistry.

## Bounded lifecycle and deployment

- Direct upload <=10 MiB; parser pixel/page budgets remain enforced.
- Operator source <=256 MiB; images <=64M source pixels and resized <=4096 per side.
  PDF `--page`/`--all-pages` (<=50 pages) emits rasters with inverse transforms,
  including rotations. Original/derived hashes are distinct.
- PDF cursor cache: two entries/64 MiB/600s, immutable; same-key misses coalesce,
  different documents can run concurrently. Expired cursors fail explicitly.
- Large first PDF chunks/CSV exports provide `result_id`. `read_result_chunk` uses
  Unicode-character offsets, <=200000 characters per call, UTF-8 whole-result SHA
  and SHA-required continuation. Results survive restart for 24h; in-progress
  parsing is not a persistent/cancellable job.
- Single-user store: 32 MiB/object, 512 MiB/256 objects total. No auto-eviction.
  `bandstructure-admin prune-expired` previews; `--apply` deletes only expired
  result pairs. Attachments require operator archival/removal and are never pruned
  by that command. The store is not a multi-tenant authentication boundary.
- Two concurrent parser subprocesses, 60/90-second timeouts and bounded pipe reads.
  Supported stdio entry points require isolation. Before native imports, each worker
  installs a 1 GiB ceiling (Windows Job Object; POSIX RLIMIT_AS). Installation failure
  refuses parsing. Operator attachment preparation/page counting uses a separately
  bounded 120-second child. Windows enforcement was tested; other platforms require
  their own acceptance run. Low-level imported parser functions are trusted-development
  primitives, not an isolated untrusted-upload interface.
  Busy requests return `UPLOAD_WORKER_BUSY`. This is **not a filesystem/network
  sandbox**. Unauthenticated or multi-tenant public hosting remains unsupported.

Exports show supplied samples only. Nulls/segment breaks remain discontinuities;
SVG segment widths follow supplied k spans (relative coordinates remain relative).
Large data use CSV chunks without claiming full graphical/physical reconstruction.

CSV text cells with formula-like prefixes are apostrophe-neutralized. Numeric
energies remain numbers. Use source JSON for exact label identity; review exports
also include `figure_review.json`. Five tools advertise local cache writes rather
than read-only/idempotent behavior. Status/read methods do not create the store.

Symbolic k anchors may declare `value_basis=relative_plot_position`; their values
must match normalized pixel positions and cannot claim inverse-angstrom units.
PDF text now flags control/replacement/private-use glyphs for native visual review,
retaining raw text. Nonempty text is not a guarantee of correct scientific OCR.
