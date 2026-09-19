# Third-party notices

The MIT license covers original project code. It does not relicense dependencies,
research papers, downloaded figures, pretrained weights, author datasets or other
third-party works. None of the private paper/1000-image corpus, credentials,
development logs, trained models or local reports belongs in a public release.

## Bundled public fixture

`mcp_server/resources/public_reference.json` contains a derived public pymatgen
Cu2O band-structure reference. Retain its provenance fields and the full upstream
MIT notice in `mcp_server/resources/PYMATGEN_LICENSE.txt`. This fixture is a public
regression example, not an unseen test or a trained prediction.

## Optional PDF dependency

PyMuPDF/MuPDF is offered under AGPL and commercial agreements. Installing or
distributing the optional `[documents]` extra does not make it MIT-licensed.
Review the applicable agreement before redistribution/deployment; consult the
[upstream licensing notice](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright).
The project's MIT source license is not a promise that a combined distribution
has no additional obligations.

## Other dependencies

The package manager installs MCP SDK, NumPy, SciPy, Pillow, jsonschema and optional
OpenCV, RapidOCR and ONNX Runtime independently. Preserve their distributed
license/notice files. `bandstructure-admin doctor` records installed versions;
the pinned Windows reference environment is `mcp_server/requirements.txt`.
This file is an inventory, not a substitute for a full dependency-license review.

Version 2 additionally uses PyJWT/cryptography for OAuth token verification and
Uvicorn/Starlette for HTTP delivery. The Docker image includes the optional PDF/OCR
stack and upstream notice files installed by pip; its project-label MIT describes
the original source, not a relicensing of the complete combined image. Docker
Desktop and optional proprietary AI hosts have separate terms and are not bundled.
