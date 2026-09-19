# Version 2 verification / 第二版验证范围

This record separates engineering checks from scientific acceptance. Final counts
are updated after the corresponding run, not inferred from a workflow definition.

## Already established before v2

0.7.1: 453 Windows contract tests; six supplied papers/69 pages/76 document calls;
seven visible electronic-panel exports; three VASP meshes/three POSCAR imports;
two additional author line-mode datasets (43×176 and65×232); native VS Code receipts.
These are exposed integration cases, not independent scientific accuracy estimates.

## Version 2 work in progress

- Windows initial regression:460 passed, including7 initial HTTP/authentication tests.
- Docker Desktop29.8.0 with Linux containers: image builds and the service becomes healthy.
- HTTP initialization, tool/resource listing, numerical calls and negative auth/origin
  controls ran. The image parser returned requires_calibration with POSIX memory
  enforcement; the smoke-test expectation was corrected accordingly.
- Compose's tmpfs comma-separated option initially needed YAML quoting; corrected.
- Final-image, clean-source/Linux and hosted-CI results will be recorded after completion.

## Unperformed acceptance

No new model training, calibrated UQ/OOD or AI-only superiority experiment. No claim
that every Hermes/Claude/VS Code/ChatGPT account or host was tested. The operator has
not supplied the exact public HTTPS URL, issuer, JWKS or owner subject, so no named
public OAuth deployment is reported. JWT verification tests do not constitute a live
identity-provider or ChatGPT account-level acceptance run. No multi-user isolation.

Tokens, private receipts, original PDFs, full model weights and development logs are
excluded from the release. Historical research source and aggregate figure inputs
are included separately, with their limitations.
