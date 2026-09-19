# BandStructure MCP · Second Edition (v2.0.0)

[中文](README.md) · [Deployment](docs/DEPLOYMENT.md) · [Historical training](docs/TRAINING.md) · [Verification](docs/V2_VERIFICATION.md) · [MIT](LICENSE)

An evidence-aware assistant for band-structure literature and supplied scientific data.
Recommended deployment: Docker on **Windows / Linux, x86-64**, using stdio or
authenticated Streamable HTTP. No GPU, CUDA, Materials Project key or historical
model weights are required for the current MCP service.

> Research software, **not a DFT engine or a scientifically validated prediction
> service**. Unknown confidence stays null. A band image does not uniquely identify
> composition, structure or a synthesis recipe. AI labels are not human approval.

## 1. Purpose and applications

The server exposes 17 tools, five resources and two prompts for semiconductor,
thermoelectric, photovoltaic, battery and superconductivity literature reading,
introductory teaching and data organization. It distinguishes electronic bands from
phonons, DOS, defect levels and transport plots; checks units, axes, references,
path breaks and provenance; measures supplied numerical bands; and exports
traceable JSON/CSV/SVG or paged results. Operator-side inputs include supported
images, PDFs, VASP XML and POSCAR files.

Structure, composition, material properties and synthesis statements require
sources. A contract check does not authenticate every quote or prove feasibility.

## 2. Host-model and MCP responsibilities

![Architecture](docs/figures/mcp_architecture.png)

The operator registers selected bytes as opaque attachment IDs with SHA-256 hashes.
The host AI uses its native vision/OCR and language understanding to interpret the
original, research sources, select a region and propose structured observations.
Its client discovers and calls MCP tools. The server validates contracts and
computes supplied-data quantities, returning results or missing-evidence requests.
The host explains them; an independent expert performs scientific acceptance.

**The host calls its model; this MCP does not call a second model. Historical neural
weights, ANN retrieval and DFT execution are disabled.** Local OCR/PDF parsing is
a fallback. Unknown samples, invisible branches and path discontinuities remain
unresolved. See the [MCP architecture](https://modelcontextprotocol.io/specification/2025-11-25/architecture).

| Capability | Without this MCP | With this MCP |
| --- | --- | --- |
| Vision and explanation | Host capabilities | Still host capabilities |
| Numerical work | Host tools/code and prompt dependent | Shared path/mesh contracts |
| Provenance and axes | Host-managed | Digest, region and anchor checks |
| Missing evidence/exports | Workflow-dependent | Explicit refusal/questions and traceable results |
| Accuracy or speed gain | No matched experiment | **No invented improvement percentage** |

AI-only systems can also use code and other tools. This is a capability comparison,
not evidence of superiority.

## 3. Historical model training

The current MCP requires no training. Earlier offline research used AFLOW
**numerical bands**, masked-point reconstruction and classification/band-edge
fine-tuning—not the local 1000-paper-figure collection.

- Recorded server: two NVIDIA Tesla V100-SXM2 GPUs, 16 GB each; this does not imply
  that every experiment used both GPUs.
- Archived environment: Python 3.11, TensorFlow 2.21, CUDA 12.5.82 and cuDNN
  9.3.0.75; separate from the CPU MCP container.
- 60,000 source records, 59,899 usable 128×6 inputs and 101 skipped for missing
  required band-edge envelopes.
- Space-group-disjoint training/outer-test partition: 47,912 / 11,987 records.
  Inner validation selects checkpoints.
- The plotted archive is internally consistent: seed 42, 54 epochs, best epoch 34.
  Its stored confusion matrix gives **94.20% accuracy and 90.56% macro F1**.

![Historical numerical-model evaluation](docs/figures/historical_training_evaluation.png)

These are archived numerical-model results, **not MCP/image accuracy, a blind
benchmark, or an AI-only comparison**. A different 51-epoch log was not mixed in.
The gap task also has a direct analytic baseline; tiny regression error is not
independent evidence of physical prediction. [Methods, code and limitations](docs/TRAINING.md);
[reviewed numbers and hashes](docs/figures/paper_metrics.json).

## 4. Practical paper cases

The earlier 0.7.1 integration recorded six papers, 69 pages, 76 document-tool calls,
seven visible electronic-panel exports, and two author-data plots retaining 43 and
65 supplied bands. Version-2 deployment/authentication tests are listed separately
in [verification](docs/V2_VERIFICATION.md).

| Paper/material | Scope |
| --- | --- |
| [PRX Energy: BaB₂, ZrRuSb, TaRu₃C](https://doi.org/10.1103/sb28-fjc9) | Separate electronic/phonon plots; author-data expansion is not unseen-band inference |
| [ACS Energy Letters: NMC](https://doi.org/10.1021/acsenergylett.1c02028) | Tested phonon plots are not electronic-gap input |
| [JMCA: Na₃OCl](https://doi.org/10.1039/D1TA07588H) | Ionic transport is not E(k); apply the [published correction](https://doi.org/10.1039/D2TA90105F) |
| [ACS AMI: ZnGa₂O₄](https://doi.org/10.1021/acsami.5c19146) | Visible panel recovery; reported static indirect gap 5.08 eV, no unseen conduction branches |
| [EES: kesterite photovoltaics](https://doi.org/10.1039/D0EE00291G) | Defect levels/configuration coordinates are distinct from E(k) |
| [JMCA: Bi₂MO₄Cl](https://doi.org/10.1039/D5TA05523G) | Keep path and mesh quantities separate; trace synthesis to experimental sources |

These exposed cases are not blind evaluations. Papers, downloaded images, weights
and the private 1000-item corpus are not redistributed.

## 5. Installation and clients

Prerequisites: Git, a running Linux-container Docker Engine/Desktop and Compose v2.
Windows uses WSL2. Allow at least 4 GB available RAM and 3 GB disk; the first build
needs network access. Docker Desktop has separate terms; Linux/WSL Docker Engine
is an open-source alternative.

    git clone https://github.com/peiyang-z6/BandStructure_AI_Project.git
    cd BandStructure_AI_Project
    docker compose run --build --rm setup
    docker compose up -d --wait --wait-timeout 120
    docker compose cp bandstructure:/data/client-configs ./.bandstructure-clients

PowerShell: ./scripts/start.ps1. Linux: sh scripts/start.sh. Generated configs contain
credentials: **keep them private**. Default endpoint: http://127.0.0.1:8765/mcp,
loopback-only with mandatory authentication.

- Hermes: merge mcp_servers from hermes.config.json into your own configuration.
- Claude Code: merge claude-code.mcp.json for HTTP.
- Claude Desktop: merge claude-desktop.json for Docker-exec stdio.
- VS Code: merge vscode.mcp.json into user MCP settings or .vscode/mcp.json.
- ChatGPT: authorized Secure MCP Tunnel or operator-managed HTTPS/OAuth. Localhost
  is not a cloud endpoint; static bearer authentication is not ChatGPT UI OAuth.

[Full instructions, attachment import, OAuth, backup and troubleshooting](docs/DEPLOYMENT.md).
No domain, OAuth application, cloud account or paid service is created automatically.
One instance/credential shares one owner's attachments. Separate users need separate
instances, volumes and authentication.

Original code is [MIT](LICENSE); dependencies retain their own licenses. The Docker
image includes the PDF stack: review [third-party notices](THIRD_PARTY_NOTICES.md),
including PyMuPDF/MuPDF obligations. Anyone may fork, open issues and submit pull
requests; anonymous direct writes to the default branch are not enabled.
[Contributing](CONTRIBUTING.md).

## 6. Personal developer statement

I study mechanical engineering and am not formally trained in computational
materials science. I developed this learning project with AI-agent assistance,
drawing on open-source protocols, libraries and public research. I hope it helps
beginners read band literature and learn about DFT. Researchers' and developers'
constructive corrections are very welcome.

The original source is public and the core runtime uses open-source components.
Development also involved AI tools and document software, and connected hosts can
be proprietary, so I do not claim that no closed-source tool was ever used. AI
assistance does not replace professional validation; I welcome reproducible feedback.
