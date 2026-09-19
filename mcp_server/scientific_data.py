"""Scientific input adapters and explicitly mesh-scoped analytic measurements."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

MAX_CELLS = 262144
MAX_BANDS = 4096
METADATA_FIELDS = {
    "functional",
    "hubbard_u",
    "pseudopotential",
    "soc",
    "spin_mode",
    "temperature_K",
    "cell_setting",
    "k_path_convention",
    "source_sha256",
    "structure_sha256",
    "energy_reference_description",
    "electron_count",
    "band_channels",
}


def validate_metadata(value):
    if not isinstance(value, dict):
        return ["physical_metadata"]
    bad = ["physical_metadata." + str(k) for k in value if k not in METADATA_FIELDS]
    for key, item in value.items():
        if key == "soc":
            valid = item is None or type(item) is bool
        elif key in {"temperature_K", "electron_count"}:
            valid = item is None or (
                type(item) in (int, float) and 0 <= item <= 1e6 and math.isfinite(item)
            )
        elif key == "band_channels":
            valid = (
                isinstance(item, list)
                and len(item) <= MAX_BANDS
                and all(
                    x in ("up", "down", "spinor", "unknown") for x in item if isinstance(x, str)
                )
                and all(isinstance(x, str) for x in item)
            )
        else:
            valid = item is None or (isinstance(item, str) and len(item) <= 2000)
        if not valid:
            bad.append("physical_metadata." + key)
    return bad


def _array(value, shape=None):
    def check(x):
        if isinstance(x, list):
            return all(check(y) for y in x)
        return type(x) in (int, float) and math.isfinite(x) and abs(x) <= 1e6

    if not isinstance(value, list) or not check(value):
        raise ValueError("INVALID_NUMERIC_VALUES")
    a = np.asarray(value, dtype=float)
    if shape is not None and a.shape != shape:
        raise ValueError("INVALID_ARRAY_SHAPE")
    return a


def analyze(data):
    base = {
        "confidence": None,
        "ood_flag": None,
        "human_audited": False,
        "eligible_for_scientific_acceptance": False,
        "global_gap_eV": None,
    }
    try:
        if not isinstance(data, dict):
            raise ValueError("INVALID_SCIENTIFIC_DATA")
        if validate_metadata(data.get("physical_metadata", {})):
            raise ValueError("INVALID_PHYSICAL_METADATA")
        if data.get("data_kind") == "line_mode":
            from src.utils.physics_validator import analyze_numerical_band_data

            result = analyze_numerical_band_data(data)
            return {
                **result,
                **base,
                "data_kind": "line_mode",
                "physical_metadata": data.get("physical_metadata", {}),
            }
        if data.get("data_kind") != "uniform_mesh":
            raise ValueError("EXPLICIT_LINE_OR_MESH_KIND_REQUIRED")
        allowed = {
            "data_kind",
            "energies_eV",
            "occupancies",
            "occupation_full",
            "k_fractional",
            "k_weights",
            "lattice_A",
            "band_ids",
            "physical_metadata",
            "source",
        }
        if set(data) - allowed:
            raise ValueError("UNEXPECTED_MESH_FIELD")
        values = data.get("energies_eV")
        if not isinstance(values, list) or not 1 <= len(values) <= MAX_BANDS:
            raise ValueError("BAND_BUDGET")
        if any(not isinstance(row, list) for row in values) or sum(map(len, values)) > MAX_CELLS:
            raise ValueError("CELL_BUDGET")
        energies = _array(values)
        if energies.ndim != 2 or not 1 <= energies.shape[1] <= 65536:
            raise ValueError("INVALID_ARRAY_SHAPE")
        bands, points = energies.shape
        occupation = _array(data.get("occupancies"), energies.shape)
        full = data.get("occupation_full")
        if type(full) not in (int, float) or full not in (1, 2):
            raise ValueError("OCCUPATION_CONVENTION_REQUIRED")
        if np.any((occupation < 0) | (occupation > full)):
            raise ValueError("INVALID_OCCUPATIONS")
        k = _array(data.get("k_fractional"), (points, 3))
        weights = _array(data.get("k_weights"), (points,))
        lattice = _array(data.get("lattice_A"), (3, 3))
        if abs(np.linalg.det(lattice)) < 1e-8:
            raise ValueError("SINGULAR_LATTICE")
        if np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0, atol=1e-5):
            raise ValueError("NORMALIZED_POSITIVE_MESH_WEIGHTS_REQUIRED")
        ids = data.get("band_ids", [str(i + 1) for i in range(bands)])
        if (
            not isinstance(ids, list)
            or len(ids) != bands
            or any(not isinstance(x, str) or not 0 < len(x) <= 128 for x in ids)
            or len(set(ids)) != bands
        ):
            raise ValueError("INVALID_BAND_IDENTITIES")
        occupied = occupation >= full - 1e-6
        empty = occupation <= 1e-6
        partial = int(np.count_nonzero(~(occupied | empty)))
        result = {
            **base,
            "status": "partial_mesh" if partial else "sampled_mesh_unverified",
            "data_kind": "uniform_mesh",
            "band_count": bands,
            "k_point_count": points,
            "band_ids": ids,
            "partially_occupied_samples": partial,
            "sampled_mesh_gap_eV": None,
            "sampled_direct_gap_eV": None,
            "line_mode_gap_eV": None,
            "line_mode_topology": None,
            "physical_metadata": data.get("physical_metadata", {}),
            "source": data.get("source"),
            "weighted_electron_count": float((occupation.sum(axis=0) * weights).sum()),
            "limitations": [
                "Finite supplied mesh, not a converged global Brillouin-zone result.",
                "Partial occupations can reflect smearing; no metallicity is certified from them.",
            ],
        }
        if not partial and occupied.any() and empty.any():
            v = float(energies[occupied].max())
            c = float(energies[empty].min())
            result.update(
                sampled_mesh_edge_separation_eV=c - v,
                sampled_mesh_gap_eV=max(0.0, c - v),
                mesh_vbm_eV=v,
                mesh_cbm_eV=c,
            )
            vi = np.flatnonzero(
                np.any(occupied & np.isclose(energies, v, atol=1e-8, rtol=0), axis=0)
            )
            ci = np.flatnonzero(np.any(empty & np.isclose(energies, c, atol=1e-8, rtol=0), axis=0))
            result["extrema_fractional"] = {"vbm": k[vi].tolist(), "cbm": k[ci].tolist()}
            direct = []
            for i in range(points):
                if occupied[:, i].any() and empty[:, i].any():
                    direct.append(
                        float(energies[empty[:, i], i].min() - energies[occupied[:, i], i].max())
                    )
            if direct:
                result["sampled_direct_gap_eV"] = max(0.0, min(direct))
        else:
            result["missing_evidence"] = ["integer_occupations_and_both_band_edges"]
        # Keep conclusions ahead of identity arrays for hosts that page tool JSON.
        # All IDs remain available; a host must still read the complete response.
        result["band_ids"] = result.pop("band_ids")
        return result
    except (ValueError, TypeError, OverflowError, KeyError, RecursionError) as exc:
        return {**base, "status": "refused", "error_code": str(exc)[:160]}


def parse_poscar(payload: bytes) -> dict:
    if len(payload) > 1024 * 1024:
        raise ValueError("POSCAR_TOO_LARGE")
    lines = payload.decode("utf-8-sig").splitlines()
    scale = float(lines[1])
    lattice = np.array([[float(x) for x in line.split()] for line in lines[2:5]])
    if (
        not math.isfinite(scale)
        or scale <= 0
        or not np.isfinite(lattice).all()
        or lattice.shape != (3, 3)
    ):
        raise ValueError("UNSUPPORTED_POSCAR_SCALE")
    lattice *= scale
    if abs(np.linalg.det(lattice)) < 1e-8:
        raise ValueError("SINGULAR_LATTICE")
    symbols = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    if len(symbols) != len(counts) or any(x <= 0 for x in counts) or sum(counts) > 100000:
        raise ValueError("INVALID_SPECIES_COUNTS")
    index = 7
    if lines[index].lower().startswith("s"):
        index += 1
    mode = lines[index].strip().lower()
    index += 1
    positions = np.array(
        [[float(x) for x in row.split()[:3]] for row in lines[index : index + sum(counts)]]
    )
    if positions.shape != (sum(counts), 3) or not np.isfinite(positions).all():
        raise ValueError("INVALID_POSITIONS")
    if mode.startswith(("c", "k")):
        positions = positions * scale @ np.linalg.inv(lattice)
    elif not mode.startswith("d"):
        raise ValueError("INVALID_COORDINATE_MODE")
    return {
        "data_kind": "structure",
        "lattice_A": lattice.tolist(),
        "species": [s for s, n in zip(symbols, counts) for _ in range(n)],
        "fractional_coordinates": positions.tolist(),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "space_group": None,
        "symmetry_status": "not_inferred_from_filename_or_formula",
    }


def parse_vasprun(path: Path) -> dict:
    """Operator-side streaming import. The path is never accepted from an MCP tool."""
    if path.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("VASP_SOURCE_TOO_LARGE")
    # Scan the entire bounded XML, including declarations across block boundaries.
    with path.open("rb") as source:
        tail = b""
        for block in iter(lambda: source.read(1024 * 1024), b""):
            window = tail + block
            if b"<!DOCTYPE" in window or b"<!ENTITY" in window or b"\x00" in window:
                raise ValueError("XML_ENTITIES_OR_NON_UTF8_FORBIDDEN")
            tail = window[-16:]
    stack = []
    varray = None
    rows = []
    eigen = False
    spin = 0
    k = 0
    spectra = {}
    vectors = {}
    parameters = {}
    generation = None
    cells = 0
    for event, node in ET.iterparse(path, events=("start", "end")):
        if event == "start":
            stack.append((node.tag, dict(node.attrib)))
            if node.tag == "varray":
                varray = node.get("name")
                rows = []
            if node.tag == "generation" and any(t == "kpoints" for t, a in stack):
                generation = node.get("param")
            if node.tag == "eigenvalues" and len(stack) > 1 and stack[-2][0] == "calculation":
                eigen = True
                spectra = {}
                cells = 0
            if eigen and node.tag == "set":
                label = node.get("comment", "")
                if label.startswith("spin "):
                    spin = int(label.split()[-1])
                if label.startswith("kpoint "):
                    k = int(label.split()[-1])
                    spectra.setdefault(spin, {})[k] = []
            continue
        name = node.get("name")
        text = (node.text or "").strip()
        if node.tag == "i" and name in {"LSORBIT", "LHFCALC", "GGA", "HFSCREEN", "NELECT", "ISPIN"}:
            parameters[name] = text
        if node.tag == "v" and varray in {"basis", "kpointlist", "weights"}:
            rows.append([float(x) for x in text.split()])
        if node.tag == "varray":
            if varray in {"basis", "kpointlist", "weights"}:
                vectors[varray] = rows
            varray = None
            rows = []
        if eigen and node.tag == "r":
            cells += 1
            if cells > MAX_CELLS:
                raise ValueError("CELL_BUDGET")
            spectra[spin][k].append([float(x) for x in text.split()])
            if len(spectra[spin][k]) > MAX_BANDS:
                raise ValueError("BAND_BUDGET")
        if node.tag == "eigenvalues" and eigen:
            eigen = False
        stack.pop()
        node.clear()
    if generation not in {"Gamma", "Monkhorst-Pack"}:
        raise ValueError("EXPLICIT_PATH_SEGMENTS_REQUIRED")
    arrays = [np.array([channel[i] for i in sorted(channel)]) for channel in spectra.values()]
    if not arrays:
        raise ValueError("EIGENVALUES_MISSING")
    energies = np.concatenate([x[:, :, 0].T for x in arrays])
    occupancies = np.concatenate([x[:, :, 1].T for x in arrays])
    if energies.size > MAX_CELLS:
        raise ValueError("CELL_BUDGET")
    soc = parameters.get("LSORBIT") == "T"
    full = 1 if soc or parameters.get("ISPIN") == "2" else 2
    metadata = {
        "functional": "hybrid " + parameters.get("GGA", "unknown")
        if parameters.get("LHFCALC") == "T"
        else parameters.get("GGA", "unknown"),
        "soc": soc,
        "spin_mode": "spinor" if soc else "collinear" if len(arrays) > 1 else "unpolarized",
        "electron_count": float(parameters.get("NELECT", "0")),
    }
    result = {
        "data_kind": "uniform_mesh",
        "energies_eV": energies.tolist(),
        "occupancies": occupancies.tolist(),
        "occupation_full": full,
        "k_fractional": vectors["kpointlist"],
        "k_weights": [x[0] for x in vectors["weights"]],
        "lattice_A": vectors["basis"],
        "band_ids": [
            f"channel{spin_index + 1}:band{b + 1}"
            for spin_index, array in enumerate(arrays)
            for b in range(array.shape[1])
        ],
        "physical_metadata": metadata,
    }
    check = analyze(result)
    if check["status"] == "refused":
        raise ValueError(check["error_code"])
    return result
