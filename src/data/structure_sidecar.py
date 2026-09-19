"""P1 structure sidecar: read-only structure fields paired to the band HDF5.

Constitution 5.0 §8 P1: the immutable aflow_bands.h5 is never modified; a
separate sidecar keyed by material_id stores lattice 3x3, species,
fractional_coordinates, magnetic/spin, DFT functional/U/pseudopotential,
reciprocal lattice, 3D fractional k-points, k-path convention, source and a
structure SHA-256.

Data comes from AFLOW REST endpoints (verified reachable 2026-09-06):
  ?geometry                -> [a,b,c,alpha,beta,gamma]  (degrees)
  ?positions_fractional    -> fractional coords
  ?positions_cartesian     -> cartesian coords (audit only)
  ?species / ?species_pp   -> element list / pseudopotential list
  ?dft_type                -> functional (e.g. ["PAW_PBE"])
  ?spin_cell / ?spin_atom / ?spinD -> magnetic/spin
  ?kpoints_bands_path      -> high-symmetry path segment labels
  ?kpoints                 -> [relax_mesh, static_mesh, segments, nkpts]
  ?code                    -> VASP version
  KPOINTS.bands            -> line-mode 3D fractional k-points per segment

`?lattice` is 404; lattice 3x3 is constructed from `?geometry`. Reciprocal
lattice is derived from the direct lattice.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import os
import tempfile
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _parse_aflow_list(value: Any) -> Optional[List]:
    """Parse a scalar/list-of-scalars or a JSON/`ast.literal_eval` string."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, (int, float, str)):
        text = str(value).strip()
        if not text:
            return None
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            try:
                return json.loads(text)
            except (ValueError, TypeError):
                return [value]
    return None


def geometry_to_lattice_vectors(
    a: float, b: float, c: float, alpha_deg: float, beta_deg: float, gamma_deg: float
) -> np.ndarray:
    """Standard crystallographic direct-lattice matrix from cell parameters.

    a along x; b in xy-plane; c general. Angles in degrees. Returns (3,3)
    row-vector matrix (rows are lattice vectors).
    """
    parameters = np.asarray([a, b, c, alpha_deg, beta_deg, gamma_deg], dtype=np.float64)
    if (not np.isfinite(parameters).all() or np.any(parameters[:3] <= 0)
            or np.any(parameters[3:] <= 0) or np.any(parameters[3:] >= 180)):
        raise ValueError("invalid cell lengths/angles")
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz_squared = c * c - cx * cx - cy * cy
    if cz_squared <= c * c * 1e-20:
        raise ValueError("impossible or degenerate cell angles")
    cz = np.sqrt(cz_squared)
    a3 = np.array([cx, cy, cz])
    return _validate_lattice(np.stack([a1, a2, a3], axis=0).astype(np.float64))


def build_lattice_from_geometry(geometry: Any) -> Optional[np.ndarray]:
    vals = _parse_aflow_list(geometry)
    if not isinstance(vals, (list, tuple)) or len(vals) != 6:
        return None
    try:
        return geometry_to_lattice_vectors(*(float(v) for v in vals))
    except (TypeError, ValueError, OverflowError):
        return None


def reciprocal_lattice_from_vectors(lattice: np.ndarray) -> np.ndarray:
    """Reciprocal lattice rows b_i (a_i . b_j = 2*pi*delta_ij)."""
    lattice = _validate_lattice(lattice)
    vol = np.dot(lattice[0], np.cross(lattice[1], lattice[2]))
    if abs(vol) < 1e-12:
        raise ValueError("zero cell volume")
    b1 = 2.0 * np.pi * np.cross(lattice[1], lattice[2]) / vol
    b2 = 2.0 * np.pi * np.cross(lattice[2], lattice[0]) / vol
    b3 = 2.0 * np.pi * np.cross(lattice[0], lattice[1]) / vol
    return np.stack([b1, b2, b3], axis=0)


def _validate_lattice(value: Any) -> np.ndarray:
    lattice = np.asarray(value, dtype=np.float64)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise ValueError("lattice must be finite 3x3")
    lengths = np.linalg.norm(lattice, axis=1)
    det = float(np.linalg.det(lattice))
    # A right-handed, well-conditioned cell is the supported representation.
    if (np.any(lengths <= 0) or not np.isfinite(det) or det <= 0
            or det / float(np.prod(lengths)) <= 1e-10):
        raise ValueError("singular, left-handed or degenerate lattice")
    return lattice


def _validate_coords(value: Any) -> np.ndarray:
    coords = np.asarray(value, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) == 0 or not np.isfinite(coords).all():
        raise ValueError("fractional_coordinates must be finite nonempty Nx3")
    # Fractional images outside [0,1) are legal; do not silently wrap/reorder.
    return coords


def _validate_species(value: Any) -> list[str]:
    from pymatgen.core.periodic_table import Element
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("species must be a nonempty ordered list")
    if any(not isinstance(s, str) or not Element.is_valid_symbol(s) for s in value):
        raise ValueError("unsupported/invalid chemical species")
    return list(value)


def _validate_structure(record: Dict[str, Any]) -> None:
    _validate_lattice(record.get("lattice"))
    coords = _validate_coords(record.get("fractional_coordinates"))
    species = _validate_species(record.get("species"))
    atoms = _validate_species(record.get("species_per_atom"))
    if len(atoms) != len(coords) or set(atoms) != set(species):
        raise ValueError("species_per_atom count/composition mismatch")
    _validate_conventions(record.get("structure_conventions"))


def _validate_conventions(conventions: Any) -> None:
    if not isinstance(conventions, dict) or (
        conventions.get("lattice_units") not in ("angstrom", "bohr")
        or conventions.get("lattice_vectors") != "rows"
        or conventions.get("coordinates") != "fractional"
        or conventions.get("coordinate_basis") != "direct_lattice"
        or not isinstance(conventions.get("cell_setting"), str)
        or not conventions["cell_setting"].strip()
    ):
        raise ValueError("missing/unsupported structure units or basis conventions")


STRUCTURE_HASH_VERSION = "structure-v2"
LEGACY_HASH_VERSION = "legacy-v1"
SIDECAR_VERSION = "p1-sidecar-v2"


def compute_structure_sha256(record: Dict[str, Any], *, version: str = STRUCTURE_HASH_VERSION) -> str:
    """Hash ordered atoms and explicit representation, not symmetry equivalence.

    v2 retains float64 precision and binds the supplied units/cell basis. A hash
    certifies content integrity, NOT the truth of its source/basis declarations.
    Use ``version="legacy-v1"`` only to replay the historical rounded hash.
    """
    if version == LEGACY_HASH_VERSION:
        return _legacy_structure_sha256(record)
    if version != STRUCTURE_HASH_VERSION:
        raise ValueError(f"unsupported structure hash version: {version}")
    _validate_structure(record)
    canonical = {
        "version": version,
        "lattice": np.asarray(record["lattice"], dtype=np.float64).tolist(),
        "species": list(record["species"]),
        "species_per_atom": list(record["species_per_atom"]),
        "fractional_coordinates": np.asarray(record["fractional_coordinates"], dtype=np.float64).tolist(),
        "structure_conventions": record["structure_conventions"],
    }
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_structure_sha256(record: Dict[str, Any], *, version: str,
                            expected: Optional[str] = None) -> bool:
    """Verify an explicitly named algorithm; never infer v2 from a legacy match."""
    expected = record.get("structure_sha256") if expected is None else expected
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        return False
    try:
        return compute_structure_sha256(record, version=version) == expected
    except (ValueError, TypeError, KeyError, OverflowError):
        return False


def _legacy_structure_sha256(record: Dict[str, Any]) -> str:
    """Historical v1 bytes; deliberately omits per-atom assignment and basis."""
    lattice = record.get("lattice")
    lattice = np.asarray(lattice, dtype=np.float64) if lattice is not None else np.empty((0, 0))
    species = list(record.get("species") or [])
    coords = record.get("fractional_coordinates")
    coords = np.asarray(coords, dtype=np.float64) if coords is not None else np.empty((0, 0))
    canonical = {
        "lattice": np.round(lattice, 8).tolist(),
        "species": species,
        "fractional_coordinates": np.round(coords, 8).tolist(),
    }
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


AFLOW_GEOMETRY_CONVENTIONS = {
    "lattice_units": "angstrom", "lattice_vectors": "rows", "coordinates": "fractional",
    "coordinate_basis": "direct_lattice", "cell_setting": "aflow_geometry_cell",
}


def aflow_auid_to_material_id(auid: Any) -> str:
    """Validate exact provider token before conversion; no trimming/repair."""
    if not isinstance(auid, str) or re.fullmatch(r"aflow:[0-9a-f]{16}", auid) is None:
        raise ValueError("invalid AFLOW AUID: expected aflow: followed by 16 lowercase hex digits")
    return "aflow-" + auid[6:]


def sidecar_record_from_aflow_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Map explicit provider responses to a NEW v2 record; never impute protocol.

    The geometry construction convention is documented here. It is not proof
    that the KPOINTS calculation used that same primitive/conventional cell.
    Malformed values remain in raw_fields and produce invalid field outcomes.
    """
    auid = fields.get("auid")
    material_id = aflow_auid_to_material_id(auid) if auid is not None else None
    lattice = build_lattice_from_geometry(fields.get("geometry"))
    frac = _parse_aflow_list(fields.get("positions_fractional"))
    try:
        coords = np.asarray(frac, dtype=float) if frac is not None else None
    except (ValueError, TypeError):
        coords = frac
    dft_type = _parse_aflow_list(fields.get("dft_type"))
    kpoints = _parse_aflow_list(fields.get("kpoints"))
    kpoints = kpoints if isinstance(kpoints, list) else []
    rec = {
        "material_id": material_id, "auid": fields.get("auid"),
        "source": fields.get("source", "aflow"), "aurl": fields.get("aurl"),
        "lattice": lattice, "fractional_coordinates": coords,
        "species": _parse_aflow_list(fields.get("species")),
        "species_per_atom": fields.get("species_per_atom"),
        "structure_conventions": deepcopy(AFLOW_GEOMETRY_CONVENTIONS),
        "conventions_provenance": "AFLOW geometry/positions_fractional endpoint contract; basis not independently verified",
        "reciprocal_lattice": reciprocal_lattice_from_vectors(lattice) if lattice is not None else None,
        "reciprocal_convention": "rows_2pi_inverse_transpose",
        "dft_type": dft_type,
        "dft_functional": dft_type[0] if isinstance(dft_type, list) and dft_type else None,
        "spin_cell": _parse_aflow_list(fields.get("spin_cell")),
        "spin_atom": _parse_aflow_list(fields.get("spin_atom")),
        "pseudopotential": _parse_aflow_list(fields.get("species_pp")),
        "pseudopotential_version": _parse_aflow_list(fields.get("species_pp_version")),
        "kpath_segments": _parse_aflow_list(fields.get("kpoints_bands_path")),
        "kpoints_relax_mesh": kpoints[0] if kpoints else None,
        "kpoints_static_mesh": kpoints[1] if len(kpoints) > 1 else None,
        "raw_fields": deepcopy(fields),
    }
    for field in ("magnetic_moments", "spin_polarized", "soc", "hubbard_u", "code",
                  "protocol_evidence", "structure_basis_evidence", "kpoint_basis_evidence",
                  "band_kpoint_evidence", "kpath_convention", "kpoints_3d", "field_read_errors",
                  "field_provenance"):
        if field in fields:
            rec[field] = deepcopy(fields[field])
    if lattice is None and not _missing(fields.get("geometry")):
        rec["field_invalid_errors"] = {"lattice": "invalid_source_geometry"}
    return enrich_structure_record(rec)

QUALITY_STATES = ("valid", "ambiguous", "invalid", "read_error")
QUALITY_FIELDS = (
    "material_id", "source", "aurl", "lattice", "species", "species_per_atom",
    "fractional_coordinates", "structure_conventions", "structure_basis_evidence",
    "reciprocal_lattice", "reciprocal_convention", "structure_sha256",
    "spin_cell", "spin_atom", "magnetic_moments", "spin_polarized", "soc",
    "dft_functional", "dft_type", "hubbard_u", "pseudopotential",
    "pseudopotential_version", "code", "protocol_evidence", "kpath_segments",
    "kpath_convention", "kpoints_3d", "kpoints_dense", "kpoint_basis_evidence",
)
BASIC_FIELDS = ("lattice", "species", "species_per_atom", "fractional_coordinates")


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("", "unknown", "null", "none", "n/a")
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return len(value) == 0
    return False


def _combined_status(states) -> str:
    states = set(states)
    for state in ("invalid", "read_error", "ambiguous"):
        if state in states:
            return state
    return "valid"


def _validate_endpoints(bits: Any) -> tuple[int, list]:
    if not isinstance(bits, dict) or bits.get("error"):
        raise ValueError("invalid KPOINTS payload")
    n = bits.get("nkpts")
    segments = bits.get("segments")
    if type(n) is not int or n < 2 or not isinstance(segments, list) or not segments:
        raise ValueError("invalid points-per-segment or empty segments")
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("invalid segment")
        ends = np.asarray([segment.get("k_from"), segment.get("k_to")], dtype=float)
        if ends.shape != (2, 3) or not np.isfinite(ends).all() or np.linalg.norm(ends[1] - ends[0]) <= 0:
            raise ValueError("nonfinite/degenerate segment endpoints")
    return n, segments


def parse_kpoints_bands(text: str) -> Dict[str, Any]:
    """Strict VASP line-mode endpoint syntax, with exact text content digest."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    try:
        if len(lines) < 6 or (len(lines) - 4) % 2:
            raise ValueError("incomplete endpoint pairs")
        if lines[2].split("!")[0].strip().lower() not in ("line-mode", "line_mode", "line mode"):
            raise ValueError("not line-mode")
        mode = lines[3].split("!")[0].strip().lower()
        if mode not in ("reciprocal", "cartesian"):
            raise ValueError("unknown coordinate mode")
        n = int(lines[1].split("!")[0].strip())
        points = []
        for line in lines[4:]:
            coords, _, label = line.partition("!")
            tokens = coords.split()
            if len(tokens) != 3:
                raise ValueError("endpoint must have exactly 3 coordinates")
            points.append(([float(t) for t in tokens], label.strip() or None))
        segments = [{"label_from": points[i][1], "label_to": points[i+1][1],
                     "k_from": points[i][0], "k_to": points[i+1][0]}
                    for i in range(0, len(points), 2)]
        out = {"path_line": lines[0], "nkpts": n, "segments": segments,
               "coordinate_mode": mode, "representation": "line_mode_endpoints",
               "verification_status": "endpoints_only",
               "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
        _validate_endpoints(out)
        return out
    except (ValueError, TypeError, OverflowError) as exc:
        return {"error": str(exc), "verification_status": "invalid"}


def _read_evidence_bytes(path: Any, expected: Any, *, max_bytes: int = 2_000_000) -> bytes:
    if not isinstance(path, str) or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("evidence requires local path and exact SHA256")
    target = Path(path)
    if not target.is_file() or target.stat().st_size > max_bytes:
        raise ValueError("evidence missing or exceeds bounded source size")
    data = target.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("evidence SHA256 mismatch")
    return data


def _verify_kpoint_basis(bits: dict, record: dict, evidence: Any) -> None:
    from pymatgen.io.vasp.inputs import Poscar
    if not isinstance(evidence, dict) or evidence.get("relationship") != "same_calculation_cell" or _missing(evidence.get("source")):
        raise ValueError("same-calculation basis provenance absent")
    _validate_structure(record)
    if record["structure_conventions"]["lattice_units"] != "angstrom":
        raise ValueError("POSCAR comparison requires explicit angstrom structure")
    poscar_bytes = _read_evidence_bytes(evidence.get("poscar_path"), evidence.get("poscar_sha256"))
    kp_bytes = _read_evidence_bytes(evidence.get("kpoints_path"), evidence.get("kpoints_sha256"))
    source_bits = parse_kpoints_bands(kp_bytes.decode("utf-8"))
    if any(source_bits.get(k) != bits.get(k) for k in ("segments", "nkpts", "coordinate_mode", "source_sha256")):
        raise ValueError("KPOINTS source differs from parsed endpoint content")
    poscar_text = poscar_bytes.decode("utf-8")
    source_lines = poscar_text.splitlines()
    if len(source_lines) <= 5:
        raise ValueError("POSCAR explicit element names absent")
    # VASP4 counts-only files cannot certify species: pymatgen may invent H/He.
    # Require actual element symbols in the VASP5 header, not parser defaults.
    _validate_species(source_lines[5].split())
    poscar = Poscar.from_str(poscar_text)
    source = poscar.structure
    if not np.allclose(source.lattice.matrix, record["lattice"], rtol=0, atol=1e-6):
        raise ValueError("POSCAR lattice basis differs; transformation evidence required")
    if [str(site.specie) for site in source] != record["species_per_atom"]:
        raise ValueError("POSCAR ordered species mismatch")
    if source.frac_coords.shape != np.asarray(record["fractional_coordinates"]).shape or not np.allclose(
            source.frac_coords, record["fractional_coordinates"], rtol=0, atol=1e-6):
        raise ValueError("POSCAR ordered fractional coordinates mismatch")


def build_dense_kpath(bits: Dict[str, Any], record: Dict[str, Any], *,
                      basis_evidence: Optional[Dict[str, Any]] = None,
                      band_evidence: Optional[Dict[str, Any]] = None,
                      max_points: int = 1_000_000) -> Dict[str, Any]:
    """Interpolate each line segment independently, retaining duplicate endpoints.

    Endpoints alone cannot establish reciprocal basis or energy-row alignment.
    Unverified physical lengths are None, never zero-filled.
    """
    out = {"schema_version": "p1-dense-kpath-v2", "construction_status": "invalid",
           "verification_status": "unverified", "basis_status": "unverified",
           "alignment_status": "unverified", "fractional_points": None,
           "cartesian_points": None, "segment_lengths": None, "distance": None,
           "length_units": None, "reasons": []}
    try:
        n, segments = _validate_endpoints(bits)
    except (ValueError, TypeError, OverflowError) as exc:
        out["reasons"].append(str(exc))
        return out
    if type(max_points) is not int or max_points < 1 or n * len(segments) > max_points:
        out["reasons"].append("dense_path_allocation_limit_exceeded")
        return out
    out.update({"construction_status": "valid", "segment_count": len(segments),
                "points_per_segment": n, "point_count": n * len(segments),
                "segment_ids": np.repeat(np.arange(len(segments)), n).tolist(),
                "endpoint_policy": "each_segment_inclusive_duplicates_retained"})
    if bits.get("coordinate_mode") != "reciprocal":
        out["reasons"].append("reciprocal_coordinate_mode_not_proven")
        return out
    points = np.concatenate([np.linspace(s["k_from"], s["k_to"], n) for s in segments])
    out["fractional_points"] = points.tolist()
    try:
        _verify_kpoint_basis(bits, record, basis_evidence)
    except (ValueError, TypeError, KeyError, OSError, UnicodeError) as exc:
        out["reasons"].append("basis_source_not_verified: " + str(exc))
        return out
    reciprocal = reciprocal_lattice_from_vectors(_validate_lattice(record["lattice"]))
    cartesian = points @ reciprocal
    lengths = [float(np.linalg.norm(cartesian[(i+1)*n-1] - cartesian[i*n])) for i in range(len(segments))]
    distance = []
    offset = 0.0
    for length in lengths:
        distance.extend((offset + np.linspace(0, length, n)).tolist())
        offset += length
    out.update({"basis_status": "verified", "cartesian_points": cartesian.tolist(),
                "segment_lengths": lengths, "distance": distance,
                "length_units": "angstrom^-1 (2pi)", "basis_evidence": dict(basis_evidence)})
    if not isinstance(band_evidence, dict):
        out["reasons"].append("independent_dense_band_coordinates_not_verified")
        return out
    try:
        band = json.loads(_read_evidence_bytes(band_evidence.get("path"), band_evidence.get("sha256")))
        if not isinstance(band, dict):
            raise ValueError("band coordinate source must be an object")
        if (band.get("material_id") != record.get("material_id")
                or band.get("coordinate_mode") != "reciprocal"
                or band.get("poscar_sha256") != basis_evidence["poscar_sha256"]
                or _missing(band.get("source"))):
            raise ValueError("band point provenance/basis mismatch")
        observed = np.asarray(band.get("fractional_points"), dtype=float)
        segment_ids = band.get("segment_ids")
        if (observed.shape != points.shape or not np.isfinite(observed).all()
                or not np.allclose(observed, points, rtol=0, atol=1e-7)
                or not isinstance(segment_ids, list)
                or any(type(i) is not int for i in segment_ids)
                or segment_ids != out["segment_ids"]):
            raise ValueError("dense point count, segment IDs or pointwise coordinates differ")
    except (ValueError, TypeError, KeyError, OSError, UnicodeError) as exc:
        out["alignment_status"] = "invalid"
        out["reasons"].append("band_alignment_not_verified: " + str(exc))
        return out
    out.update({"alignment_status": "verified", "verification_status": "verified",
                "band_evidence": dict(band_evidence),
                "verification_scope": "local_digest_bound_artifacts_not_external_authenticity"})
    return out


def _resolve_species(record: dict) -> tuple[Any, str]:
    existing = record.get("species_per_atom")
    if existing is not None:
        return existing, record.get("species_assignment_source") or "explicit"
    try:
        species = _validate_species(record.get("species"))
        coords = _validate_coords(record.get("fractional_coordinates"))
    except (ValueError, TypeError):
        return None, "ambiguous"
    if len(species) == 1:
        return species * len(coords), "single_species"
    if len(species) == len(coords):
        return species, "legacy_one_of_each_order_unverified"
    return None, "ambiguous"


PROTOCOL_FIELDS = ("spin_cell", "spin_atom", "magnetic_moments", "spin_polarized", "soc", "hubbard_u",
                   "dft_functional", "dft_type", "pseudopotential", "pseudopotential_version", "code")


def _verify_protocol_snapshot(record: dict) -> None:
    evidence = record.get("protocol_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("protocol source snapshot absent")
    source = json.loads(_read_evidence_bytes(evidence.get("path"), evidence.get("sha256")))
    if not isinstance(source, dict):
        raise ValueError("protocol source must be an object")
    if source.get("material_id") != record.get("material_id") or source.get("source") != record.get("source"):
        raise ValueError("protocol source identity mismatch")
    values = source.get("fields")
    if not isinstance(values, dict):
        raise ValueError("protocol source fields absent")
    for field in PROTOCOL_FIELDS:
        if _missing(record.get(field)) or field not in values or values[field] != record[field]:
            raise ValueError(f"protocol field not corroborated: {field}")


def audit_structure_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only field QA. Numerical validity is not independent source proof.

    Missing/unknown is ambiguous, not zero/false. Legacy digest agreement does
    not validate atom assignment. The basic and full contracts stay separate.
    """
    record = dict(record)
    if record.get("read_error"):
        return {"material_id": record.get("material_id"), "status": "read_error",
                "basic_structure_status": "read_error", "missing_fields": list(QUALITY_FIELDS),
                "field_quality": {f: {"status": "read_error", "reason": str(record["read_error"])} for f in QUALITY_FIELDS}}
    record["species_per_atom"], assignment_source = _resolve_species(record)
    quality = {}
    missing = []
    for field in QUALITY_FIELDS:
        value = record.get(field)
        if _missing(value):
            quality[field] = {"status": "ambiguous", "reason": "missing_or_unknown"}
            missing.append(field)
        else:
            quality[field] = {"status": "valid", "reason": "present"}

    def mark(field, status, reason):
        quality[field] = {"status": status, "reason": reason}

    validators = {"lattice": _validate_lattice, "species": _validate_species,
                  "species_per_atom": _validate_species, "fractional_coordinates": _validate_coords,
                  "structure_conventions": _validate_conventions}
    for field, validator in validators.items():
        if field not in missing:
            try:
                validator(record[field])
                mark(field, "valid", "numerically_valid" if field != "structure_conventions" else "explicit_convention")
            except (ValueError, TypeError, OverflowError):
                mark(field, "invalid", "invalid_shape_value_or_convention")
    if all(quality[f]["status"] == "valid" for f in ("species", "species_per_atom", "fractional_coordinates")):
        if (len(record["species_per_atom"]) != len(record["fractional_coordinates"])
                or set(record["species_per_atom"]) != set(record["species"])):
            mark("species_per_atom", "invalid", "atom_count_or_composition_mismatch")
    for field in ("spin_cell", "spin_atom", "magnetic_moments"):
        if field in missing:
            continue
        try:
            if any(isinstance(v, (bool, np.bool_)) for v in np.asarray(record[field], dtype=object).flat):
                raise ValueError("boolean is not a magnetic moment")
            values = np.asarray(record[field], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("nonfinite")
            if field == "spin_cell":
                if values.ndim != 0:
                    raise ValueError("not_scalar")
            elif values.ndim == 0:
                mark(field, "ambiguous", "scalar_summary_not_per_atom_moments")
            elif values.shape not in ((len(record.get("fractional_coordinates", [])),),
                                      (len(record.get("fractional_coordinates", [])), 3)):
                raise ValueError("atom_count")
        except (TypeError, ValueError):
            mark(field, "invalid", "nonfinite_or_wrong_moment_shape")
    if "reciprocal_lattice" not in missing:
        try:
            reciprocal = _validate_lattice(record["reciprocal_lattice"])
            lattice = _validate_lattice(record.get("lattice"))
            if not np.allclose(lattice @ reciprocal.T, 2 * np.pi * np.eye(3), rtol=1e-7, atol=1e-7):
                raise ValueError("reciprocal mismatch")
            mark("reciprocal_lattice", "valid", "derived_duality_not_independent_source_check")
        except (ValueError, TypeError, OverflowError):
            mark("reciprocal_lattice", "invalid", "invalid_reciprocal_or_direct_duality")
    if "reciprocal_convention" not in missing and record["reciprocal_convention"] != "rows_2pi_inverse_transpose":
        mark("reciprocal_convention", "invalid", "unsupported_reciprocal_convention")
    if record.get("auid") is not None:
        try:
            if aflow_auid_to_material_id(record["auid"]) != record.get("material_id"):
                raise ValueError("AUID identity mismatch")
        except ValueError:
            mark("material_id", "invalid", "invalid_AUID_or_material_id_mismatch")
    for field in ("soc", "spin_polarized"):
        if field not in missing and type(record[field]) is not bool:
            mark(field, "invalid", "explicit_boolean_required")
    for field in ("dft_functional", "code", "kpath_convention", "source", "aurl", "material_id"):
        if field not in missing and not isinstance(record[field], str):
            mark(field, "invalid", "nonempty_string_required")
    for field in ("dft_type", "pseudopotential", "pseudopotential_version", "kpath_segments"):
        if field not in missing:
            value = record[field]
            if not isinstance(value, list) or any(not isinstance(v, str) or _missing(v) for v in value):
                mark(field, "invalid", "nonempty_string_list_required")
    if quality["species"]["status"] == "valid":
        for field in ("pseudopotential", "pseudopotential_version"):
            if quality[field]["status"] == "valid" and len(record[field]) != len(set(record["species"])):
                mark(field, "invalid", "pseudopotential_species_cardinality_mismatch")
    if "hubbard_u" not in missing:
        value = record["hubbard_u"]
        if not isinstance(value, dict) or type(value.get("enabled")) is not bool:
            mark("hubbard_u", "ambiguous", "explicit_enabled_and_per_species_protocol_required")
        elif value["enabled"]:
            try:
                values = np.asarray(value.get("values_eV"), dtype=float)
                if values.ndim != 1 or not values.size or not np.isfinite(values).all():
                    raise ValueError("invalid U values")
                if not value.get("species") or len(value["species"]) != len(values) or not value.get("formulation"):
                    mark("hubbard_u", "ambiguous", "U_species_or_formulation_unknown")
            except (ValueError, TypeError):
                mark("hubbard_u", "invalid", "invalid_U_values")
    for field in ("structure_basis_evidence", "protocol_evidence", "kpoint_basis_evidence", "kpoints_dense"):
        if field not in missing:
            mark(field, "ambiguous", "external_evidence_not_reverified")
    if "kpoints_3d" not in missing:
        try:
            _validate_endpoints(record["kpoints_3d"])
            mark("kpoints_3d", "valid", "endpoints_only_not_dense_alignment")
        except (ValueError, TypeError, OverflowError):
            mark("kpoints_3d", "invalid", "invalid_segment_endpoints")
    if "protocol_evidence" not in missing:
        try:
            _verify_protocol_snapshot(record)
            mark("protocol_evidence", "valid", "local_digest_bound_protocol_snapshot_matches_not_external_authenticity")
        except (ValueError, TypeError, KeyError, OSError, UnicodeError):
            mark("protocol_evidence", "ambiguous", "protocol_source_not_corroborated")
    bits = record.get("kpoints_3d")
    if quality["kpoints_3d"]["status"] == "valid" and quality["kpath_segments"]["status"] == "valid":
        if len(record["kpath_segments"]) != len(bits["segments"]):
            mark("kpath_segments", "invalid", "segment_count_differs_from_endpoints")
    for field in ("structure_basis_evidence", "kpoint_basis_evidence"):
        if field not in missing:
            try:
                _verify_kpoint_basis(bits or {}, record, record[field])
                mark(field, "valid", "local_POSCAR_KPOINTS_and_structure_basis_agree")
            except (ValueError, TypeError, KeyError, OSError, UnicodeError):
                mark(field, "ambiguous", "local_basis_source_not_verified")
    if "kpoints_dense" not in missing:
        dense = build_dense_kpath(bits or {}, record, basis_evidence=record.get("kpoint_basis_evidence"),
                                 band_evidence=record.get("band_kpoint_evidence"))
        if dense["alignment_status"] == "invalid":
            mark("kpoints_dense", "invalid", "dense_band_point_alignment_failed")
        elif dense["verification_status"] == "verified":
            saved = record["kpoints_dense"]
            try:
                if not isinstance(saved, dict):
                    raise ValueError("invalid dense object")
                for key in ("fractional_points", "cartesian_points", "segment_lengths", "distance", "segment_ids"):
                    actual = np.asarray(saved.get(key), dtype=float)
                    expected = np.asarray(dense[key], dtype=float)
                    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=1e-7):
                        raise ValueError("stored dense content differs")
                if any(saved.get(k) != dense[k] for k in ("point_count", "segment_count", "points_per_segment", "length_units")):
                    raise ValueError("stored dense metadata differs")
                mark("kpoints_dense", "valid", "dense_points_segments_lengths_and_local_source_reverified")
            except (ValueError, TypeError):
                mark("kpoints_dense", "invalid", "stored_dense_content_mismatch")
    if "structure_sha256" not in missing:
        version = record.get("structure_hash_version", LEGACY_HASH_VERSION)
        if not verify_structure_sha256(record, version=version):
            mark("structure_sha256", "invalid", "digest_mismatch_or_invalid_structure")
        elif version == LEGACY_HASH_VERSION:
            mark("structure_sha256", "ambiguous", "legacy_digest_omits_atoms_and_basis")
        else:
            mark("structure_sha256", "valid", "v2_content_digest_matches_not_source_certification")
    invalid_errors = record.get("field_invalid_errors")
    if isinstance(invalid_errors, dict):
        for field, error in invalid_errors.items():
            if field in quality:
                mark(field, "invalid", str(error))
    input_hash = record.get("input_hash_verification")
    if isinstance(input_hash, dict) and input_hash.get("valid") is False:
        mark("structure_sha256", "invalid", "upstream_input_digest_mismatch")
    errors = record.get("field_read_errors")
    if isinstance(errors, dict):
        for field, error in errors.items():
            if field in quality and field in missing:
                mark(field, "read_error", str(error))
    return {
        "material_id": record.get("material_id"),
        "status": "invalid" if record.get("merge_conflicts") else _combined_status(q["status"] for q in quality.values()),
        "basic_structure_status": _combined_status(quality[f]["status"] for f in BASIC_FIELDS),
        "missing_fields": missing, "field_quality": quality,
        "species_assignment_source": assignment_source,
        "merge_conflicts": _json_safe(record.get("merge_conflicts", [])),
    }


def enrich_structure_record(record: Dict[str, Any], *, conventions: Optional[dict] = None) -> Dict[str, Any]:
    """Create a v2 derivative; legacy input and unknown protocol remain untouched.

    Legacy records without explicit units/basis do not receive a v2 digest.
    ``conventions`` is an explicit caller declaration, not independent proof.
    """
    out = deepcopy(record)
    out["schema_version"] = SIDECAR_VERSION
    old_hash = record.get("structure_sha256")
    old_version = record.get("structure_hash_version", LEGACY_HASH_VERSION)
    if old_hash:
        verification = {"version": old_version, "sha256": old_hash,
                        "valid": verify_structure_sha256(record, version=old_version)}
        # Retain an upstream failure across repeated enrichment.
        if not isinstance(out.get("input_hash_verification"), dict) or out["input_hash_verification"].get("valid") is not False:
            out["input_hash_verification"] = verification
    if old_version == LEGACY_HASH_VERSION:
        out["legacy_structure_sha256"] = old_hash
        out["legacy_hash_verified"] = verify_structure_sha256(record, version=LEGACY_HASH_VERSION) if old_hash else None
    if conventions is not None:
        _validate_conventions(conventions)
        if not _missing(out.get("structure_conventions")) and out["structure_conventions"] != conventions:
            raise ValueError("cannot overwrite existing structure conventions")
        out["structure_conventions"] = deepcopy(conventions)
        out["conventions_provenance"] = "explicit_caller_declaration_not_independent_verification"
    out["species_per_atom"], out["species_assignment_source"] = _resolve_species(out)
    for field in QUALITY_FIELDS:
        out.setdefault(field, None)
    out["structure_hash_version"] = STRUCTURE_HASH_VERSION
    try:
        out["structure_sha256"] = compute_structure_sha256(out)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        out["structure_sha256"] = None
        out["structure_hash_error"] = str(exc)
    if _missing(out.get("kpoints_dense")) and not _missing(out.get("kpoints_3d")):
        out["kpoints_dense"] = build_dense_kpath(out["kpoints_3d"], out,
            basis_evidence=out.get("kpoint_basis_evidence"), band_evidence=out.get("band_kpoint_evidence"))
    out["quality"] = audit_structure_record(out)
    out["missing_fields"] = out["quality"]["missing_fields"]
    return out


def _json_safe(value: Any) -> Any:
    """Preserve invalid nonfinite values as tagged data, never zero/None imputation."""
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {"__p1_nonfinite_float__": "NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")}
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def build_quality_ledger(records: list, requested_ids) -> Dict[str, Any]:
    """One outcome per fixed requested ID; extra/unidentified/duplicate rows stay visible.

    No HDF5 datasets are read. ``field_present`` is retained as a compatibility
    key but means VALID field count, not ``is not None`` coverage.
    """
    ids = list(requested_ids)
    if any(not isinstance(mid, str) or not mid or mid != mid.strip() for mid in ids) or len(set(ids)) != len(ids):
        raise ValueError("requested IDs must be unique nonempty exact strings")
    requested = set(ids)
    by_id = {}
    unidentified = []
    for index, record in enumerate(records):
        mid = record.get("material_id") if isinstance(record, dict) else None
        if not isinstance(mid, str) or not mid or mid != mid.strip():
            unidentified.append(index)
            continue
        by_id.setdefault(mid, []).append(record)
    duplicate_ids = sorted(mid for mid, rows in by_id.items() if len(rows) > 1)
    missing_ids = sorted(requested - by_id.keys())
    extra_ids = sorted(by_id.keys() - requested)
    outcomes = []
    for mid in sorted(requested):
        rows = by_id.get(mid, [])
        blocked = ("invalid", "duplicate_id") if len(rows) > 1 else None
        if not rows:
            blocked = ("read_error", "requested_record_absent")
        elif len(rows) == 1 and rows[0].get("read_error"):
            blocked = ("read_error", str(rows[0]["read_error"]))
        if blocked:
            status, reason = blocked
            outcome = {"material_id": mid, "status": status, "basic_structure_status": status,
                       "missing_fields": list(QUALITY_FIELDS),
                       "field_quality": {f: {"status": status, "reason": reason} for f in QUALITY_FIELDS}}
        else:
            outcome = audit_structure_record(rows[0])
        if len(rows) == 1:
            outcome["source"] = _json_safe(rows[0].get("source"))
            outcome["aurl"] = _json_safe(rows[0].get("aurl"))
            outcome["protocol"] = _json_safe({f: rows[0].get(f) for f in (
                "dft_functional", "dft_type", "spin_polarized", "soc", "hubbard_u",
                "pseudopotential", "pseudopotential_version", "code", "protocol_evidence")})
        outcomes.append(outcome)
    counts = {s: sum(r["status"] == s for r in outcomes) for s in QUALITY_STATES}
    field_counts = {f: {s: sum(r["field_quality"][f]["status"] == s for r in outcomes)
                          for s in QUALITY_STATES} for f in QUALITY_FIELDS}
    integrity = {"valid": not (duplicate_ids or missing_ids or extra_ids or unidentified),
                 "duplicate_ids": duplicate_ids, "missing_ids": missing_ids,
                 "extra_ids": extra_ids, "unidentified_row_indices": unidentified,
                 "input_rows": len(records),
                 "matched_rows": sum(len(rows) for mid, rows in by_id.items() if mid in requested),
                 "extra_rows": sum(len(by_id[mid]) for mid in extra_ids),
                 "unidentified_rows": len(unidentified)}
    assert sum(counts.values()) == len(requested)
    assert all(sum(c.values()) == len(requested) for c in field_counts.values())
    assert integrity["matched_rows"] + integrity["extra_rows"] + integrity["unidentified_rows"] == len(records)
    return {
        "schema_version": "p1-quality-ledger-v2", "requested": len(requested), **counts,
        "id_integrity": integrity, "sidecar_records": len(records), "hdf5_groups": len(requested),
        "h5_ids_missing_in_sidecar": len(missing_ids), "sidecar_ids_not_in_h5": len(extra_ids),
        "field_present_definition": "valid according to field contract; not merely non-null",
        "field_present": {f: c["valid"] for f, c in field_counts.items()},
        "field_quality": field_counts,
        "missing_field_breakdown": {f: sum(f in r["missing_fields"] for r in outcomes) for f in QUALITY_FIELDS},
        "basic_structure_counts": {s: sum(r["basic_structure_status"] == s for r in outcomes) for s in QUALITY_STATES},
        "full_contract_passed": integrity["valid"] and counts["valid"] == len(requested),
        "verification_scope": "field_contract_and_local_digest_bound_evidence_not_external_authenticity",
        "records": outcomes,
    }


def _index_unique_records(records: list) -> dict:
    indexed = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("record must be an object with an exact material_id")
        mid = record.get("material_id")
        if "auid" in record and record["auid"] is not None:
            expected = aflow_auid_to_material_id(record["auid"])
            if mid is not None and mid != expected:
                raise ValueError("AUID/material_id disagreement")
            if mid is None:
                record = sidecar_record_from_aflow_fields(record)
                mid = expected
        if not isinstance(mid, str) or not mid or mid != mid.strip():
            raise ValueError("invalid exact material_id")
        if mid in indexed:
            raise ValueError(f"duplicate material_id: {mid}")
        indexed[mid] = record
    return indexed


def merge_sidecar_records(existing: list, incoming: list, requested_ids, *, conventions=None) -> tuple[list, dict]:
    """New-version fixed-ID nonempty merge; duplicates fail, conflicts quarantine.

    Missing fixed IDs become explicit read-error placeholders, NOT fabricated
    structures. Original and incoming out-of-scope IDs are reported separately.
    """
    ids = list(requested_ids)
    if any(not isinstance(mid, str) or not mid or mid != mid.strip() for mid in ids) or len(set(ids)) != len(ids):
        raise ValueError("requested IDs must be unique exact strings")
    fixed = set(ids)
    old = _index_unique_records(existing)
    new = _index_unique_records(incoming)
    merged = []
    derived = {"schema_version", "quality", "missing_fields", "structure_hash_version", "structure_sha256",
               "legacy_structure_sha256", "legacy_hash_verified", "input_hash_verification",
               "structure_hash_error", "kpoints_dense", "raw_fields"}
    for mid in sorted(fixed):
        if mid not in old and mid not in new:
            record = {"material_id": mid, "read_error": "fixed_ID_not_available_in_original_or_incoming"}
        elif mid not in old:
            record = deepcopy(new[mid])
        else:
            record = deepcopy(old[mid])
            if mid in new:
                if record.get("read_error") and not new[mid].get("read_error") and any(
                        not _missing(new[mid].get(f)) for f in BASIC_FIELDS):
                    error = record.pop("read_error")
                    if error not in record.setdefault("read_error_history", []):
                        record["read_error_history"].append(error)
                for field, value in new[mid].items():
                    if field in derived or _missing(value):
                        continue
                    if _missing(record.get(field)):
                        record[field] = deepcopy(value)
                    else:
                        def serialize(v):
                            return json.dumps(v, sort_keys=True, default=lambda a: a.tolist() if isinstance(a, np.ndarray) else a.item())
                        if serialize(record[field]) != serialize(value):
                            conflict = {"field": field, "existing": record[field], "incoming": value,
                                        "incoming_source": new[mid].get("source")}
                            if conflict not in record.setdefault("merge_conflicts", []):
                                record["merge_conflicts"].append(deepcopy(conflict))
                if new[mid].get("raw_fields"):
                    record.setdefault("incoming_provenance", []).append(deepcopy(new[mid]["raw_fields"]))
        merged.append(enrich_structure_record(record, conventions=conventions))
    report = build_quality_ledger(merged, ids)
    report.update({"clipped_existing_ids": sorted(old.keys() - fixed),
                   "clipped_incoming_ids": sorted(new.keys() - fixed),
                   "original_missing_ids": sorted(fixed - old.keys()),
                   "gap_fill_added": len((new.keys() - old.keys()) & fixed)})
    return merged, report


def assert_new_outputs(paths, *, inputs=()) -> None:
    """Fail before reads/network if any output aliases an input or exists."""
    resolved = [Path(p).resolve() for p in paths]
    if len(set(resolved)) != len(resolved) or set(resolved) & {Path(p).resolve() for p in inputs if p}:
        raise ValueError("outputs must be distinct new paths, never input paths")
    for path in resolved:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"protected existing output: {path}")


def atomic_write_json(path, payload) -> None:
    """Create-only same-directory atomic publication, with file and directory fsync.

    Hard-link publication is exclusive even under concurrent writers. No --force
    path exists: changing versions means choosing a new output filename.
    """
    target = Path(path)
    assert_new_outputs([target])
    target.parent.mkdir(parents=True, exist_ok=True)
    def default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"unsupported JSON value: {type(value).__name__}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent,
                                         prefix="." + target.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, default=default)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        temporary.unlink()
        temporary = None
        if os.name == "posix":
            fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_fixed_ids(*, h5_path=None, ids_json=None) -> list[str]:
    """Read fixed IDs only; never inspect an HDF5 dataset or energy array."""
    if bool(h5_path) == bool(ids_json):
        raise ValueError("supply exactly one of h5_path and ids_json")
    if h5_path:
        import h5py
        with h5py.File(h5_path, "r") as handle:
            ids = list(handle.keys())
    else:
        with open(ids_json, encoding="utf-8") as stream:
            ids = json.load(stream)
    if not isinstance(ids, list) or any(not isinstance(mid, str) or not mid or mid != mid.strip() for mid in ids) or len(set(ids)) != len(ids):
        raise ValueError("fixed IDs must be unique exact nonempty strings")
    return ids


def _unique_json_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON object key: {key}")
        out[key] = value
    return out


def load_sidecar_records(path) -> list:
    with open(path, encoding="utf-8") as stream:
        records = json.load(stream, object_pairs_hook=_unique_json_object)
    if not isinstance(records, list):
        raise ValueError("sidecar must be a JSON list of records")
    return records


class StructureSidecarSchema:
    """Serialize/deserialize P1 sidecar records to a JSON-compatible form.

    Arrays are stored as lists; invalid nonfinite values use explicit tagged
    JSON rather than zeros or nonstandard NaN literals. Audit the actual values
    again: a stored missing_fields list or quality claim is not certification.
    """

    def serialize(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return _json_safe(record)

    def deserialize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(payload)
        for key in ("lattice", "fractional_coordinates", "reciprocal_lattice"):
            value = payload.get(key)
            if value is not None:
                out[key] = np.asarray(value, dtype=np.float64)
        return out
