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
import hashlib
import json
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
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(max(0.0, c * c - cx * cx - cy * cy))
    a3 = np.array([cx, cy, cz])
    return np.stack([a1, a2, a3], axis=0).astype(np.float64)


def build_lattice_from_geometry(geometry: Any) -> Optional[np.ndarray]:
    vals = _parse_aflow_list(geometry)
    if not vals or len(vals) < 6:
        return None
    a, b, c, alpha, beta, gamma = (float(v) for v in vals[:6])
    if min(a, b, c) <= 0.0:
        return None
    return geometry_to_lattice_vectors(a, b, c, alpha, beta, gamma)


def reciprocal_lattice_from_vectors(lattice: np.ndarray) -> np.ndarray:
    """Reciprocal lattice rows b_i (a_i . b_j = 2*pi*delta_ij)."""
    vol = np.dot(lattice[0], np.cross(lattice[1], lattice[2]))
    if abs(vol) < 1e-12:
        raise ValueError("zero cell volume")
    b1 = 2.0 * np.pi * np.cross(lattice[1], lattice[2]) / vol
    b2 = 2.0 * np.pi * np.cross(lattice[2], lattice[0]) / vol
    b3 = 2.0 * np.pi * np.cross(lattice[0], lattice[1]) / vol
    return np.stack([b1, b2, b3], axis=0)


def compute_structure_sha256(record: Dict[str, Any]) -> str:
    """Deterministic structure hash over lattice + species + fractional coords."""
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


def sidecar_record_from_aflow_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Map one AFLUX/AFLOW field bundle to a P1 sidecar record.

    `fields` keys mirror the AFLUX property names (geometry, species,
    positions_fractional, dft_type, spin_cell, spin_atom, species_pp,
    kpoints_bands_path, kpoints, code). auid may be 'aflow:...'.
    """
    auid = str(fields.get("auid") or "").strip()
    material_id = "aflow-" + auid.split(":", 1)[-1] if auid else None

    geometry = _parse_aflow_list(fields.get("geometry"))
    lattice = build_lattice_from_geometry(geometry)
    species = _parse_aflow_list(fields.get("species"))
    frac = _parse_aflow_list(fields.get("positions_fractional"))
    dft_type = _parse_aflow_list(fields.get("dft_type"))
    spin_cell = fields.get("spin_cell")
    if spin_cell is not None and not isinstance(spin_cell, (int, float)):
        spin_cell = _parse_aflow_list(spin_cell)
    spin_atom = _parse_aflow_list(fields.get("spin_atom"))
    species_pp = _parse_aflow_list(fields.get("species_pp"))
    kpath = _parse_aflow_list(fields.get("kpoints_bands_path"))
    kpoints = _parse_aflow_list(fields.get("kpoints"))

    missing = []
    if material_id is None:
        missing.append("material_id")
    if lattice is None:
        missing.append("lattice")
    if species is None:
        missing.append("species")
    if frac is None:
        missing.append("fractional_coordinates")
    if dft_type is None:
        missing.append("dft_functional")
    if spin_cell is None:
        missing.append("spin_cell")
    if species_pp is None:
        missing.append("pseudopotential")
    if kpath is None:
        missing.append("kpath_segments")

    record = {
        "material_id": material_id,
        "lattice": lattice,
        "species": species,
        "fractional_coordinates": np.asarray(frac, dtype=np.float64)
        if frac is not None else None,
        "dft_functional": dft_type[0] if dft_type else None,
        "spin_cell": spin_cell if isinstance(spin_cell, (int, float)) else None,
        "spin_atom": spin_atom,
        "pseudopotential": species_pp,
        "kpath_segments": kpath,
        "kpoints_relax_mesh": kpoints[0] if kpoints and len(kpoints) > 0 else None,
        "kpoints_static_mesh": kpoints[1] if kpoints and len(kpoints) > 1 else None,
        "code": str(fields.get("code") or "").strip() or None,
        "aurl": str(fields.get("aurl") or "").strip() or None,
        "source": "aflow",
        "missing_fields": missing,
    }
    if lattice is not None:
        try:
            record["reciprocal_lattice"] = reciprocal_lattice_from_vectors(lattice)
        except ValueError:
            record["reciprocal_lattice"] = None
            missing.append("reciprocal_lattice")
    if lattice is not None and species and frac is not None:
        record["structure_sha256"] = compute_structure_sha256(record)
    else:
        record["structure_sha256"] = None
    return record


class StructureSidecarSchema:
    """Serialize/deserialize P1 sidecar records to a JSON-compatible form.

    NumPy arrays are round-tripped as lists; the `missing_fields` audit list
    is preserved so coverage can be reported per-field without inspecting
    every record again.
    """

    def serialize(self, record: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for key, value in record.items():
            if isinstance(value, np.ndarray):
                out[key] = value.tolist()
            else:
                out[key] = value
        return out

    def deserialize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(payload)
        for key in ("lattice", "fractional_coordinates", "reciprocal_lattice"):
            value = payload.get(key)
            if value is not None:
                out[key] = np.asarray(value, dtype=np.float64)
        return out
