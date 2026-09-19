"""
Physics validation utilities for reconstructed or predicted band tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Dict

import numpy as np


def local_curvature(
    band: np.ndarray,
    segment_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Local quadratic curvature via ±2 windows confined to one k-path segment."""
    band = np.asarray(band, dtype=np.float64)
    n = len(band)
    if segment_ids is None:
        segments = np.zeros(n, dtype=np.int32)
    else:
        segments = np.asarray(segment_ids, dtype=np.int32)
        if segments.shape != (n,):
            raise ValueError(f"Expected segment_ids shape {(n,)}, got {segments.shape}")
    curv = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - 2)
        hi = min(n, i + 3)
        indices = np.arange(lo, hi, dtype=np.int32)
        indices = indices[segments[indices] == segments[i]]
        x_local = indices.astype(np.float64) - float(i)
        y_local = band[indices]
        if len(indices) >= 3:
            coeff = np.polyfit(x_local, y_local, deg=2)
            curv[i] = 2.0 * coeff[0]
        elif 0 < i < n - 1 and segments[i - 1] == segments[i] == segments[i + 1]:
            curv[i] = band[i + 1] - 2.0 * band[i] + band[i - 1]
    return curv.astype(np.float32)


def _local_curvature(band: np.ndarray) -> np.ndarray:
    """Backward-compatible unsegmented wrapper."""
    return local_curvature(band)


@dataclass
class PhysicsValidator:
    """Score basic physical consistency of VBM/CBM tensors."""

    violation_tolerance: float = 0.05

    def validate(
        self,
        predicted_gap: np.ndarray,
        band_tensors: np.ndarray,
        segment_ids: np.ndarray | None = None,
    ) -> Dict[str, float]:
        predicted_gap = np.asarray(predicted_gap, dtype=np.float32).reshape(-1)
        tensors = np.asarray(band_tensors, dtype=np.float32)

        if tensors.ndim != 4 or tensors.shape[1] != 2:
            raise ValueError(f"Expected band_tensors with shape (N, 2, K, C), got {tensors.shape}")
        if segment_ids is None:
            segments = np.zeros((len(tensors), tensors.shape[2]), dtype=np.int32)
        else:
            segments = np.asarray(segment_ids, dtype=np.int32)
            expected = (len(tensors), tensors.shape[2])
            if segments.shape != expected:
                raise ValueError(
                    f"Expected segment_ids with shape {expected}, got {segments.shape}"
                )

        vbm_energy = tensors[:, 0, :, 0]
        cbm_energy = tensors[:, 1, :, 0]
        vbm_center = np.argmax(vbm_energy, axis=1)
        cbm_center = np.argmin(cbm_energy, axis=1)

        vbm_curvature = np.asarray(
            [
                local_curvature(row, segment_ids=segments[index])
                for index, row in enumerate(vbm_energy)
            ]
        )
        cbm_curvature = np.asarray(
            [
                local_curvature(row, segment_ids=segments[index])
                for index, row in enumerate(cbm_energy)
            ]
        )
        vbm_at_extreme = np.asarray([vbm_curvature[i, vbm_center[i]] for i in range(len(tensors))])
        cbm_at_extreme = np.asarray([cbm_curvature[i, cbm_center[i]] for i in range(len(tensors))])

        tensor_gap = np.min(cbm_energy, axis=1) - np.max(vbm_energy, axis=1)
        residual = predicted_gap - tensor_gap

        violations = {
            "negative_predicted_gap_rate": float(np.mean(predicted_gap < 0.0)),
            "vbm_positive_curvature_rate": float(np.mean(vbm_at_extreme > 0.0)),
            "cbm_negative_curvature_rate": float(np.mean(cbm_at_extreme < 0.0)),
            "gap_identity_large_residual_rate": float(
                np.mean(np.abs(residual) > max(self.violation_tolerance, 1e-6))
            ),
        }
        violation_rate = float(max(violations.values()))
        return {
            **violations,
            "gap_identity_mae": float(np.mean(np.abs(residual))),
            "physics_violation_rate": violation_rate,
            "physics_score": float(max(0.0, 1.0 - violation_rate)),
        }

    def physics_score(
        self,
        predicted_gap: np.ndarray,
        band_tensors: np.ndarray,
        segment_ids: np.ndarray | None = None,
    ) -> float:
        return self.validate(
            predicted_gap,
            band_tensors,
            segment_ids=segment_ids,
        )["physics_score"]


def _numerical_array(value) -> np.ndarray:
    """Reject coercible strings/bools as well as nonfinite or huge values."""
    if isinstance(value, np.ndarray):
        if value.ndim > 2 or value.size > 262144 or value.dtype.kind not in "iuf":
            raise ValueError("invalid_numeric")
    else:
        pending = [(value, 0)]
        leaves = 0
        while pending:
            item, depth = pending.pop()
            if isinstance(item, (list, tuple)):
                if depth >= 2 or len(item) > 8192:
                    raise ValueError("invalid_numeric")
                pending.extend((child, depth + 1) for child in item)
            else:
                leaves += 1
                if (
                    leaves > 262144
                    or isinstance(item, (bool, np.bool_))
                    or not isinstance(item, Real)
                ):
                    raise ValueError("invalid_numeric")
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any(np.abs(array) > 1e6):
        raise ValueError("invalid_numeric")
    return array


def _directional_mass(
    energies, cartesian, distance, k_unit, segments, location, carrier, energy_resolution
):
    """Local five-sample physical-coordinate quadratic fit, not a tensor mass."""
    from scipy.constants import hbar, electron_mass, electron_volt

    k = location["k_index"]
    if not 2 <= k < energies.shape[1] - 2:
        return None, {"reason": "endpoint_or_insufficient_points"}
    indices = np.arange(k - 2, k + 3)
    if not np.all(segments[indices] == segments[k]):
        return None, {"reason": "endpoint_or_insufficient_points"}
    offsets = cartesian[indices] - cartesian[k]
    direction = offsets[-1] - offsets[0]
    length = np.linalg.norm(direction)
    if length <= 1e-12:
        return None, {"reason": "nonstraight_or_retraced_k_window"}
    direction /= length
    x = offsets @ direction
    span = np.max(np.abs(x))
    # Check the bend in local dimensionless coordinates, independent of k scale.
    residual_geometry = offsets / length - (x / length)[:, None] * direction
    if np.any(np.diff(x) <= 1e-12) or np.max(np.linalg.norm(residual_geometry, axis=1)) > 1e-8:
        return None, {"reason": "nonstraight_or_retraced_k_window"}
    if k_unit == "angstrom^-1" and not np.allclose(
        np.diff(distance[indices]) / np.diff(x), 1.0, rtol=1e-5, atol=0.0
    ):
        return None, {"reason": "inconsistent_physical_k_distance"}
    u = x / span
    band_index = location["band_index"]
    y = energies[band_index, indices]
    differences = np.delete(energies[:, indices], band_index, axis=0) - y
    if np.any(np.abs(differences) <= 1e-6) or np.any(
        np.signbit(differences[:, 1:]) != np.signbit(differences[:, :-1])
    ):
        return None, {"reason": "nonisolated_or_crossing_branch"}
    design = np.column_stack((u * u, u, np.ones(5)))
    energy_scale = np.ptp(y)
    if energy_scale <= 0:
        return None, {"reason": "nonquadratic_or_unstable_curvature"}
    # Dimensionless fits: an absolute eV tolerance cannot certify a small curvature.
    centered = y - y[2]
    normalized = centered / energy_scale
    coefficients = np.linalg.lstsq(design, normalized, rcond=None)[0]
    curvature = 2.0 * coefficients[0] * energy_scale / span**2
    central = np.linalg.lstsq(design[1:4], normalized[1:4], rcond=None)[0]
    # Propagate a conservative one-ULP input/subtraction bound to the quadratic
    # coefficient, then to its reciprocal mass. This is not calibrated uncertainty.
    roundoff = energy_resolution[band_index, indices] + np.spacing(np.abs(centered))
    coefficient_error = np.abs(np.linalg.pinv(design)[0]) @ (roundoff / energy_scale)
    expected_sign = 1.0 if carrier == "electron" else -1.0
    residual = float(np.max(np.abs(design @ coefficients - normalized)))
    vertex = -coefficients[1] / (2.0 * coefficients[0]) if coefficients[0] != 0 else np.inf
    if (
        expected_sign * curvature <= 1e-10
        or not np.all(expected_sign * (y[[1, 3]] - y[2]) > 0)
        or not u[1] < vertex < u[3]
        or np.linalg.cond(design) > 1e8
        or residual > 1e-5
        or coefficient_error > 1e-5 * (abs(coefficients[0]) - coefficient_error)
        or not np.isclose(coefficients[0], central[0], rtol=1e-5, atol=0.0)
    ):
        return None, {"reason": "nonquadratic_or_unstable_curvature"}
    mass = hbar**2 / (electron_mass * electron_volt * 1e-20 * abs(curvature))
    return float(mass), {
        "reason": "local_quadratic_fit",
        "fit_k_indices": indices.tolist(),
        "direction_cartesian": direction.tolist(),
        "curvature_eV_angstrom2": float(curvature),
    }


def analyze_numerical_band_data(data: dict, *, reference_from_roles: bool = False) -> dict:
    """Measure a supplied line-mode spectrum; never predict or calibrate it.

    Required input: energies_eV[B,K], k_distance[K], contiguous-run integer
    segment_ids[K], finite scalar fermi_eV, k_unit='relative'|'angstrom^-1'.
    Optional: band_roles[B] ('valence'|'conduction'), k_cartesian_invA[K,3].
    Bands are input trajectories, not eigenvector-verified branch identities.
    Limits: 1<=B<=4096, 2<=K<=8192, B*K<=262144; numeric magnitudes <=1e6.
    Internal reference_from_roles mode requires explicit roles and reference_eV;
    it does not infer occupations or Fermi crossings from the energy zero.
    No boolean/string coercion. Segment labels are nonnegative int32 values,
    each appearing in one run; k_distance increases strictly within each run.

    Output status: refused (invalid input), unknown (insufficient occupancy or
    k identity), ok (resolved sampled-path topology). A known gap may accompany
    unknown topology. Extrema energies are relative to the supplied Fermi level;
    locations include every degeneracy within 1e-8 eV. Coordinate identity uses
    1e-8 inv-A distance, without reciprocal-lattice modulo equivalence.
    Metal requires opposite strict Fermi signs in one band/segment (zeros may
    lie between); EF touching an extremum alone is not a metal certificate.

    Masses are positive electron/hole directional m0 values, or None with a
    diagnostics.effective_mass reason. Only an isolated, unique internal edge
    on a straight five-point physical-k window is fitted; full/central local
    quadratics must agree. A relative scalar axis never supplies mass units;
    explicit inv-A Cartesian coordinates can. All confidence/OOD/interval
    fields stay None. No model, network, training, or asset reads occur here.
    """
    result = {
        "status": "refused",
        "analysis_kind": "analytic_measurement",
        "line_mode_gap_eV": None,
        "line_mode_topology": "unknown",
        "extrema": {"vbm": None, "cbm": None},
        "diagnostics": {
            "fermi_crossings": [],
            "fermi_crossing_count": 0,
            "fermi_crossings_truncated": False,
            "effective_mass": {
                carrier: {"reason": "gap_or_occupancy_not_resolved"}
                for carrier in ("electron", "hole")
            },
        },
        "effective_mass_electron_m0": None,
        "effective_mass_hole_m0": None,
        "confidence": None,
        "ood_flag": None,
        "uncertainty_interval": None,
        "reasons": ["invalid_input"],
        "limitations": [
            "sampled_line_mode_only",
            "not_a_prediction_or_calibration",
            "spin_soc_u_not_assessed",
            "directional_mass_not_tensor",
            "input_branch_identity_not_wavefunction_verified",
        ],
        "units": {
            "energy": "eV",
            "k_distance": None,
            "effective_mass": "m0",
            "k_cartesian": "angstrom^-1",
        },
    }
    if not isinstance(data, dict) or not data:
        return result
    reference_key = "reference_eV" if reference_from_roles else "fermi_eV"
    required = {"energies_eV", "k_distance", "segment_ids", reference_key, "k_unit"}
    if reference_from_roles:
        required.add("band_roles")
    if not required.issubset(data):
        result["reasons"] = ["missing_required_fields"]
        return result
    if not isinstance(data["k_unit"], str) or data["k_unit"] not in ("relative", "angstrom^-1"):
        result["reasons"] = ["invalid_k_unit"]
        return result
    try:
        raw = data["energies_eV"]
        if (
            not 1 <= len(raw) <= 4096
            or any(not 2 <= len(row) <= 8192 for row in raw)
            or sum(len(row) for row in raw) > 262144
        ):
            raise ValueError
    except (TypeError, ValueError):
        result["reasons"] = ["invalid_shape_or_size"]
        return result
    try:
        energies = _numerical_array(data["energies_eV"])
        distance = _numerical_array(data["k_distance"])
        fermi = _numerical_array(data[reference_key])
    except (ValueError, TypeError, OverflowError):
        result["reasons"] = ["invalid_numeric"]
        return result
    if energies.ndim != 2 or distance.shape != (energies.shape[1],) or fermi.shape != ():
        result["reasons"] = ["invalid_shape_or_size"]
        return result
    try:
        raw_ids = data["segment_ids"]
        if (
            not isinstance(raw_ids, (list, tuple, np.ndarray))
            or np.ndim(raw_ids) != 1
            or len(raw_ids) != len(distance)
        ):
            raise ValueError
        raw_segments = np.asarray(raw_ids, dtype=object)
        if raw_segments.shape != distance.shape or any(
            isinstance(x, (bool, np.bool_))
            or not isinstance(x, Integral)
            or not 0 <= x <= 2**31 - 1
            for x in raw_segments.flat
        ):
            raise ValueError
        segments = raw_segments.astype(np.int64)
        same_segment = segments[1:] == segments[:-1]
        runs = segments[np.r_[True, ~same_segment]]
        if len(set(runs.tolist())) != len(runs) or np.any(np.diff(distance)[same_segment] <= 0):
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        result["reasons"] = ["invalid_segments_or_distance"]
        return result
    cartesian = None
    if "k_cartesian_invA" in data:
        try:
            cartesian = _numerical_array(data["k_cartesian_invA"])
            if cartesian.shape != (energies.shape[1], 3):
                raise ValueError
        except (ValueError, TypeError, OverflowError):
            result["reasons"] = ["invalid_k_cartesian"]
            return result
    roles = None
    if "band_roles" in data:
        roles = data["band_roles"]
        if (
            not isinstance(roles, (list, tuple, np.ndarray))
            or (isinstance(roles, np.ndarray) and roles.ndim != 1)
            or len(roles) != len(energies)
            or any(
                not isinstance(role, str) or role not in ("valence", "conduction") for role in roles
            )
        ):
            result["reasons"] = ["invalid_band_roles"]
            return result
    energy_resolution = np.spacing(np.abs(energies))
    result["band_count"] = int(energies.shape[0])
    result["k_point_count"] = int(energies.shape[1])
    energies = energies - fermi
    energy_resolution += np.spacing(np.abs(energies))
    result["units"]["k_distance"] = data["k_unit"]
    for sid in [] if reference_from_roles else runs:
        indices = np.flatnonzero(segments == sid)
        for band_index, band in enumerate(energies):
            nonzero = indices[band[indices] != 0]
            flips = np.signbit(band[nonzero[:-1]]) != np.signbit(band[nonzero[1:]])
            result["diagnostics"]["fermi_crossing_count"] += int(np.count_nonzero(flips))
            remaining = 128 - len(result["diagnostics"]["fermi_crossings"])
            for left, right in zip(nonzero[:-1][flips][:remaining], nonzero[1:][flips][:remaining]):
                result["diagnostics"]["fermi_crossings"].append(
                    {
                        "band_index": band_index,
                        "left_k_index": int(left),
                        "right_k_index": int(right),
                        "segment_id": int(sid),
                    }
                )
    result["diagnostics"]["fermi_crossings_truncated"] = (
        result["diagnostics"]["fermi_crossing_count"] > 128
    )
    if result["diagnostics"]["fermi_crossing_count"]:
        result["diagnostics"]["effective_mass"] = {
            carrier: {"reason": "metallic_path"} for carrier in ("electron", "hole")
        }
        result.update(
            status="ok",
            line_mode_gap_eV=0.0,
            line_mode_topology="metal",
            reasons=["strict_fermi_crossing_on_segment"],
        )
        return result
    valence = np.flatnonzero((energies.max(axis=1) <= 0) & (energies.min(axis=1) < 0))
    conduction = np.flatnonzero(energies.min(axis=1) > 0)
    result["diagnostics"]["occupancy_method"] = "conservative_fermi_inference"
    if roles is not None:
        valence = np.asarray([i for i, role in enumerate(roles) if role == "valence"], dtype=int)
        conduction = np.asarray(
            [i for i, role in enumerate(roles) if role == "conduction"], dtype=int
        )
        result["diagnostics"]["occupancy_method"] = "explicit_band_roles"
    result["status"] = "unknown"
    result["reasons"] = []
    if len(valence) + len(conduction) != len(energies):
        result["reasons"].append("ambiguous_fermi_occupancy")
    if not len(valence):
        result["reasons"].append("missing_valence_bands")
    if not len(conduction):
        result["reasons"].append("missing_conduction_bands")
    for name, bands, minimize in (("vbm", valence, False), ("cbm", conduction, True)):
        if len(bands):
            selected = energies[bands]
            edge = float(selected.min() if minimize else selected.max())
            positions = np.argwhere(np.isclose(selected, edge, rtol=0.0, atol=1e-8))
            result["extrema"][name] = {
                (
                    "energy_eV_relative_to_reference"
                    if reference_from_roles
                    else "energy_eV_relative_to_fermi"
                ): edge,
                "locations": [
                    {"band_index": int(bands[b]), "k_index": int(k), "segment_id": int(segments[k])}
                    for b, k in positions
                ],
            }
    vbm, cbm = result["extrema"]["vbm"], result["extrema"]["cbm"]
    if result["reasons"]:
        return result
    if vbm is not None and cbm is not None:
        edge_key = (
            "energy_eV_relative_to_reference"
            if reference_from_roles
            else "energy_eV_relative_to_fermi"
        )
        separation = cbm[edge_key] - vbm[edge_key]
        result["diagnostics"]["band_edge_separation_eV"] = separation
        if separation < 0:
            result["reasons"] = ["overlapping_roles_without_fermi_crossing"]
            return result
        result["line_mode_gap_eV"] = separation
        v_indices = {loc["k_index"] for loc in vbm["locations"]}
        c_indices = {loc["k_index"] for loc in cbm["locations"]}
        if v_indices & c_indices:
            result["line_mode_topology"] = "direct"
            result["status"] = "ok"
            result["reasons"] = ["direct_on_sampled_path"]
        elif cartesian is not None:
            from scipy.spatial import cKDTree

            separations, _ = cKDTree(cartesian[sorted(c_indices)]).query(
                cartesian[sorted(v_indices)]
            )
            direct = bool(np.any(separations <= 1e-8))
            result["status"] = "ok"
            result["line_mode_topology"] = "direct" if direct else "indirect"
            result["reasons"] = [
                "direct_at_equivalent_cartesian_sample"
                if direct
                else "indirect_on_provided_cartesian_path"
            ]
            result["limitations"].append("reciprocal_lattice_equivalence_not_assessed")
        else:
            result["reasons"] = ["physical_k_identity_unavailable"]

    result["diagnostics"]["effective_mass"] = {}
    for carrier, edge in (("electron", cbm), ("hole", vbm)):
        mass, detail = None, {"reason": "physical_k_coordinates_required"}
        if cartesian is not None:
            if len(edge["locations"]) != 1:
                detail = {"reason": "degenerate_extremum"}
            else:
                mass, detail = _directional_mass(
                    energies,
                    cartesian,
                    distance,
                    data["k_unit"],
                    segments,
                    edge["locations"][0],
                    carrier,
                    energy_resolution,
                )
        result[f"effective_mass_{carrier}_m0"] = mass
        result["diagnostics"]["effective_mass"][carrier] = detail
    return result
