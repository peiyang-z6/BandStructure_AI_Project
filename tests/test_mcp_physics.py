"""MCP analytic measurement contract: synthetic, bounded physics oracles only."""
import copy
import json

import numpy as np
import pytest

from src.utils import physics_validator


def analyze(data):
    fn = getattr(physics_validator, "analyze_numerical_band_data", None)
    assert callable(fn), "Missing stable analyze_numerical_band_data interface"
    result = fn(data)
    json.dumps(result, allow_nan=False)
    return result


def payload(energies=None, **updates):
    data = {
        "energies_eV": energies if energies is not None else [[-2., 0., -2.], [3., 1., 3.]],
        "k_distance": [0., .5, 1.],
        "segment_ids": [0, 0, 0],
        "fermi_eV": 0.,
        "k_unit": "relative",
    }
    data.update(updates)
    return data


def test_interface_has_uncalibrated_json_refusal_envelope():
    result = analyze({})
    assert result["status"] == "refused"
    assert result["analysis_kind"] == "analytic_measurement"
    assert result["line_mode_topology"] == "unknown"
    for key in ("line_mode_gap_eV", "effective_mass_electron_m0", "effective_mass_hole_m0",
                "confidence", "ood_flag", "uncertainty_interval"):
        assert result[key] is None
    assert result["extrema"] == {"vbm": None, "cbm": None}
    assert result["reasons"] and result["limitations"]
    assert result["units"]["energy"] == "eV"
    assert not any(key in result for key in ("spin", "soc", "hubbard_u"))


def test_gap_is_fermi_relative_direct_measurement_not_prediction():
    for shift in (0., 1234.5, -77.):
        for order in ([0, 1], [1, 0]):
            data = payload((np.array(payload()["energies_eV"])[order] + shift).tolist(),
                           fermi_eV=shift)
            result = analyze(data)
            assert result["status"] == "ok"
            np.testing.assert_allclose(result["line_mode_gap_eV"], 1., rtol=1e-5, atol=1e-5)
            assert result["line_mode_topology"] == "direct"
            assert "direct_on_sampled_path" in result["reasons"]
            np.testing.assert_allclose(result["extrema"]["vbm"]["energy_eV_relative_to_fermi"],
                                       0., rtol=1e-5, atol=1e-5)
            assert result["extrema"]["cbm"]["locations"][0]["k_index"] == 1
            assert result["confidence"] is result["ood_flag"] is None
            assert result["effective_mass_electron_m0"] is None
            assert result["units"]["k_distance"] == "relative"


@pytest.mark.parametrize("field,value", [
    ("energies_eV", [[-2., False, -2.], [3., 1., 3.]]),
    ("energies_eV", [[-2., "0", -2.], [3., 1., 3.]]),
    ("energies_eV", [[-2., float("nan"), -2.], [3., 1., 3.]]),
    ("energies_eV", [[-2., 0., -2.], [3., float("inf"), 3.]]),
    ("k_distance", [0., True, 1.]), ("k_distance", [0., .5, float("inf")]),
    ("fermi_eV", True), ("fermi_eV", "0"), ("fermi_eV", None),
    ("fermi_eV", float("nan")), ("fermi_eV", 1e100),
    ("energies_eV", [[-2., 0., -2.], [3., 1e100, 3.]]),
    ("k_distance", [0., .5, 1e100]),
])
def test_rejects_nonfinite_coerced_or_unbounded_numerics(field, value):
    result = analyze(payload(**{field: value}))
    assert result["status"] == "refused"
    assert result["line_mode_gap_eV"] is None
    assert "invalid_numeric" in result["reasons"]


@pytest.mark.parametrize("updates", [
    {"energies_eV": []}, {"energies_eV": [1., 2., 3.]},
    {"energies_eV": [[1., 2.], [2., 3., 4.]]},
    {"energies_eV": np.zeros((1, 2, 3))}, {"energies_eV": [[-1.], [1.]]},
    {"energies_eV": [[-1., 0., -1.]] * 4097},
    {"energies_eV": [[-1.] * 8193, [1.] * 8193]},
    {"k_distance": [0., 1.]}, {"k_distance": [[0., .5, 1.]]},
    {"fermi_eV": [0.]}, {"k_unit": "nm^-1"}, {"k_unit": False},
])
def test_refuses_bad_shapes_sizes_and_units(updates):
    result = analyze(payload(**updates))
    assert result["status"] == "refused"
    assert result["line_mode_gap_eV"] is None
    assert set(result["reasons"]) & {"invalid_shape_or_size", "invalid_numeric", "invalid_k_unit"}


@pytest.mark.parametrize("missing", ["energies_eV", "k_distance", "segment_ids", "fermi_eV", "k_unit"])
def test_refuses_missing_required_fields(missing):
    data = payload()
    del data[missing]
    result = analyze(data)
    assert result["status"] == "refused"
    assert "missing_required_fields" in result["reasons"]


@pytest.mark.parametrize("updates", [
    {"segment_ids": [0, False, 0]}, {"segment_ids": [0., 0., 0.]},
    {"segment_ids": [0, 1, 0]}, {"segment_ids": [0, 0]},
    {"segment_ids": [-1, -1, -1]}, {"segment_ids": [0, 0, 2**64]},
    {"segment_ids": [[0, 0, 0]]}, {"segment_ids": ["0", "0", "0"]},
    {"k_distance": [0., 0., 1.]}, {"k_distance": [0., .5, .4]},
])
def test_refuses_noncontiguous_segments_or_nonmonotone_local_axis(updates):
    result = analyze(payload(**updates))
    assert result["status"] == "refused"
    assert "invalid_segments_or_distance" in result["reasons"]


def test_strict_fermi_crossing_is_same_band_same_segment_only():
    crossed = analyze(payload([[-1., 1., 2.]]))
    assert crossed["line_mode_topology"] == "metal"
    assert crossed["status"] == "ok"
    assert crossed["line_mode_gap_eV"] == 0.
    assert "strict_fermi_crossing_on_segment" in crossed["reasons"]
    through_zero = analyze(payload([[-1., 0., 1.]]))
    assert through_zero["line_mode_topology"] == "metal"
    assert crossed["diagnostics"]["fermi_crossings"] == [
        {"band_index": 0, "left_k_index": 0, "right_k_index": 1, "segment_id": 0}]
    for data in (
        payload([[-1., -2., 1., 2.]], segment_ids=[0, 0, 1, 1], k_distance=[0., 1., 0., 1.]),
        payload([[-2., 0., -2.], [3., 1., 3.]]),
    ):
        result = analyze(data)
        assert result["line_mode_topology"] != "metal"
        assert result["diagnostics"]["fermi_crossings"] == []


@pytest.mark.parametrize("energies,reason", [
    ([[-2., 0., -2.]], "missing_conduction_bands"),
    ([[3., 1., 3.]], "missing_valence_bands"),
    ([[0., 0., 0.], [3., 1., 3.]], "ambiguous_fermi_occupancy"),
])
def test_incomplete_occupancy_cannot_fabricate_zero_gap(energies, reason):
    result = analyze(payload(energies))
    assert result["status"] == "unknown"
    assert result["line_mode_topology"] == "unknown"
    assert result["line_mode_gap_eV"] is None
    assert reason in result["reasons"]


def test_explicit_band_roles_resolve_fermi_ambiguity_without_guessing():
    data = payload([[8., 10., 8.], [13., 11., 13.]], band_roles=["valence", "conduction"])
    result = analyze(data)
    assert result["status"] == "ok"
    assert result["line_mode_topology"] == "direct"
    np.testing.assert_allclose(result["line_mode_gap_eV"], 1., rtol=1e-5, atol=1e-5)
    assert result["diagnostics"]["occupancy_method"] == "explicit_band_roles"
    swapped = copy.deepcopy(data)
    swapped["energies_eV"].reverse()
    swapped["band_roles"].reverse()
    assert analyze(swapped)["line_mode_gap_eV"] == result["line_mode_gap_eV"]
    for bad in (["valence"], [True, "conduction"], ["vbm", "cbm"], "valence", None):
        refused = analyze(payload(band_roles=bad))
        assert refused["status"] == "refused"
        assert "invalid_band_roles" in refused["reasons"]


def test_directness_checks_every_degenerate_extremum_not_first_argmax():
    data = payload([[-2., 0., -2., 0., -2.], [4., 3., 2., 1., 2.]],
                   k_distance=[0., .2, .4, .6, .8], segment_ids=[0] * 5)
    result = analyze(data)
    assert result["line_mode_topology"] == "direct"
    assert {loc["k_index"] for loc in result["extrema"]["vbm"]["locations"]} == {1, 3}
    assert "direct_on_sampled_path" in result["reasons"]
    separate = analyze(payload([[-2., 0., -2.], [1., 2., 3.]]))
    assert separate["line_mode_topology"] == "unknown"
    assert separate["line_mode_gap_eV"] == 1.
    assert "physical_k_identity_unavailable" in separate["reasons"]


def test_cartesian_coordinates_identify_repeated_physical_k_across_segments():
    data = payload([[-2., 0., -2., -2.], [2., 2., 1., 3.]],
                   segment_ids=[0, 0, 1, 1], k_distance=[0., 1., 0., 1.],
                   k_cartesian_invA=[[0., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 1., 0.]])
    result = analyze(data)
    assert result["status"] == "ok"
    assert result["line_mode_topology"] == "direct"
    assert "direct_at_equivalent_cartesian_sample" in result["reasons"]
    data["k_cartesian_invA"][2] = [0., 1., 0.]
    result = analyze(data)
    assert result["line_mode_topology"] == "indirect"
    assert "indirect_on_provided_cartesian_path" in result["reasons"]
    for bad in ([[0., 0., 0.]], [[0., 0., False]] * 4, [[float("nan"), 0., 0.]] * 4, None):
        refused = analyze({**data, "k_cartesian_invA": bad})
        assert refused["status"] == "refused"
        assert "invalid_k_cartesian" in refused["reasons"]


def parabolic_payload(electron=.2, hole=.7, scale=1.):
    from scipy.constants import hbar, electron_mass, electron_volt

    x = np.array([-.20, -.11, -.035, 0., .06, .13, .24]) * scale
    # Independent SI free-particle dispersion: E = hbar^2 k_SI^2 / (2 m).
    energy_factor = hbar**2 / (2. * electron_mass * electron_volt)
    valence = -energy_factor * (x / scale * 1e10)**2 / hole
    conduction = 1. + energy_factor * (x / scale * 1e10)**2 / electron
    direction = np.array([1., 2., -2.]) / 3.
    return payload([valence.tolist(), conduction.tolist()], k_distance=(x - x[0]).tolist(),
                   segment_ids=[0] * len(x), k_unit="angstrom^-1",
                   k_cartesian_invA=(x[:, None] * direction).tolist())


@pytest.mark.parametrize("scale", [.5, 1., 2.])
def test_nonuniform_si_parabola_yields_directional_not_normalized_mass(scale):
    data = parabolic_payload(scale=scale)
    for shift in (0., 321.):
        shifted = {**data, "energies_eV": (np.array(data["energies_eV"]) + shift).tolist(),
                   "fermi_eV": shift}
        result = analyze(shifted)
        assert result["effective_mass_electron_m0"] is not None
        np.testing.assert_allclose(result["effective_mass_electron_m0"], .2 * scale**2,
                                   rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(result["effective_mass_hole_m0"], .7 * scale**2,
                                   rtol=1e-5, atol=1e-5)
        assert "directional_mass_not_tensor" in result["limitations"]
        details = result["diagnostics"]["effective_mass"]["electron"]
        assert details["fit_k_indices"] == [1, 2, 3, 4, 5]
        np.testing.assert_allclose(details["direction_cartesian"], [1/3, 2/3, -2/3],
                                   rtol=1e-5, atol=1e-5)
        assert result["confidence"] is None


@pytest.mark.parametrize("case,reason", [
    ("bend", "nonstraight_or_retraced_k_window"),
    ("retrace", "nonstraight_or_retraced_k_window"),
    ("duplicate", "nonstraight_or_retraced_k_window"),
    ("segment", "endpoint_or_insufficient_points"),
    ("endpoint", "endpoint_or_insufficient_points"),
    ("no_coords", "physical_k_coordinates_required"),
])
def test_directional_mass_refuses_unqualified_geometric_windows(case, reason):
    data = parabolic_payload()
    points = np.array(data["k_cartesian_invA"])
    if case == "bend":
        points[4:, 0] += .03
    elif case == "retrace":
        points[4] = points[2]
    elif case == "duplicate":
        points[4] = points[3]
    elif case == "segment":
        data["segment_ids"] = [0, 0, 0, 0, 1, 1, 1]
    elif case == "endpoint":
        data["energies_eV"] = [[-float(i*i) for i in range(7)], [1.+i*i for i in range(7)]]
    data["k_cartesian_invA"] = points.tolist()
    if case == "no_coords":
        del data["k_cartesian_invA"]
    result = analyze(data)
    for carrier in ("electron", "hole"):
        assert result[f"effective_mass_{carrier}_m0"] is None
        assert result["diagnostics"]["effective_mass"][carrier]["reason"] == reason


@pytest.mark.parametrize("case,reason", [
    ("crossing_branch", "nonisolated_or_crossing_branch"),
    ("degenerate", "degenerate_extremum"),
    ("nonsmooth", "nonquadratic_or_unstable_curvature"),
    ("flat", "degenerate_extremum"),
])
def test_mass_requires_smooth_isolated_nondegenerate_branch(case, reason):
    data = parabolic_payload()
    if case == "crossing_branch":
        other = np.array(data["energies_eV"][1]) + [1., -.04, -.01, .06, .15, .15, .15]
        data["energies_eV"].append(other.tolist())
    elif case == "degenerate":
        data["energies_eV"].append(list(data["energies_eV"][1]))
    elif case == "nonsmooth":
        data["energies_eV"][1][4] += .02
    elif case == "flat":
        data["energies_eV"][1] = [1.] * 7
    result = analyze(data)
    assert result["effective_mass_electron_m0"] is None
    assert result["diagnostics"]["effective_mass"]["electron"]["reason"] == reason


def test_overlapping_roles_do_not_invent_metal_without_fermi_crossing():
    result = analyze(payload([[3., 5., 3.], [4., 4., 6.]],
                             band_roles=["valence", "conduction"]))
    assert result["line_mode_gap_eV"] is None
    assert result["line_mode_topology"] == "unknown"
    assert result["status"] == "unknown"
    assert result["diagnostics"]["band_edge_separation_eV"] == -1.
    assert "overlapping_roles_without_fermi_crossing" in result["reasons"]
    touching = analyze(payload([[-1., 0., -1.], [1., 0., 1.]],
                               band_roles=["valence", "conduction"]))
    assert touching["line_mode_gap_eV"] == 0.
    assert touching["line_mode_topology"] == "direct"


@pytest.mark.parametrize("data", [
    {}, payload([[-1., 1., 2.]]), payload([[-2., 0., -2.]]),
    payload([[3., 5., 3.], [4., 4., 6.]], band_roles=["valence", "conduction"]),
])
def test_every_null_mass_has_structured_nonprobabilistic_reason(data):
    result = analyze(data)
    for carrier in ("electron", "hole"):
        assert result[f"effective_mass_{carrier}_m0"] is None
        assert result["diagnostics"]["effective_mass"][carrier]["reason"]
    assert result["confidence"] is result["uncertainty_interval"] is result["ood_flag"] is None
    assert "input_branch_identity_not_wavefunction_verified" in result["limitations"]


def test_total_numeric_budget_and_malformed_optional_arrays_refuse_safely():
    oversized = payload([[-1.] * 1025] * 256, k_distance=list(range(1025)), segment_ids=[0] * 1025)
    assert analyze(oversized)["status"] == "refused"
    for optional in (np.array("valence"), np.array(0.), np.zeros((2, 1)), [[], []]):
        result = analyze(payload(band_roles=optional))
        assert result["status"] == "refused"
        assert "invalid_band_roles" in result["reasons"]
    for field in ("k_distance", "k_cartesian_invA", "segment_ids"):
        data = payload(**{field: np.zeros(300000)})
        assert analyze(data)["status"] == "refused"


def test_crossing_diagnostics_are_bounded_but_total_is_exact():
    k = 1024
    data = payload([[-1., 1.] * (k // 2)], k_distance=list(range(k)), segment_ids=[0] * k)
    result = analyze(data)
    assert result["line_mode_topology"] == "metal"
    assert result["diagnostics"]["fermi_crossing_count"] == k - 1
    assert len(result["diagnostics"]["fermi_crossings"]) == 128
    assert result["diagnostics"]["fermi_crossings_truncated"] is True
    assert result["diagnostics"]["effective_mass"]["electron"]["reason"] == "metallic_path"


@pytest.mark.parametrize("scale", [1., 1e-8, 1e-9, 1e-10])
def test_bend_rejection_is_invariant_under_physical_k_scale(scale):
    # NUM-MASS-2: the same dimensionless transverse bend at every length scale.
    data = parabolic_payload(scale=scale)
    data["k_unit"] = "relative"  # Isolate Cartesian straightness from metric consistency.
    points = np.array(data["k_cartesian_invA"])
    points[4] += .05 * scale * np.array([2., -1., 0.]) / np.sqrt(5.)
    data["k_cartesian_invA"] = points.tolist()
    result = analyze(data)
    assert result["status"] == "ok" and result["line_mode_topology"] == "direct"
    np.testing.assert_allclose(result["line_mode_gap_eV"], 1., rtol=1e-5, atol=0.)
    for carrier in ("electron", "hole"):
        assert result[f"effective_mass_{carrier}_m0"] is None
        assert result["diagnostics"]["effective_mass"][carrier]["reason"] == "nonstraight_or_retraced_k_window"
    assert result["confidence"] is result["ood_flag"] is result["uncertainty_interval"] is None


@pytest.mark.parametrize("scale", [1., 1e-8, 1e-9, 1e-10])
def test_resolved_straight_small_k_window_keeps_squared_scale_mass(scale):
    result = analyze(parabolic_payload(scale=scale))
    for carrier, reference_mass in (("electron", .2), ("hole", .7)):
        assert result[f"effective_mass_{carrier}_m0"] is not None
        # Normalize first: an absolute 1e-5 assertion would accept any tiny mass.
        np.testing.assert_allclose(result[f"effective_mass_{carrier}_m0"] / (reference_mass * scale**2),
                                   1., rtol=1e-5, atol=0.)
        assert result["diagnostics"]["effective_mass"][carrier]["reason"] == "local_quadratic_fit"


@pytest.mark.parametrize("scale", [1., 1e-9, 1e-10])
@pytest.mark.parametrize("metric", [.1, 10., 1.00002, "redistributed_steps"])
def test_physical_metric_conflict_rejects_mass_at_every_k_scale(scale, metric):
    # NUM-MASS-3: compare each physical step; matching total lengths alone is insufficient.
    data = parabolic_payload(scale=scale)
    distance = np.array(data["k_distance"])
    if metric == "redistributed_steps":
        steps = np.diff(distance)
        steps[2] += .01 * scale
        steps[3] -= .01 * scale
        distance = np.r_[distance[0], distance[0] + np.cumsum(steps)]
    else:
        distance *= metric
    data["k_distance"] = distance.tolist()
    result = analyze(data)
    assert result["status"] == "ok" and result["line_mode_topology"] == "direct"
    np.testing.assert_allclose(result["line_mode_gap_eV"], 1., rtol=1e-5, atol=0.)
    for carrier in ("electron", "hole"):
        assert result[f"effective_mass_{carrier}_m0"] is None
        assert result["diagnostics"]["effective_mass"][carrier]["reason"] == "inconsistent_physical_k_distance"
    assert result["confidence"] is result["ood_flag"] is result["uncertainty_interval"] is None
    # A declared relative scalar axis is not a competing physical ruler.
    data["k_unit"] = "relative"
    control = analyze(data)
    for carrier, reference_mass in (("electron", .2), ("hole", .7)):
        assert control[f"effective_mass_{carrier}_m0"] is not None
        np.testing.assert_allclose(control[f"effective_mass_{carrier}_m0"] / (reference_mass * scale**2),
                                   1., rtol=1e-5, atol=0.)


@pytest.mark.parametrize("scale", [1., 1e-9, 1e-10])
def test_small_physical_metric_keeps_existing_relative_tolerance(scale):
    data = parabolic_payload(scale=scale)
    data["k_distance"] = (np.array(data["k_distance"]) * (1. + 5e-6)).tolist()
    result = analyze(data)
    for carrier, reference_mass in (("electron", .2), ("hole", .7)):
        assert result[f"effective_mass_{carrier}_m0"] is not None
        np.testing.assert_allclose(result[f"effective_mass_{carrier}_m0"] / (reference_mass * scale**2),
                                   1., rtol=1e-5, atol=0.)


def test_conflicting_physical_distance_cannot_certify_a_mass():
    data = parabolic_payload()
    data["k_distance"] = np.linspace(0., 1., 7).tolist()
    result = analyze(data)
    assert result["effective_mass_electron_m0"] is None
    assert result["diagnostics"]["effective_mass"]["electron"]["reason"] == "inconsistent_physical_k_distance"
    data["k_unit"] = "relative"
    np.testing.assert_allclose(analyze(data)["effective_mass_electron_m0"], .2, rtol=1e-5, atol=1e-5)
    del data["k_cartesian_invA"]
    data["k_unit"] = "angstrom^-1"
    assert analyze(data)["effective_mass_electron_m0"] is None


@pytest.mark.parametrize("carrier", ["electron", "hole"])
@pytest.mark.parametrize("coefficient", [5e-8, 5e-6, .05])
def test_mass_rejects_relative_quartic_instability_at_small_energy(coefficient, carrier):
    # NUM-MASS-1: full/central curvature differs despite tiny eV residuals.
    x = np.array([-1., -.5, 0., .5, 1.])
    signal = coefficient * x**2 + .1 * coefficient * x**4
    bands = [-1. - x**2, 1. + x**2]
    bands[1 if carrier == "electron" else 0] = (
        1. + signal if carrier == "electron" else -1. - signal)
    data = payload(np.array(bands).tolist(), k_distance=(x + 1.).tolist(),
                   segment_ids=[0] * 5, k_unit="angstrom^-1",
                   k_cartesian_invA=np.column_stack((x, x * 0, x * 0)).tolist())
    result = analyze(data)
    assert result["status"] == "ok" and result["line_mode_topology"] == "direct"
    np.testing.assert_allclose(result["line_mode_gap_eV"], 2., rtol=1e-5, atol=0.)
    assert result[f"effective_mass_{carrier}_m0"] is None
    assert result["diagnostics"]["effective_mass"][carrier]["reason"] == "nonquadratic_or_unstable_curvature"
    other = "hole" if carrier == "electron" else "electron"
    assert result[f"effective_mass_{other}_m0"] is not None
    assert result["confidence"] is result["ood_flag"] is result["uncertainty_interval"] is None


@pytest.mark.parametrize("carrier", ["electron", "hole"])
@pytest.mark.parametrize("coefficient", [5e-8, 5e-6, .05])
def test_resolved_small_energy_parabola_keeps_si_mass(coefficient, carrier):
    from scipy.constants import hbar, electron_mass, electron_volt

    x = np.array([-1., -.5, 0., .5, 1.])
    bands = [-1. - x**2, 1. + x**2]
    bands[1 if carrier == "electron" else 0] = (
        1. + coefficient * x**2 if carrier == "electron" else -1. - coefficient * x**2)
    data = payload(np.array(bands).tolist(), k_distance=(x + 1.).tolist(),
                   segment_ids=[0] * 5, k_unit="angstrom^-1",
                   k_cartesian_invA=np.column_stack((x, x * 0, x * 0)).tolist())
    result = analyze(data)
    expected = hbar**2 / (2. * electron_mass * electron_volt * 1e-20 * coefficient)
    assert result[f"effective_mass_{carrier}_m0"] is not None
    np.testing.assert_allclose(result[f"effective_mass_{carrier}_m0"] / expected,
                               1., rtol=1e-5, atol=0.)
    assert result["diagnostics"]["effective_mass"][carrier]["reason"] == "local_quadratic_fit"


@pytest.mark.parametrize("carrier", ["electron", "hole"])
@pytest.mark.parametrize("fermi", [0., float(2**19)])
def test_mass_refuses_unresolved_energy_precision_even_for_exact_fit(fermi, carrier):
    # Binary samples fit exactly, but one input ULP overwhelms 1e-5 mass precision.
    # Fermi subtraction must not erase the raw-input resolution limitation.
    x = np.array([-1., -.5, 0., .5, 1.])
    coefficient = 2.**-24
    bands = [-1. - x**2, 1. + x**2]
    bands[1 if carrier == "electron" else 0] = (
        1. + coefficient * x**2 if carrier == "electron" else -1. - coefficient * x**2)
    data = payload((np.array(bands) + 2**19).tolist(), fermi_eV=fermi,
                   band_roles=["valence", "conduction"], k_distance=(x + 1.).tolist(),
                   segment_ids=[0] * 5, k_unit="angstrom^-1",
                   k_cartesian_invA=np.column_stack((x, x * 0, x * 0)).tolist())
    result = analyze(data)
    assert result["status"] == "ok" and result["line_mode_topology"] == "direct"
    assert result["line_mode_gap_eV"] == 2.
    assert result[f"effective_mass_{carrier}_m0"] is None
    assert result["diagnostics"]["effective_mass"][carrier]["reason"] == "nonquadratic_or_unstable_curvature"
    other = "hole" if carrier == "electron" else "electron"
    assert result[f"effective_mass_{other}_m0"] is not None
    assert result["confidence"] is result["ood_flag"] is result["uncertainty_interval"] is None
















