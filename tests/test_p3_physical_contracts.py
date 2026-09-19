"""P3 capability boundaries: diagnostic execution is not scientific acceptance."""
import json

import numpy as np
import pytest


def test_normalized_scalar_k_is_explicitly_ineligible_for_physical_derivatives():
    from src.evaluation import multiband_metrics as module
    assert callable(getattr(module, "physical_metric_eligibility", None))
    result = module.physical_metric_eligibility(np.linspace(0., 1., 256),
                                                 np.zeros(256, dtype=int), k_unit="1/angstrom")
    assert result["eligible"] is False
    assert "scalar_k_is_not_3d_reciprocal_geometry" in result["reasons"]
    assert result["metrics"] == {"slope": False, "curvature": False,
                                  "directional_effective_mass": False}
    assert result["scope"] == "selected_slot_directional_not_full_mass_tensor"
    json.dumps(result, allow_nan=False)


def test_dense_looking_3d_and_verified_flags_cannot_manufacture_physical_acceptance():
    from src.evaluation.multiband_metrics import physical_metric_eligibility
    import inspect
    assert "source_evidence" in inspect.signature(physical_metric_eligibility).parameters
    result = physical_metric_eligibility([[0., 0., 0.], [.1, 0., 0.]], [0, 0],
        k_unit="1/angstrom", source_evidence={"physical_k_path_verified": True,
                                             "source_verified": True, "accepted": True})
    assert result["geometry_shape_compatible"] is True
    assert result["eligible"] is False
    assert result["source_binding_verified"] is False
    assert "pointwise_energy_k_source_binding_not_verified" in result["reasons"]
    assert "physical_derivatives_not_implemented" in result["reasons"]


@pytest.mark.parametrize("k, segments, unit, reason", [
    ([[0., 0., 0.], [1., 0., 0.]], [0, 0, 0], "1/angstrom", "k_point_count_or_dimension_mismatch"),
    ([[0., 0., 0.], [1., 0., 0.]], [0, 0], "normalized", "reciprocal_length_unit_missing_or_unsupported"),
    ([[0., 0., 0.], [np.nan, 0., 0.]], [0, 0], "1/angstrom", "nonfinite_k_geometry"),
    ([[0., 0., 0.], [1., 0., 0.]], [0., .5], "1/angstrom", "invalid_segment_ids"),
])
def test_physical_eligibility_reports_specific_geometry_failures(k, segments, unit, reason):
    from src.evaluation.multiband_metrics import physical_metric_eligibility
    result = physical_metric_eligibility(k, segments, k_unit=unit)
    assert not result["eligible"]
    assert reason in result["reasons"]


def test_coverage_counts_computable_diagnostics_without_promoting_p3():
    from src.evaluation import multiband_metrics as module
    assert callable(getattr(module, "p3_coverage_report", None))
    records = [
        {"material_id": "synthetic-resolved", "bands": [[-1., -1.], [1., 1.]],
         "mask": [True, True], "segment_ids": [0, 0], "k_points": [[0., 0., 0.], [.1, 0., 0.]],
         "k_unit": "1/angstrom", "source_status": "verified_valid", "accepted": True,
         "source_evidence": {"source_verified": True, "physical_k_path_verified": True}},
        {"material_id": "synthetic-unknown", "bands": [[0., 0.]], "mask": [True],
         "segment_ids": [0, 0], "k_points": [0., 1.]},
    ]
    result = module.p3_coverage_report(records)
    assert result["n_requested"] == result["n_diagnostic_eligible"] == 2
    assert result["n_gap_resolved"] == result["n_gap_unknown"] == 1
    assert result["gap_resolution_coverage"] == .5
    assert result["n_physical_metric_eligible"] == 0
    assert result["declared_source_status_counts"] == {"verified_valid": 1, "unknown": 1}
    assert result["accepted"] is False and result["status"] == "NO-GO"
    assert {"independent_source_verification", "pointwise_3d_energy_k_binding",
            "same_data_split_fair_bandformer_baseline", "multiseed_scientific_comparison"} <= set(result["unverified_gates"])
    json.dumps(result, allow_nan=False)


def test_coverage_retains_invalid_empty_and_unresolved_denominators():
    from src.evaluation.multiband_metrics import p3_coverage_report
    records = [
        {"material_id": "valid-unknown", "bands": [[0., 0.]], "mask": [True], "segment_ids": [0, 0]},
        {"material_id": "invalid", "bands": [[np.nan, 1.]], "mask": [True], "segment_ids": [0, 0]},
        {"material_id": "empty", "bands": [[np.nan, np.nan]], "mask": [False], "segment_ids": [0, 0]},
    ]
    result = p3_coverage_report(records)
    assert result["n_requested"] == 3
    assert result["n_diagnostic_eligible"] == result["n_invalid"] == result["n_empty"] == 1
    assert result["n_gap_unknown"] == 1 and result["n_gap_resolved"] == 0
    assert result["diagnostic_coverage"] == pytest.approx(1 / 3)
    assert len(result["records"]) == 3 and not result["accepted"]
    empty = p3_coverage_report([])
    assert empty["diagnostic_coverage"] is None and empty["gap_resolution_coverage"] is None
    assert empty["status"] == "NO-GO"
    json.dumps([result, empty], allow_nan=False)


@pytest.mark.parametrize("ids", [["duplicate", "duplicate"], [""], [None]])
def test_coverage_refuses_unidentifiable_or_duplicate_population(ids):
    from src.evaluation.multiband_metrics import p3_coverage_report
    records = [{"material_id": mid, "bands": [[0., 0.]], "mask": [True],
                "segment_ids": [0, 0]} for mid in ids]
    with pytest.raises(ValueError, match="material_id"):
        p3_coverage_report(records)
