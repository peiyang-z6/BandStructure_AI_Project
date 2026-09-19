"""Versioned JSON Schemas exposed in tools/list and as a protocol resource."""

from copy import deepcopy

NUMBER = {"type": "number"}
NUMBERS = {"type": "array", "items": NUMBER, "maxItems": 8192}
SOURCE = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "document_id": {"type": "string", "minLength": 1, "maxLength": 512},
        "page_number": {"type": "integer", "minimum": 1, "maximum": 100000},
        "panel_label": {"type": "string", "minLength": 1, "maxLength": 128},
        "figure_label": {"type": "string", "maxLength": 256},
        "material_label": {"type": "string", "maxLength": 256},
        "source_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "attachment_id": {"type": "string", "pattern": "^att_[0-9a-f]{32}$"},
    },
}
CALIBRATION = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "energy_tick_count": {"type": "integer", "minimum": 0, "maximum": 10000},
        "k_anchor_count": {"type": "integer", "minimum": 0, "maximum": 10000},
        "fermi_reference_observed": {"type": "boolean"},
    },
}
ROLE = {"enum": ["valence", "conduction", "unknown"]}
V2_PROPERTIES = {
    "schema_version": {"const": 2},
    "source": SOURCE,
    "energy_unit": {"const": "eV"},
    "energy_reference": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "kind": {"enum": ["fermi", "vbm", "arbitrary"]},
            "value_eV": NUMBER,
            "observed": {"type": "boolean"},
            "evidence": {"type": "string", "maxLength": 1000},
        },
        "required": ["kind", "value_eV", "observed", "evidence"],
    },
    "fermi_eV": {"type": ["number", "null"]},
    "k_unit": {"enum": ["relative", "angstrom^-1"]},
    "k_distance": NUMBERS,
    "segment_ids": {
        "type": "array",
        "maxItems": 8192,
        "items": {"type": "integer", "minimum": 0, "maximum": 2147483647},
    },
    "bands": {
        "type": "array",
        "maxItems": 4096,
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "band_id": {"type": "string", "minLength": 1, "maxLength": 128},
                "role": ROLE,
                "energies_eV": {
                    "type": "array",
                    "maxItems": 8192,
                    "items": {"type": ["number", "null"]},
                },
                "pixel_points": {
                    "type": "array",
                    "maxItems": 8192,
                    "items": {
                        "anyOf": [
                            {"type": "null"},
                            {"type": "array", "minItems": 2, "maxItems": 2, "items": NUMBER},
                        ]
                    },
                },
            },
            "required": ["band_id", "role", "energies_eV"],
        },
    },
    "calibration_evidence": CALIBRATION,
    "calibration_id": {"type": "string", "pattern": "^res_[0-9a-f]{32}$"},
    "ambiguities": {
        "type": "array",
        "maxItems": 64,
        "items": {"type": "string", "maxLength": 1000},
    },
    "human_confirmed": {"type": "boolean"},
    "perception_confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
    "ocr_text": {"type": "string", "maxLength": 200000},
    "k_cartesian_invA": {
        "type": "array",
        "maxItems": 8192,
        "items": {"type": "array", "minItems": 3, "maxItems": 3, "items": NUMBER},
    },
    "path_segments": {
        "type": "array",
        "maxItems": 8192,
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "segment_id": {"type": "integer", "minimum": 0},
                "start_label": {"type": "string", "maxLength": 128},
                "end_label": {"type": "string", "maxLength": 128},
            },
            "required": ["segment_id", "start_label", "end_label"],
        },
    },
    "physical_metadata": {
        "type": "object",
        "maxProperties": 16,
        "additionalProperties": False,
        "properties": {
            **{
                key: {"type": ["string", "null"], "maxLength": 2000}
                for key in [
                    "functional",
                    "hubbard_u",
                    "pseudopotential",
                    "spin_mode",
                    "cell_setting",
                    "k_path_convention",
                    "source_sha256",
                    "structure_sha256",
                    "energy_reference_description",
                ]
            },
            "soc": {"type": ["boolean", "null"]},
            "temperature_K": {"type": ["number", "null"], "minimum": 0},
            "electron_count": {"type": ["number", "null"], "minimum": 0},
            "band_channels": {
                "type": "array",
                "maxItems": 4096,
                "items": {"enum": ["up", "down", "spinor", "unknown"]},
            },
        },
    },
    "ocr_corrections": {
        "type": "array",
        "maxItems": 128,
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                **{
                    key: {"type": "string", "minLength": 1, "maxLength": 1000}
                    for key in ["raw_text", "corrected_text", "evidence"]
                },
                "bbox": {"type": "array", "minItems": 4, "maxItems": 4, "items": NUMBER},
                "reviewer_kind": {"const": "ai_client"},
                "source_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "coordinate_unit": {"enum": ["pixel", "pdf_point"]},
            },
            "required": [
                "raw_text",
                "corrected_text",
                "bbox",
                "evidence",
                "reviewer_kind",
                "source_sha256",
                "coordinate_unit",
            ],
        },
    },
}
V2_PARTIAL = {
    "type": "object",
    "properties": V2_PROPERTIES,
    "required": ["schema_version"],
    "additionalProperties": False,
    "description": "Version 2 evidence bundle. Missing fields are permitted for actionable evidence requests; invalid types are rejected.",
}
V2_COMPLETE = {
    **V2_PARTIAL,
    "required": [
        "schema_version",
        "source",
        "energy_unit",
        "energy_reference",
        "k_unit",
        "k_distance",
        "segment_ids",
        "bands",
        "calibration_evidence",
        "ambiguities",
    ],
}


def observation_input_schema():
    legacy = deepcopy(V2_PROPERTIES)
    for key in [
        "schema_version",
        "energy_reference",
        "path_segments",
        "physical_metadata",
        "ocr_corrections",
        "calibration_id",
    ]:
        legacy.pop(key, None)
    legacy["bands"]["items"]["properties"].pop("band_id")
    legacy["bands"]["items"]["properties"].pop("pixel_points")
    legacy["source"]["properties"].pop("attachment_id")
    legacy["source"]["properties"].pop("source_sha256")
    legacy["bands"]["items"]["required"] = ["role", "energies_eV"]
    legacy["bands"]["maxItems"] = 4096
    return {
        "anyOf": [
            V2_PARTIAL,
            {"type": "object", "properties": legacy, "additionalProperties": False},
        ],
        "description": "Prefer schema_version=2. Unversioned legacy inputs require an observed Fermi reference.",
    }


OBSERVATION_OUTPUT = {
    "type": "object",
    "required": [
        "status",
        "confidence",
        "ood_flag",
        "human_audited",
        "eligible_for_scientific_acceptance",
    ],
    "properties": {
        "status": {
            "enum": [
                "refused",
                "needs_more_evidence",
                "partial_observations",
                "ai_observation_unverified",
            ]
        },
        "confidence": {"type": "null"},
        "ood_flag": {"type": "null"},
        "human_audited": {"const": False},
        "eligible_for_scientific_acceptance": {"const": False},
        "line_mode_gap_eV": {"type": ["number", "null"]},
        "line_mode_topology": {"enum": ["metal", "direct", "indirect", "unknown"]},
        "missing_evidence": {"type": "array", "items": {"type": "string"}},
        "invalid_fields": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": True,
}

MATRIX = {
    "type": "array",
    "minItems": 1,
    "maxItems": 4096,
    "items": {"type": "array", "minItems": 1, "maxItems": 65536, "items": NUMBER},
}
XYZ = {"type": "array", "items": {"type": "array", "minItems": 3, "maxItems": 3, "items": NUMBER}}
SCIENTIFIC_INPUT = {
    "oneOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "data_kind",
                "energies_eV",
                "k_distance",
                "segment_ids",
                "fermi_eV",
                "k_unit",
            ],
            "properties": {
                "data_kind": {"const": "line_mode"},
                "energies_eV": MATRIX,
                "k_distance": NUMBERS,
                "segment_ids": V2_PROPERTIES["segment_ids"],
                "fermi_eV": NUMBER,
                "k_unit": V2_PROPERTIES["k_unit"],
                "k_cartesian_invA": V2_PROPERTIES["k_cartesian_invA"],
                "band_roles": {"type": "array", "items": ROLE},
                "physical_metadata": V2_PROPERTIES["physical_metadata"],
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "data_kind",
                "energies_eV",
                "occupancies",
                "occupation_full",
                "k_fractional",
                "k_weights",
                "lattice_A",
            ],
            "properties": {
                "data_kind": {"const": "uniform_mesh"},
                "energies_eV": MATRIX,
                "occupancies": MATRIX,
                "occupation_full": {"enum": [1, 2]},
                "k_fractional": {**XYZ, "minItems": 1, "maxItems": 65536},
                "k_weights": {
                    "type": "array",
                    "items": {"type": "number", "exclusiveMinimum": 0},
                    "minItems": 1,
                    "maxItems": 65536,
                },
                "lattice_A": {**XYZ, "minItems": 3, "maxItems": 3},
                "band_ids": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4096,
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1, "maxLength": 128},
                },
                "physical_metadata": V2_PROPERTIES["physical_metadata"],
                "source": {"type": "object", "maxProperties": 16},
            },
        },
    ],
    "description": "Total energy-cell budget 262144; no slicing or implicit mesh-to-path conversion.",
}
