"""Invoke the trained physics model brain on reconstructed Plot-to-Physics tensors."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf

from scripts.finetune_supervised import SupervisedBandGapModel, load_norm_stats
from src.models import load_ssl_encoder

from .physics_reconstructor import ReconstructedTensor


TYPE_LABELS = ["metal", "direct", "indirect"]


def _uncertainty_level(score: float) -> str:
    score = float(np.clip(score, 0.0, 1.0))
    if score >= 0.6:
        return "green"
    if score >= 0.35:
        return "yellow"
    return "red"


def compute_brain_uncertainty(
    type_probs: Dict[str, float],
    vbm_probability,
    cbm_probability,
    gap_ev: float,
) -> Dict[str, Any]:
    """Aggregate brain-output uncertainty into a 0-1 confidence score.

    Components (measured from real outputs, no MC-Dropout — the supervised
    eval path has no dropout layers so MC variance would be a constant zero):
      - normalized entropy of the type softmax (1 - H/Hmax);
      - extremum peak sharpness of the VBM/CBM probability heads
        (1 - mean/peak, worst of the two);
      - physical plausibility of the predicted gap.
    """
    probs = np.asarray(
        [float(type_probs.get(label, 0.0)) for label in TYPE_LABELS], dtype=np.float64
    )
    probs = probs / max(float(probs.sum()), 1e-12)
    entropy_raw = float(-np.sum(probs * np.log(probs + 1e-12)))
    h_max = math.log(len(probs))
    entropy = entropy_raw / h_max if h_max > 0 else 0.0
    entropy_confidence = 1.0 - entropy

    def sharpness(arr) -> float:
        a = np.asarray(arr, dtype=np.float64).reshape(-1)
        if a.size == 0:
            return 0.0
        peak = float(a.max())
        mean = float(a.mean())
        if peak <= 0.0:
            return 0.0
        return float(np.clip(1.0 - mean / peak, 0.0, 1.0))

    peak_sharpness = min(sharpness(vbm_probability), sharpness(cbm_probability))
    gap_plausible = bool(-0.05 <= float(gap_ev) <= 15.0)
    gap_term = 1.0 if gap_plausible else 0.3
    score = float(np.clip(0.45 * entropy_confidence + 0.35 * peak_sharpness + 0.2 * gap_term, 0.0, 1.0))
    return {
        "score": score,
        "level": _uncertainty_level(score),
        "entropy": entropy,
        "entropy_confidence": entropy_confidence,
        "peak_sharpness": peak_sharpness,
        "gap_plausible": gap_plausible,
        "gap_ev": float(gap_ev),
    }


@dataclass
class BrainPrediction:
    """Physics inference emitted by the supervised model brain."""

    line_mode_gap_ev: float
    type_probabilities: Dict[str, float]
    predicted_type: str
    vbm_probability: np.ndarray
    cbm_probability: np.ndarray
    vbm_index: int
    cbm_index: int
    vbm_effective_mass_proxy: float
    cbm_effective_mass_proxy: float
    application_recommendations: List[str] = field(default_factory=list)
    application_recommendation_details: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PhysicsBrainInvoker:
    """Load and call the trained SupervisedBandGapModel.

    This class does not define a new predictor. It reconstructs the exact
    fine-tuned model wrapper, loads existing weights, normalizes the 6D tensor
    with the SSL normalization stats, and calls the model brain.
    """

    def __init__(
        self,
        encoder_path: str = "artifacts/models/aflow_noleak_v7_60k_seed42/ssl_mbm_pretrained.keras",
        weights_path: str = "artifacts/models/aflow_noleak_v7_60k_seed42/finetuned.weights.h5",
        norm_path: str = "artifacts/models/aflow_noleak_v7_60k_seed42/ssl_mbm_norm_stats.json",
        config_path: str = "artifacts/models/aflow_noleak_v7_60k_seed42/finetuned_config.json",
    ) -> None:
        self.encoder_path = Path(encoder_path)
        self.weights_path = Path(weights_path)
        self.norm_path = Path(norm_path)
        self.config_path = Path(config_path)
        self.mean, self.std = load_norm_stats(str(self.norm_path))
        self.model = self._load_model()

    def predict(self, reconstructed: ReconstructedTensor) -> BrainPrediction:
        x_raw = reconstructed.flat_tensor.astype(np.float32)
        x_norm = (x_raw - self.mean) / self.std
        outputs = self.model(tf.constant(x_norm, dtype=tf.float32), training=False)
        topo_features, probs = self.model.extremum_probabilities(tf.constant(x_norm, dtype=tf.float32), training=False)

        gap = float(outputs["gap"].numpy().reshape(-1)[0])
        type_probs_arr = outputs["type"].numpy().reshape(-1)
        type_probs = {label: float(type_probs_arr[i]) for i, label in enumerate(TYPE_LABELS)}
        raw_predicted_type = TYPE_LABELS[int(np.argmax(type_probs_arr))]

        vbm_prob = probs.numpy()[0, :, 0].astype(np.float32)
        cbm_prob = probs.numpy()[0, :, 1].astype(np.float32)
        vbm_idx = int(np.argmax(vbm_prob))
        cbm_idx = int(np.argmax(cbm_prob))
        predicted_type, topology_override = self._resolve_gap_type(
            raw_predicted_type=raw_predicted_type,
            type_probs=type_probs,
            vbm_index=vbm_idx,
            cbm_index=cbm_idx,
            reconstructed=reconstructed,
        )
        vbm_mass = effective_mass_proxy(float(reconstructed.vbm_curvature[reconstructed.vbm_index]))
        cbm_mass = effective_mass_proxy(float(reconstructed.cbm_curvature[reconstructed.cbm_index]))

        prediction = BrainPrediction(
            line_mode_gap_ev=gap,
            type_probabilities=type_probs,
            predicted_type=predicted_type,
            vbm_probability=vbm_prob,
            cbm_probability=cbm_prob,
            vbm_index=vbm_idx,
            cbm_index=cbm_idx,
            vbm_effective_mass_proxy=vbm_mass,
            cbm_effective_mass_proxy=cbm_mass,
            metadata={
                "decision_role": "physics_brain",
                "encoder_path": str(self.encoder_path),
                "weights_path": str(self.weights_path),
                "norm_path": str(self.norm_path),
                "vision_metadata": reconstructed.metadata,
                "topology_features": topo_features.numpy().reshape(-1).astype(float).tolist(),
                "input_tensor_gap_ev": reconstructed.tensor_gap,
                "input_is_metallic": reconstructed.is_metallic,
                "raw_predicted_type": raw_predicted_type,
                "topology_override": topology_override,
            },
        )
        prediction.metadata["confidence"] = compute_brain_uncertainty(
            type_probs, vbm_prob, cbm_prob, gap
        )
        recommender = ApplicationRecommender()
        prediction.application_recommendation_details = recommender.recommend_structured(prediction, reconstructed)
        prediction.application_recommendations = [
            f"{item['recommendation_area']}: {item['physical_basis']}"
            for item in prediction.application_recommendation_details
        ]
        return prediction

    def _load_model(self) -> SupervisedBandGapModel:
        if not self.encoder_path.exists():
            raise FileNotFoundError(self.encoder_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(self.weights_path)

        encoder = load_ssl_encoder(str(self.encoder_path), compile=False)
        feature_mean = self.mean.reshape(-1).astype(np.float32)
        feature_std = self.std.reshape(-1).astype(np.float32)
        model = SupervisedBandGapModel(encoder, feature_mean=feature_mean, feature_std=feature_std)
        model(tf.zeros([1, 128, 6], dtype=tf.float32), training=False)
        model.load_weights(str(self.weights_path))
        return model

    def _resolve_gap_type(
        self,
        raw_predicted_type: str,
        type_probs: Dict[str, float],
        vbm_index: int,
        cbm_index: int,
        reconstructed: ReconstructedTensor,
    ) -> tuple[str, Dict[str, Any]]:
        """Fuse classifier probabilities with visual extremum topology.

        The small supervised classifier can be biased toward the majority
        indirect class on textbook figures. In Plot-to-Physics mode the
        extremum probability heads are a stronger physical signal: if the VBM
        and CBM probability peaks coincide within a few sampled k-points, the
        final user-facing type should be direct even when the softmax head says
        indirect.
        """

        brain_distance = abs(int(vbm_index) - int(cbm_index))
        reconstructed_distance = abs(int(reconstructed.vbm_index) - int(reconstructed.cbm_index))
        direct_tolerance = max(3, int(round(0.035 * max(len(reconstructed.k_axis), 1))))
        metal_prob = float(type_probs.get("metal", 0.0))
        direct_by_brain_topology = brain_distance <= direct_tolerance
        direct_by_reconstructed_topology = reconstructed_distance <= direct_tolerance
        override = {
            "applied": False,
            "reason": "classifier_softmax",
            "brain_peak_distance": brain_distance,
            "reconstructed_peak_distance": reconstructed_distance,
            "direct_tolerance": direct_tolerance,
        }
        if raw_predicted_type != "metal" and metal_prob < 0.5 and (direct_by_brain_topology or direct_by_reconstructed_topology):
            type_probs["direct"] = max(float(type_probs.get("direct", 0.0)), 0.90)
            type_probs["indirect"] = min(float(type_probs.get("indirect", 0.0)), 0.10)
            override.update(
                {
                    "applied": raw_predicted_type != "direct",
                    "reason": "vbm_cbm_peak_topology_direct",
                    "direct_by_brain_topology": direct_by_brain_topology,
                    "direct_by_reconstructed_topology": direct_by_reconstructed_topology,
                }
            )
            return "direct", override
        return raw_predicted_type, override


class P2RetrievalInvoker:
    """Frozen MBM -> local HNSW query, separate from supervised inference.

    No new model and no supervised normalization or heuristic recommendation.
    numeric_band is band-band retrieval with paired structures, not a verified
    structure encoder. Manual confirmation is NOT an external human audit.
    """

    FEATURE_SCHEMA = {
        'name': 'mbm_flat6d', 'seq_len': 128,
        'channels': ['VBM_E', 'VBM_curv', 'VBM_k_dist', 'CBM_E', 'CBM_curv', 'CBM_k_dist'],
        'energy_reference': 'fermi_zero', 'k_axis': 'normalized_0_1',
        'segment_policy': 'single_segment',
    }

    def __init__(self, *, store_root, index_name, encoder_path, norm_path):
        from src.evaluation.retrieval import LocalHNSWStore
        self.encoder_path, self.norm_path = Path(encoder_path), Path(norm_path)
        self.encoder_sha256 = self._file_sha(self.encoder_path)
        self.norm_sha256 = self._file_sha(self.norm_path)
        self.norm_stats = json.loads(self.norm_path.read_text(encoding='utf-8'))
        self.gallery = LocalHNSWStore(store_root).load(
            index_name, encoder_sha256=self.encoder_sha256, norm_sha256=self.norm_sha256,
            feature_schema=self.FEATURE_SCHEMA)
        self.encoder = load_ssl_encoder(str(self.encoder_path), compile=False)
        self.encoder.trainable = False
        self._norm_snapshot = json.dumps(self.norm_stats, sort_keys=True, allow_nan=False)
        self._frozen_weights_sha = self._weights_sha()
        self._assert_frozen()

    def _weights_sha(self):
        import hashlib
        digest = hashlib.sha256()
        for variable in self.encoder.weights:
            digest.update(variable.numpy().tobytes())
        return digest.hexdigest()

    def _assert_frozen(self):
        if (self.encoder.trainable or self._weights_sha() != self._frozen_weights_sha
                or json.dumps(self.norm_stats, sort_keys=True, allow_nan=False) != self._norm_snapshot
                or self._file_sha(self.norm_path) != self.norm_sha256
                or self._file_sha(self.encoder_path) != self.encoder_sha256):
            raise ValueError('frozen encoder/norm changed; reload an exactly bound artifact')

    @staticmethod
    def _file_sha(path):
        import hashlib
        with Path(path).open('rb') as source:
            return hashlib.file_digest(source, 'sha256').hexdigest()

    def query_manual(self, source_path, annotations, calibration, *, manual_confirmed=False, k=5,
                     expected_image_sha256=None):
        from src.data import band_structure_dataset as mbm
        from src.vision.physics_reconstructor import PhysicsReconstructor
        from src.evaluation.retrieval import decode_raster_bytes, validate_manual_calibration, validate_manual_extrema
        if manual_confirmed is not True:
            raise ValueError('manual calibration confirmation required for this image')
        if not Path(source_path).is_file():
            raise FileNotFoundError('missing source image: ' + str(source_path))
        import hashlib
        image_bytes = Path(source_path).read_bytes()
        image_sha = hashlib.sha256(image_bytes).hexdigest()
        if expected_image_sha256 is not None and expected_image_sha256 != image_sha:
            raise ValueError('source image changed; reload and manually recalibrate')
        with decode_raster_bytes(image_bytes) as image:
            image_size = image.size
        validate_manual_calibration(annotations, calibration, image_size=image_size)
        self._assert_frozen()
        ann = annotations
        panel = ann['panel']
        tensor = PhysicsReconstructor().reconstruct_from_manual(
            source_path=str(source_path),
            panel_bbox=[panel[0], panel[1], panel[0] + panel[2], panel[1] + panel[3]],
            y_calibration=[{'y': p[1], 'value': v} for p, v in zip(ann['yaxis_pts'], calibration['y_values'])],
            x_calibration=[{'x': p[0], 'value': v} for p, v in zip(ann['xaxis_pts'], calibration['x_values'])],
            vbm_pixel=dict(zip(('x', 'y'), ann['vbm'])), cbm_pixel=dict(zip(('x', 'y'), ann['cbm'])),
            valence_points=[dict(zip(('x', 'y'), p)) for s in ann['vb_strokes'] for p in s],
            conduction_points=[dict(zip(('x', 'y'), p)) for s in ann['cb_strokes'] for p in s],
            fermi_y_pixel=ann['fermi_y'])
        validate_manual_extrema(annotations, calibration, reconstructed=tensor.flat_tensor)
        normalized = mbm.normalize_mbm_inputs(tensor.flat_tensor, self.norm_stats)
        embedding = self.encoder(normalized, training=False, return_features=True).numpy()
        self._assert_frozen()
        result = self.gallery.search(embedding, k=k)
        result['query_provenance'] = {
            'input_space': 'raw6d', 'normalization_count': 1,
            'encoder_sha256': self.encoder_sha256, 'norm_sha256': self.norm_sha256,
            'schema_sha256': self.gallery.manifest['schema_sha256'],
            'image_sha256': image_sha,
            'annotation_sha256': hashlib.sha256(json.dumps(
                {'annotations': annotations, 'calibration': calibration},
                sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest(),
        }
        return result


class ApplicationRecommender:
    """Rule-based device mapping grounded in the model brain outputs.

    The recommender translates the 6D tensor and model-brain predictions into
    structured materials guidance. It does not replace DFT or experiment; every
    recommendation records the physical rule that triggered it.
    """

    def __init__(
        self,
        light_mass_threshold: float = 0.5,
        heavy_mass_threshold: float = 5.0,
        flat_curvature_threshold: float = 0.02,
    ) -> None:
        self.light_mass_threshold = float(light_mass_threshold)
        self.heavy_mass_threshold = float(heavy_mass_threshold)
        self.flat_curvature_threshold = float(flat_curvature_threshold)

    def recommend(self, prediction: BrainPrediction, reconstructed: ReconstructedTensor) -> List[str]:
        return [
            f"{item['recommendation_area']}: {item['physical_basis']}"
            for item in self.recommend_structured(prediction, reconstructed)
        ]

    def recommend_structured(self, prediction: BrainPrediction, reconstructed: ReconstructedTensor) -> List[Dict[str, Any]]:
        gap = float(prediction.line_mode_gap_ev)
        direct_prob = float(prediction.type_probabilities.get("direct", 0.0))
        indirect_prob = float(prediction.type_probabilities.get("indirect", 0.0))
        metal_prob = float(prediction.type_probabilities.get("metal", 0.0))
        is_direct = bool(prediction.predicted_type == "direct" or direct_prob >= max(0.5, indirect_prob))

        vbm_curvature = float(reconstructed.vbm_curvature[reconstructed.vbm_index])
        cbm_curvature = float(reconstructed.cbm_curvature[reconstructed.cbm_index])
        vbm_mass = abs(prediction.vbm_effective_mass_proxy)
        cbm_mass = abs(prediction.cbm_effective_mass_proxy)
        fermi_crossing = bool(reconstructed.is_metallic or prediction.predicted_type == "metal" or metal_prob >= 0.5)
        strong_spin_splitting = bool(reconstructed.metadata.get("strong_spin_splitting", False))
        dirac_cone = bool(reconstructed.metadata.get("dirac_cone_detected", False))

        recs: List[Dict[str, Any]] = []
        if 1.0 <= gap <= 1.6 and is_direct:
            recs.append(
                self._item(
                    "[S级] 理想光伏吸收层/发光二极管 (LED) 候选材料",
                    confidence=max(0.70, direct_prob),
                    physical_basis=(
                        f"Line-mode band gap {gap:.3f} eV is in the 1.0-1.6 eV optoelectronic window "
                        f"and direct-gap probability is {direct_prob:.3f}."
                    ),
                    rule_id="optoelectronics_direct_1p0_1p6ev",
                )
            )
        if gap > 3.0 and self._is_light_mass(cbm_mass):
            recs.append(
                self._item(
                    "[A级] 超宽禁带半导体/透明导电氧化物 (TCO) 候选",
                    confidence=0.78,
                    physical_basis=(
                        f"Band gap {gap:.3f} eV exceeds 3.0 eV and CBM effective-mass proxy "
                        f"{finite_or_none(cbm_mass)} is below {self.light_mass_threshold}."
                    ),
                    rule_id="wide_gap_light_cbm_tco",
                )
            )
        if 0.5 <= gap <= 1.5 and (
            self._is_heavy_mass(vbm_mass)
            or self._is_heavy_mass(cbm_mass)
            or abs(vbm_curvature) <= self.flat_curvature_threshold
            or abs(cbm_curvature) <= self.flat_curvature_threshold
        ):
            recs.append(
                self._item(
                    "[A级] 潜在热电材料 (高 Seebeck 系数特征)",
                    confidence=0.74,
                    physical_basis=(
                        f"Gap {gap:.3f} eV is in the 0.5-1.5 eV thermoelectric window; "
                        f"VBM/CBM curvature=({vbm_curvature:.4g}, {cbm_curvature:.4g}) indicates a flat-band/heavy-mass feature."
                    ),
                    rule_id="thermoelectric_flat_band_mid_gap",
                )
            )
        if dirac_cone or (fermi_crossing and strong_spin_splitting):
            recs.append(
                self._item(
                    "[S级] 拓扑半金属/自旋电子学器件候选",
                    confidence=0.82 if dirac_cone else 0.68,
                    physical_basis=(
                        "The reconstructed tensor indicates a Fermi-level crossing with spin-splitting/Dirac-cone metadata; "
                        "surface-state and Berry-curvature calculations are recommended."
                    ),
                    rule_id="quantum_spintronics_warning",
                )
            )

        hetero = self._heterostructure_recommendation(reconstructed)
        if hetero is not None:
            recs.append(hetero)

        if not recs:
            recs.append(
                self._item(
                    "[B级] 通用半导体候选",
                    confidence=max(0.45, 1.0 - metal_prob),
                    physical_basis=(
                        "No high-specificity optoelectronic, thermoelectric, quantum, or heterostructure rule was triggered; "
                        "inspect full Brillouin-zone bands, stability, and synthesis feasibility before assignment."
                    ),
                    rule_id="general_semiconductor_candidate",
                )
            )
        return recs

    def _item(self, recommendation_area: str, confidence: float, physical_basis: str, rule_id: str) -> Dict[str, Any]:
        return {
            "recommendation_area": recommendation_area,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "physical_basis": physical_basis,
            "rule_id": rule_id,
        }

    def _is_light_mass(self, mass_proxy: float) -> bool:
        return math.isfinite(float(mass_proxy)) and abs(float(mass_proxy)) < self.light_mass_threshold

    def _is_heavy_mass(self, mass_proxy: float) -> bool:
        return (not math.isfinite(float(mass_proxy))) or abs(float(mass_proxy)) > self.heavy_mass_threshold

    def _heterostructure_recommendation(self, reconstructed: ReconstructedTensor) -> Optional[Dict[str, Any]]:
        hetero = reconstructed.metadata.get("heterostructure")
        if not isinstance(hetero, dict):
            return None
        required = {"material_a_vbm", "material_a_cbm", "material_b_vbm", "material_b_cbm"}
        if not required.issubset(hetero):
            return None
        a_vbm = float(hetero["material_a_vbm"])
        a_cbm = float(hetero["material_a_cbm"])
        b_vbm = float(hetero["material_b_vbm"])
        b_cbm = float(hetero["material_b_cbm"])
        if a_vbm >= b_vbm and a_cbm <= b_cbm:
            kind = "Type-I"
            basis = "Material A band edges straddle material B, favoring carrier confinement."
        elif b_vbm >= a_vbm and b_cbm <= a_cbm:
            kind = "Type-I"
            basis = "Material B band edges straddle material A, favoring carrier confinement."
        elif (a_vbm > b_vbm and a_cbm > b_cbm) or (b_vbm > a_vbm and b_cbm > a_cbm):
            kind = "Type-II"
            basis = "Staggered VBM/CBM alignment separates electrons and holes; recommended for photocatalysis."
        else:
            kind = "Type-III"
            basis = "Broken-gap alignment suggests tunneling or infrared-device behavior."
        return self._item(
            f"[A级] {kind} 异质结设计建议",
            confidence=0.76,
            physical_basis=basis,
            rule_id=f"heterostructure_{kind.lower()}",
        )


def effective_mass_proxy(curvature: float, eps: float = 1e-6) -> float:
    """Return an inverse-curvature proxy for effective mass."""

    if abs(curvature) < eps:
        return float("inf")
    return float(1.0 / curvature)


def prediction_to_jsonable(prediction: BrainPrediction) -> Dict[str, Any]:
    predicted_properties = {
        "band_gap_ev": prediction.line_mode_gap_ev,
        "gap_type": prediction.predicted_type,
        "is_direct": bool(prediction.predicted_type == "direct" or prediction.type_probabilities.get("direct", 0.0) >= 0.5),
        "type_probabilities": prediction.type_probabilities,
        "vbm_effective_mass_proxy": finite_or_none(prediction.vbm_effective_mass_proxy),
        "cbm_effective_mass_proxy": finite_or_none(prediction.cbm_effective_mass_proxy),
        "vbm_index": prediction.vbm_index,
        "cbm_index": prediction.cbm_index,
    }
    return {
        "line_mode_gap_ev": prediction.line_mode_gap_ev,
        "type_probabilities": prediction.type_probabilities,
        "predicted_type": prediction.predicted_type,
        "vbm_index": prediction.vbm_index,
        "cbm_index": prediction.cbm_index,
        "vbm_effective_mass_proxy": finite_or_none(prediction.vbm_effective_mass_proxy),
        "cbm_effective_mass_proxy": finite_or_none(prediction.cbm_effective_mass_proxy),
        "predicted_properties": predicted_properties,
        "application_recommendations": prediction.application_recommendation_details,
        "application_recommendation_text": prediction.application_recommendations,
        "metadata": prediction.metadata,
    }


def save_prediction_json(prediction: BrainPrediction, path: str) -> None:
    Path(path).write_text(json.dumps(prediction_to_jsonable(prediction), indent=2), encoding="utf-8")


def finite_or_none(value: float) -> Optional[float]:
    return float(value) if math.isfinite(float(value)) else None
