"""P2 metrics, local FAISS HNSW artifacts and independent image traceability.

Only numeric-band galleries are supported: image-derived bands query numeric
bands and return their paired structures. This is band-band
retrieval, NOT qualified structure-band alignment. Scores are uncalibrated cosine
similarities, never probabilities or trustworthy rejection guarantees.

Legacy cross-modal metric helpers below treat the diagonal as the paired match
(no self-exclusion). They do not provide an ANN implementation or certify an
externally validated model. LocalHNSWStore supplies the real persistent ANN.
"""
from __future__ import annotations

from typing import Dict, Tuple
from pathlib import Path
import copy
import hashlib
import json
import re

import numpy as np


SCORE_SEMANTICS = 'uncalibrated_cosine_similarity_not_probability'
MBM_CHANNELS = ['VBM_E', 'VBM_curv', 'VBM_k_dist', 'CBM_E', 'CBM_curv', 'CBM_k_dist']
MANUAL_EXTREMUM_PIXEL_TOLERANCE = 2.0
MANUAL_EXTREMUM_ENERGY_TOLERANCE_EV = 0.05


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _validate_binding(encoder_sha256, norm_sha256, schema):
    for digest in (encoder_sha256, norm_sha256):
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('encoder/norm SHA must be lowercase SHA-256')
    if (not isinstance(schema, dict) or schema.get('name') != 'mbm_flat6d'
            or schema.get('channels') != MBM_CHANNELS
            or type(schema.get('seq_len')) is not int or schema['seq_len'] < 2
            or schema.get('energy_reference') != 'fermi_zero'
            or schema.get('k_axis') != 'normalized_0_1'
            or schema.get('segment_policy') != 'single_segment'):
        raise ValueError('unsupported MBM feature schema; manual query requires a single segment')
    _json_bytes(schema)


def _ann_vectors(value):
    vectors = np.asarray(value, dtype=np.float64)
    if vectors.ndim != 2 or min(vectors.shape) == 0 or not np.isfinite(vectors).all():
        raise ValueError('embedding must be a nonempty finite (N,D) matrix')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError('embedding contains zero or non-finite norm')
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def _validate_records(records, count):
    if not isinstance(records, list) or len(records) != count:
        raise ValueError('record count must match embedding rows')
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('record must be a mapping')
        mid = record.get('material_id')
        if not isinstance(mid, str) or not mid.strip() or mid in seen:
            raise ValueError('material_id must be nonempty and unique (duplicate ID rejected)')
        seen.add(mid)
        try:
            structure = record['structure']
            lattice = np.asarray(structure['lattice'], dtype=float)
            coords = np.asarray(structure['fractional_coordinates'], dtype=float)
            species = structure['species']
            valid = (lattice.shape == (3, 3) and np.isfinite(lattice).all()
                     and abs(np.linalg.det(lattice)) > 1e-12
                     and isinstance(species, list) and len(species) > 0
                     and all(isinstance(s, str) and s.strip() for s in species)
                     and coords.shape == (len(species), 3) and np.isfinite(coords).all())
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError('structure requires finite lattice and per-atom species/coordinates')
    _json_bytes(records)


from src.vision.calibration_contract import (
    decode_raster_bytes, validate_manual_calibration, validate_manual_extrema,
)


class HNSWGallery:
    """Rank numeric bands, returning paired structures, not probabilities."""

    def __init__(self, index, manifest):
        self.index = index
        self.manifest = copy.deepcopy(manifest)

    def search(self, query, k=5):
        if type(k) is not int or k < 1:
            raise ValueError('k must be a positive integer')
        vectors = _ann_vectors(query)
        if vectors.shape != (1, self.index.d):
            raise ValueError('query must contain one embedding with matching dimension')
        scores, indices = self.index.search(vectors, min(k, self.index.ntotal))
        if (not np.isfinite(scores).all() or np.any(indices < 0)
                or np.any(indices >= len(self.manifest['records']))
                or len(np.unique(indices)) != indices.size):
            raise ValueError('ANN returned incomplete or invalid neighbors; no result fabricated')
        candidates = []
        for rank, (score, i) in enumerate(zip(scores[0], indices[0]), 1):
            source = self.manifest['records'][int(i)]
            record = {key: copy.deepcopy(source[key]) for key in
                      ('material_id', 'formula', 'structure', 'structure_sha256') if key in source}
            record.update(rank=rank, similarity=float(score))
            candidates.append(record)
        return {'retrieval_kind': 'band-band', 'gallery': self.manifest['gallery'],
                'score_semantics': SCORE_SEMANTICS, 'candidates': candidates}


class LocalHNSWStore:
    """Application-owned local artifacts, NOT an import/upload API.

    Keep this root and its .receipts registry writable only by the local
    operator, never under an upload directory. A receipt is minted only by
    build(), after local FAISS generation. SHA detects corruption, not a
    malicious local operator who controls this trust root. Never copy receipts
    from an untrusted source. FAISS deserialization is not a safe file parser.
    """

    def __init__(self, root):
        self.root = Path(root)

    def _paths(self, name):
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name):
            raise ValueError('local artifact name must be a simple identifier, not a path')
        directory = self.root / name
        receipt = self.root / '.receipts' / (name + '.sha256')
        for path in (self.root, directory, receipt.parent, receipt,
                     directory / 'manifest.json', directory / 'index.faiss'):
            if path.is_symlink():
                raise ValueError('symlink forbidden in local trusted artifacts')
        return directory, receipt

    def build(self, name, embeddings, records, *, encoder_sha256, norm_sha256,
              feature_schema, gallery='numeric_band'):
        if gallery != 'numeric_band':
            raise ValueError('qualified structure embeddings unavailable; only numeric_band band-band is supported')
        _validate_binding(encoder_sha256, norm_sha256, feature_schema)
        import faiss
        vectors = _ann_vectors(embeddings)
        _validate_records(records, len(vectors))
        directory, receipt = self._paths(name)
        directory.mkdir(parents=True, exist_ok=False)
        index = faiss.IndexHNSWFlat(vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = 64
        index.add(vectors)
        faiss.write_index(index, str(directory / 'index.faiss'))
        entries = copy.deepcopy(records)
        for record in entries:
            record['structure_sha256'] = _sha(_json_bytes(record['structure']))
        manifest = {'version': 1, 'algorithm': 'faiss.IndexHNSWFlat',
                    'metric': 'cosine', 'gallery': gallery, 'embedding_modality': gallery,
                    'count': len(entries), 'dimension': vectors.shape[1], 'records': entries,
                    'encoder_sha256': encoder_sha256, 'norm_sha256': norm_sha256,
                    'feature_schema': feature_schema, 'schema_sha256': _sha(_json_bytes(feature_schema)),
                    'embedding_sha256': _sha(vectors.tobytes()),
                    'index_sha256': _sha((directory / 'index.faiss').read_bytes())}
        manifest_bytes = _json_bytes(manifest)
        (directory / 'manifest.json').write_bytes(manifest_bytes)
        receipt.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with receipt.open('xb') as output:
            output.write(_sha(manifest_bytes).encode('ascii'))
        return self.load(name, encoder_sha256=encoder_sha256, norm_sha256=norm_sha256,
                         feature_schema=feature_schema)

    def load(self, name, *, encoder_sha256, norm_sha256, feature_schema):
        _validate_binding(encoder_sha256, norm_sha256, feature_schema)
        import faiss
        directory, receipt = self._paths(name)
        if not receipt.is_file():
            raise ValueError('No local build receipt; missing/untrusted index cannot be loaded')
        try:
            manifest_bytes = (directory / 'manifest.json').read_bytes()
            if receipt.read_text('ascii') != _sha(manifest_bytes):
                raise ValueError('manifest SHA mismatch with local build receipt')
            manifest = json.loads(manifest_bytes)
            if (manifest['version'] != 1 or manifest['algorithm'] != 'faiss.IndexHNSWFlat'
                    or manifest['metric'] != 'cosine' or manifest['gallery'] != 'numeric_band'
                    or manifest['embedding_modality'] != 'numeric_band'
                    or type(manifest['count']) is not int or manifest['count'] < 1
                    or type(manifest['dimension']) is not int or manifest['dimension'] < 1):
                raise ValueError('invalid manifest semantics')
            try:
                _validate_records(manifest['records'], manifest['count'])
                for record in manifest['records']:
                    if record['structure_sha256'] != _sha(_json_bytes(record['structure'])):
                        raise ValueError('structure SHA mismatch')
            except ValueError as exc:
                raise ValueError('invalid manifest records') from exc
            for field, expected in [('encoder_sha256', encoder_sha256), ('norm_sha256', norm_sha256),
                                    ('feature_schema', feature_schema),
                                    ('schema_sha256', _sha(_json_bytes(feature_schema)))]:
                if manifest[field] != expected:
                    raise ValueError('index binding mismatch: ' + field)
            payload = (directory / 'index.faiss').read_bytes()
            if _sha(payload) != manifest['index_sha256']:
                raise ValueError('index SHA mismatch')
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError('damaged local manifest/index') from exc
        # Deserialize the exact bytes just hashed, not a re-opened mutable path.
        index = faiss.deserialize_index(np.frombuffer(payload, dtype=np.uint8).copy())
        if (not isinstance(index, faiss.IndexHNSWFlat) or index.metric_type != faiss.METRIC_INNER_PRODUCT
                or index.ntotal != manifest['count'] or index.d != manifest['dimension']):
            raise ValueError('manifest/index dimension, count or algorithm mismatch')
        vectors = index.reconstruct_n(0, index.ntotal)
        _ann_vectors(vectors)
        if (_sha(vectors.tobytes()) != manifest['embedding_sha256']
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-5)):
            raise ValueError('embedding SHA or unit norm mismatch')
        return HNSWGallery(index, manifest)


def audit_real_image_manifest(manifest_path, *, trusted_reviewers=(), operator_registry_path=None):
    """Read-only traceability gate, independent of model/FAISS/synthetic tests.

    Manifest: schema_version=1, records=[{record_id, source_kind (paper_image or
    public_arpes), document_id/path/sha256, image_path/sha256, page, figure,
    material_id, group_id, split (train/validation/evaluation), annotations,
    calibration, review}]. Review requires reviewed_by, reviewed_at (ISO-8601),
    decision='approved', evidence_path, evidence_sha256. The ledger must bind
    record_sha256 = SHA256(canonical JSON of the ENTIRE record, omitting only
    review.evidence_path and review.evidence_sha256 to avoid circular hashing).
    Canonical JSON uses sorted keys, compact separators, ensure_ascii=True and
    allow_nan=False; no identifier/value normalization or path rewriting. It also
    retains the legacy record/image/document/annotation/reviewer bindings.

    operator_registry_path is an OUT-OF-BAND, operator-owned registry, outside
    the submission directory, provisioned only after independent authentication
    of the reviewer, ledger, source and frozen allocation. It is never read from
    a manifest field and this module never mints it. Schema: schema_version=1,
    scope='real_human' (or 'synthetic_test_only', which NEVER qualifies),
    reviews={record_id: {record_sha256, evidence_sha256}},
    assignments={group_id: {split, documents: {document_id: document_sha256}}}.
    All submitted records must match both the pinned review and frozen document
    assignment; missing authentication blocks even allowlisted reviewer names.
    trusted_reviewers and registry configuration must not be user-upload inputs.
    File-system ACLs/ownership are the deployment trust boundary, as for the
    local ANN root: a malicious operator controlling it is outside this model.
    Source/ledger SHA is integrity, not proof of authorship. No self-declared
    'authenticated' flag is trusted. Passing traceability is not scientific P2
    acceptance. Relative source/evidence paths resolve against the manifest.
    """
    from datetime import datetime
    path = Path(manifest_path)
    payload = path.read_bytes()
    try:
        data = json.loads(payload)
        if not isinstance(data, dict) or data.get('schema_version') != 1 or not isinstance(data.get('records'), list):
            raise ValueError('invalid real-image manifest schema')
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError('corrupt real-image manifest') from exc
    records, errors = data['records'], []
    # Deployment trust boundary: this path comes ONLY from protected operator
    # configuration, never an upload/manifest. No audit operation mints trust.
    # The operator must independently authenticate ledger authors and sources
    # before installing this registry; a SHA or an allowlisted name alone cannot.
    registry, registry_sha = None, None
    if operator_registry_path is not None:
        try:
            registry_path = Path(operator_registry_path)
            if (any(p.is_symlink() for p in (registry_path, *registry_path.parents))
                    or registry_path.resolve().is_relative_to(path.parent.resolve())):
                raise ValueError('operator registry must be outside the submission directory')
            registry_bytes = registry_path.read_bytes()
            candidate = json.loads(registry_bytes)
            if (candidate.get('schema_version') != 1
                    or candidate.get('scope') not in ('real_human', 'synthetic_test_only')
                    or not isinstance(candidate.get('reviews'), dict)):
                raise ValueError('invalid operator registry')
            registry, registry_sha = candidate, _sha(registry_bytes)
        except (OSError, ValueError, TypeError, AttributeError):
            pass  # Unavailable/invalid external authentication is fail-closed.
    keys = {field: {} for field in ('record_id', 'image_sha256', 'image_pixel_sha256')}
    documents, groups = {}, {}

    def error(i, reason):
        errors.append({'record_index': i, 'reason': reason})

    def local_bytes(record, prefix, i):
        try:
            value = record[prefix + '_path']
            if not isinstance(value, str) or not value:
                raise ValueError('missing path')
            content = (path.parent / value).read_bytes()
            digest = record[prefix + '_sha256']
            if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest) or _sha(content) != digest:
                error(i, prefix + '_sha_mismatch')
            return content
        except (OSError, KeyError, TypeError, ValueError):
            error(i, prefix + '_file_missing')
            return None

    for i, record in enumerate(records):
        if not isinstance(record, dict):
            error(i, 'invalid_record')
            continue
        for field in ('record_id', 'document_id', 'material_id', 'group_id', 'figure'):
            if not isinstance(record.get(field), str) or not record[field].strip():
                error(i, 'missing_' + field)
        if record.get('source_kind') not in ('paper_image', 'public_arpes'):
            error(i, 'synthetic_not_real')
        if type(record.get('page')) is not int or record['page'] < 1:
            error(i, 'missing_document_page')
        if record.get('split') not in ('train', 'validation', 'evaluation'):
            error(i, 'missing_split')
        local_bytes(record, 'document', i)
        image_bytes = local_bytes(record, 'image', i)
        pixel_sha, image_size = None, None
        if image_bytes is not None:
            try:
                with decode_raster_bytes(image_bytes) as rgb:
                    image_size = rgb.size
                    pixel_sha = _sha(_json_bytes(list(rgb.size)) + rgb.tobytes())
            except (OSError, ValueError):
                error(i, 'invalid_image')
        for field, value in [('record_id', record.get('record_id')),
                             ('image_sha256', record.get('image_sha256')),
                             ('image_pixel_sha256', pixel_sha)]:
            if isinstance(value, str) and value:
                keys[field].setdefault(value, []).append(i)
        group, split = record.get('group_id'), record.get('split')
        if isinstance(group, str) and isinstance(split, str):
            groups.setdefault(group, []).append((i, split))
            for document in (record.get('document_id'), record.get('document_sha256')):
                if isinstance(document, str):
                    documents.setdefault(document, []).append((i, group))
        try:
            validate_manual_calibration(record.get('annotations'), record.get('calibration'), image_size=image_size)
            annotation_sha = _sha(_json_bytes({'annotations': record['annotations'], 'calibration': record['calibration']}))
        except (ValueError, TypeError):
            annotation_sha = None
            error(i, 'missing_manual_calibration')
        review = record.get('review')
        registration = registry['reviews'].get(record.get('record_id')) if registry else None
        if (not isinstance(registration, dict) or not isinstance(review, dict)
                or registration.get('evidence_sha256') != review.get('evidence_sha256')
                or not isinstance(registration.get('evidence_sha256'), str)):
            error(i, 'missing_operator_authentication')
        if registry and registry['scope'] != 'real_human':
            error(i, 'synthetic_registry_not_real')
        try:
            assignments = registry['assignments']
            assignment = assignments[group]
            if (assignment['split'] != split
                    or assignment['documents'][record['document_id']] != record['document_sha256']):
                raise ValueError('record differs from frozen assignment')
            # Check against the full frozen pool, not only submitted records.
            for other_group, other in assignments.items():
                if other_group != group and (record['document_id'] in other['documents']
                        or record['document_sha256'] in other['documents'].values()):
                    raise ValueError('document assigned to multiple groups')
        except (KeyError, TypeError, AttributeError, ValueError):
            error(i, 'frozen_assignment_mismatch')
        try:
            if (not isinstance(review, dict) or not isinstance(review.get('reviewed_by'), str)
                    or review['reviewed_by'] not in trusted_reviewers or review.get('decision') != 'approved'
                    or not isinstance(review.get('reviewed_at'), str)):
                raise ValueError('missing authenticated human review')
            timestamp = datetime.fromisoformat(review['reviewed_at'].replace('Z', '+00:00'))
            if timestamp.tzinfo is None:
                raise ValueError('review time must include timezone')
            evidence = local_bytes({'review_path': review['evidence_path'],
                                    'review_sha256': review['evidence_sha256']}, 'review', i)
            ledger = json.loads(evidence)
            # Canonical JSON binds ALL record semantics, including extension
            # fields and source locator strings. Exclude only the two circular
            # ledger-location/digest fields, never material/group/split/review.
            semantic = copy.deepcopy(record)
            for key in ('evidence_path', 'evidence_sha256'):
                semantic['review'].pop(key, None)
            record_sha = _sha(_json_bytes(semantic))
            if (ledger.get('record_sha256') != record_sha
                    or not isinstance(registration, dict)
                    or registration.get('record_sha256') != record_sha):
                error(i, 'review_record_binding_mismatch')
            expected = {field: record[field] for field in ('record_id', 'document_sha256', 'image_sha256')}
            expected.update(annotation_sha256=annotation_sha, reviewed_by=review['reviewed_by'],
                            reviewed_at=review['reviewed_at'], decision='approved')
            if annotation_sha is None or any(ledger.get(k) != v for k, v in expected.items()):
                raise ValueError('review evidence does not bind exact annotation/image/document')
        except (ValueError, TypeError, KeyError, AttributeError):
            error(i, 'missing_human_review')
    for field, values in keys.items():
        for indices in values.values():
            if len(indices) > 1:
                reason = 'duplicate_record_id' if field == 'record_id' else 'duplicate_image_sha'
                for i in indices:
                    error(i, reason)
    for values, reason in ((groups, 'group_split_leakage'), (documents, 'document_group_conflict')):
        for members in values.values():
            if len({value for _, value in members}) > 1:
                for i, _ in members:
                    error(i, reason)
    invalid = {e['record_index'] for e in errors}
    qualified = [i for i in range(len(records)) if i not in invalid]
    evaluation = [i for i in qualified if records[i]['split'] == 'evaluation']
    passed = 200 <= len(evaluation) <= 500 and not errors
    return {'total_records': len(records), 'qualified_records': len(qualified),
            'valid_records': len(evaluation), 'required_min': 200, 'target_max': 500,
            'unique_images': len(keys['image_pixel_sha256']), 'group_count': len(groups),
            'gate_passed': passed, 'status': ('TRACEABILITY_READY_NOT_SCIENTIFIC_ACCEPTANCE' if passed
                                            else 'BLOCKED_REAL_HUMAN_EVIDENCE'),
            'errors': errors, 'manifest_sha256': _sha(payload)}


def normalize_embeddings(emb: np.ndarray) -> np.ndarray:
    """L2-normalize rows of an (N, D) embedding matrix."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-8)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N_a, D) x (N_b, D) -> (N_a, N_b) cosine similarity."""
    a = normalize_embeddings(a)
    b = normalize_embeddings(b)
    return a @ b.T


def recall_at_k(sim: np.ndarray, k: int) -> float:
    """Recall@K: row i's true match is column i (diagonal ground truth)."""
    n = sim.shape[0]
    order = np.argsort(-sim, axis=1)
    hits = sum(int(i in order[i, :k]) for i in range(n))
    return hits / n


def mean_average_precision(sim: np.ndarray) -> float:
    """mAP for the diagonal-pair retrieval task."""
    n = sim.shape[0]
    ap_sum = 0.0
    for i in range(n):
        order = np.argsort(-sim[i])
        true_rank = int(np.where(order == i)[0][0])
        ap_sum += 1.0 / (true_rank + 1)
    return ap_sum / n


def median_rank(sim: np.ndarray) -> float:
    """Median rank (1-indexed) of the true match across queries."""
    n = sim.shape[0]
    ranks = [int(np.where(np.argsort(-sim[i]) == i)[0][0]) + 1 for i in range(n)]
    return float(np.median(ranks)) if ranks else float("nan")


def retrieval_metrics(sim: np.ndarray, ks=(1, 5, 10)) -> Dict[str, float]:
    return {
        **{f"recall@{k}": recall_at_k(sim, k) for k in ks},
        "map": mean_average_precision(sim),
        "median_rank": median_rank(sim),
    }


def info_nce_temperature(emb_a: np.ndarray, emb_b: np.ndarray, temperature: float = 0.07) -> float:
    """InfoNCE loss for a batch of paired embeddings (numpy, for testing)."""
    a = normalize_embeddings(emb_a)
    b = normalize_embeddings(emb_b)
    logits = a @ b.T / temperature  # (N, N)
    logits = logits - np.max(logits, axis=1, keepdims=True)  # stability
    labels = np.arange(logits.shape[0])
    loss = -np.mean(logits[np.arange(logits.shape[0]), labels]
                    - np.log(np.sum(np.exp(logits), axis=1)))
    return float(loss)


def bidirectional_metrics(struct_emb: np.ndarray, band_emb: np.ndarray, ks=(1, 5, 10)) -> Dict[str, Dict[str, float]]:
    """Both directions: structure->band (row i = structure, col i = band) and
    band->structure (row i = band, col i = structure)."""
    s2b = cosine_similarity_matrix(struct_emb, band_emb)
    b2s = cosine_similarity_matrix(band_emb, struct_emb)
    return {
        "structure_to_band": retrieval_metrics(s2b, ks),
        "band_to_structure": retrieval_metrics(b2s, ks),
    }
