"""Regression tests for P2 metrics and locally generated FAISS artifacts."""
import hashlib
import json
from src.evaluation import retrieval


def _ann_contract():
    return dict(
        encoder_sha256=hashlib.sha256(b'test-encoder').hexdigest(),
        norm_sha256=hashlib.sha256(b'test-norm').hexdigest(),
        feature_schema={'name': 'mbm_flat6d', 'seq_len': 128,
                        'channels': ['VBM_E', 'VBM_curv', 'VBM_k_dist',
                                     'CBM_E', 'CBM_curv', 'CBM_k_dist'],
                        'energy_reference': 'fermi_zero',
                        'k_axis': 'normalized_0_1', 'segment_policy': 'single_segment'},
    )


def _ann_records():
    return [{'material_id': name, 'formula': formula,
             'structure': {'lattice': [[3., 0., 0.], [0., 3., 0.], [0., 0., 3.]],
                           'species': [formula], 'fractional_coordinates': [[0., 0., 0.]]}}
            for name, formula in [('test-Si', 'Si'), ('test-Ge', 'Ge')]]


def test_hnsw_persist_reload_material_structure_and_uncalibrated_cosine(tmp_path):
    import faiss
    assert hasattr(retrieval, 'LocalHNSWStore'), 'Persistent local HNSW store is missing'
    root = tmp_path / 'local-trusted'
    store = retrieval.LocalHNSWStore(root)
    built = store.build('gallery', np.eye(2), _ann_records(), **_ann_contract())
    assert isinstance(built.index, faiss.IndexHNSWFlat)
    assert built.index.ntotal == 2
    loaded = retrieval.LocalHNSWStore(root).load('gallery', **_ann_contract())
    assert isinstance(loaded.index, faiss.IndexHNSWFlat)
    assert (root / 'gallery' / 'index.faiss').is_file()
    assert loaded.manifest['gallery'] == 'numeric_band'
    assert loaded.manifest['embedding_modality'] == 'numeric_band'
    result = loaded.search(np.array([[1., 0.]]), k=9)
    assert result['retrieval_kind'] == 'band-band'
    assert result['score_semantics'] == 'uncalibrated_cosine_similarity_not_probability'
    assert result['candidates'][0]['material_id'] == 'test-Si'
    assert result['candidates'][0]['structure'] == _ann_records()[0]['structure']
    assert result['candidates'][0]['similarity'] == pytest.approx(1.)
    assert len(result['candidates']) == 2
    assert 'confidence' not in json.dumps(result)
    assert len(loaded.manifest['embedding_sha256']) == 64
    assert len(loaded.manifest['records'][0]['structure_sha256']) == 64

import numpy as np
import pytest

from src.evaluation.retrieval import (
    cosine_similarity_matrix,
    info_nce_temperature,
    mean_average_precision,
    median_rank,
    normalize_embeddings,
    recall_at_k,
    retrieval_metrics,
)


@pytest.mark.parametrize('case', ['duplicate_id', 'missing_id', 'record_count', 'zero', 'nan',
                                  'inf', 'rank', 'empty', 'structure'])
def test_ann_rejects_invalid_gallery_before_creating_artifact(tmp_path, case, monkeypatch):
    import faiss
    def forbidden_native_call(*args, **kwargs):
        pytest.fail('Invalid input reached native FAISS before validation')
    monkeypatch.setattr(faiss, 'IndexHNSWFlat', forbidden_native_call)
    vectors = np.eye(2)
    records = _ann_records()
    if case == 'duplicate_id': records[1]['material_id'] = records[0]['material_id']
    if case == 'missing_id': records[0]['material_id'] = ''
    if case == 'record_count': records.pop()
    if case == 'zero': vectors[0] = 0
    if case == 'nan': vectors[0, 0] = np.nan
    if case == 'inf': vectors[0, 0] = np.inf
    if case == 'rank': vectors = vectors.reshape(-1)
    if case == 'empty': vectors = vectors[:0]
    if case == 'structure': records[0]['structure']['species'] = []
    with pytest.raises(ValueError, match='embedding|record|material_id|structure'):
        retrieval.LocalHNSWStore(tmp_path).build('bad', vectors, records, **_ann_contract())
    assert not (tmp_path / 'bad').exists()


@pytest.mark.parametrize('query', [np.zeros((1, 2)), np.array([[np.nan, 1.]]),
                                   np.ones((1, 3)), np.eye(2), np.ones(2)])
def test_ann_rejects_invalid_query(tmp_path, query, monkeypatch):
    gallery = retrieval.LocalHNSWStore(tmp_path).build('ok', np.eye(2), _ann_records(), **_ann_contract())
    def forbidden_search(*args, **kwargs):
        pytest.fail('Invalid query reached native FAISS before validation')
    monkeypatch.setattr(gallery.index, 'search', forbidden_search)
    with pytest.raises(ValueError, match='query|embedding|dimension'):
        gallery.search(query)


@pytest.mark.parametrize('case', ['manifest', 'binary', 'encoder', 'norm', 'schema', 'unregistered', 'traversal', 'symlink'])
def test_ann_load_rejects_untrusted_or_unbound_bytes_before_native_read(tmp_path, monkeypatch, case):
    import faiss
    import shutil
    store = retrieval.LocalHNSWStore(tmp_path / 'trusted')
    store.build('ok', np.eye(2), _ann_records(), **_ann_contract())
    contract = _ann_contract()
    name = 'ok'
    directory = store.root / name
    if case == 'manifest':
        manifest = json.loads((directory / 'manifest.json').read_bytes())
        manifest['count'] = 999
        (directory / 'manifest.json').write_text(json.dumps(manifest))
    if case == 'binary':
        with (directory / 'index.faiss').open('ab') as f: f.write(b'corruption')
    if case in ('encoder', 'norm'): contract[case+'_sha256'] = hashlib.sha256(b'other').hexdigest()
    if case == 'schema': contract['feature_schema']['seq_len'] = 64
    if case == 'unregistered':
        shutil.copytree(directory, tmp_path / 'uploads' / 'ok')
        store = retrieval.LocalHNSWStore(tmp_path / 'uploads')
    if case == 'traversal': name = '../trusted/ok'
    if case == 'symlink':
        shutil.copytree(directory, tmp_path / 'outside')
        (directory / 'index.faiss').unlink()
        (directory / 'index.faiss').symlink_to(tmp_path / 'outside' / 'index.faiss')
    def forbidden(*args, **kwargs):
        pytest.fail('Untrusted or incompatible artifact reached FAISS deserialization')
    monkeypatch.setattr(faiss, 'read_index', forbidden)
    monkeypatch.setattr(faiss, 'deserialize_index', forbidden)
    with pytest.raises(ValueError, match='SHA|manifest|local|binding|name|trusted|symlink'):
        store.load(name, **contract)


def test_ann_never_overwrites_existing_directory(tmp_path):
    store = retrieval.LocalHNSWStore(tmp_path)
    store.build('ok', np.eye(2), _ann_records(), **_ann_contract())
    before = {p.name: p.read_bytes() for p in (tmp_path / 'ok').iterdir()}
    with pytest.raises(FileExistsError):
        store.build('ok', np.eye(2), _ann_records(), **_ann_contract())
    assert {p.name: p.read_bytes() for p in (tmp_path / 'ok').iterdir()} == before


@pytest.mark.parametrize('case', ['structure_modality', 'bad_sha', 'channels', 'seq_len'])
def test_ann_build_requires_honest_modality_and_valid_binding(tmp_path, case):
    contract = _ann_contract()
    if case == 'structure_modality': contract['gallery'] = 'structure'
    if case == 'bad_sha': contract['encoder_sha256'] = 'not-a-sha'
    if case == 'channels': contract['feature_schema']['channels'].reverse()
    if case == 'seq_len': contract['feature_schema']['seq_len'] = 0
    with pytest.raises(ValueError, match='structure|SHA|schema'):
        retrieval.LocalHNSWStore(tmp_path).build('bad', np.eye(2), _ann_records(), **contract)
    assert not (tmp_path / 'bad').exists()


@pytest.mark.parametrize('key,value', [('version', 900), ('count', 10), ('dimension', 3),
                                     ('algorithm', 'brute_force'), ('embedding_modality', 'structure'),
                                     ('embedding_sha256', '0'*64)])
def test_ann_revalidates_manifest_semantics_even_with_local_receipt(tmp_path, key, value):
    store = retrieval.LocalHNSWStore(tmp_path)
    store.build('ok', np.eye(2), _ann_records(), **_ann_contract())
    path = tmp_path / 'ok' / 'manifest.json'
    data = json.loads(path.read_bytes())
    data[key] = value
    payload = json.dumps(data).encode()
    path.write_bytes(payload)
    (tmp_path / '.receipts' / 'ok.sha256').write_text(hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError, match='manifest|embedding|dimension'):
        store.load('ok', **_ann_contract())


@pytest.mark.parametrize('k', [0, -1, 1.5, True])
def test_ann_query_rejects_invalid_k(tmp_path, k):
    gallery = retrieval.LocalHNSWStore(tmp_path).build('ok', np.eye(2), _ann_records(), **_ann_contract())
    with pytest.raises(ValueError, match='k must'):
        gallery.search(np.array([[1., 0.]]), k=k)


def test_ann_cannot_inherit_probability_claims_from_gallery_metadata(tmp_path):
    records = _ann_records()
    records[0].update(confidence=.99, probability=.99)
    gallery = retrieval.LocalHNSWStore(tmp_path).build('ok', np.eye(2), records, **_ann_contract())
    result = gallery.search(np.array([[1., 0.]]))
    assert 'confidence' not in result['candidates'][0]
    assert 'probability' not in result['candidates'][0]


def _synthetic_audit_claim(tmp_path):
    """SYNTHETIC ONLY adversarial paper claim; NOT a reviewer or real paper."""
    from PIL import Image
    submitted = tmp_path / 'untrusted-submission'
    submitted.mkdir()
    image = submitted / 'SYNTHETIC_ONLY.png'
    document = submitted / 'SYNTHETIC_ONLY_NOT_A_PAPER.txt'
    Image.new('RGB', (200, 200), 'white').save(image)
    document.write_text('SYNTHETIC ONLY - not human evidence')
    ann = {'panel': [10., 10., 180., 180.], 'fermi_y': 100.,
           'xaxis_pts': [[10., 190.], [190., 190.]], 'yaxis_pts': [[10., 190.], [10., 10.]],
           'vbm': [100., 130.], 'cbm': [100., 70.],
           'vb_strokes': [[[10., 155.], [100., 130.], [190., 155.]]],
           'cb_strokes': [[[10., 45.], [100., 70.], [190., 45.]]]}
    cal = {'x_values': [0., 1.], 'y_values': [-3., 3.]}
    reviewer = 'SYNTHETIC_ONLY_NOT_A_REAL_REVIEWER'
    record = dict(record_id='SYNTHETIC_ONLY_RECORD', source_kind='paper_image',
                  document_id='SYNTHETIC_ONLY_DOC', document_path=document.name,
                  document_sha256=hashlib.sha256(document.read_bytes()).hexdigest(),
                  image_path=image.name, image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                  page=1, figure='SYNTHETIC_ONLY_FIG', material_id='SYNTHETIC_ONLY_Si',
                  group_id='SYNTHETIC_ONLY_GROUP', split='train', annotations=ann, calibration=cal,
                  fixture_notice='SYNTHETIC ONLY - forged source claim, not real evidence')
    review = dict(reviewed_by=reviewer, reviewed_at='2000-01-01T00:00:00Z', decision='approved')
    ledger = {k: record[k] for k in ('record_id', 'document_sha256', 'image_sha256')}
    canonical = lambda obj: json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()
    ledger.update(review, annotation_sha256=hashlib.sha256(canonical(dict(annotations=ann, calibration=cal))).hexdigest())
    evidence = submitted / 'SYNTHETIC_ONLY_ledger.json'
    evidence.write_bytes(canonical(ledger))
    record['review'] = dict(review, evidence_path=evidence.name,
                            evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())
    manifest = submitted / 'SYNTHETIC_ONLY_manifest.json'
    manifest.write_text(json.dumps(dict(schema_version=1, records=[record], authenticated=True)))
    return manifest, record, reviewer, canonical


def test_audit_does_not_authenticate_submission_from_allowlist_and_self_claim(tmp_path):
    manifest, record, reviewer, canonical = _synthetic_audit_claim(tmp_path)
    result = retrieval.audit_real_image_manifest(manifest, trusted_reviewers=[reviewer])
    assert result['qualified_records'] == 0, 'Self-claimed ledger is not operator authentication'
    assert 'missing_operator_authentication' in {e['reason'] for e in result['errors']}
    assert result['status'] == 'BLOCKED_REAL_HUMAN_EVIDENCE'


def _synthetic_operator_registry(tmp_path, manifest, record, canonical):
    """SYNTHETIC ONLY local trust fixture. Never registered as real_human."""
    import copy
    semantic = copy.deepcopy(record)
    for key in ('evidence_path', 'evidence_sha256'):
        semantic['review'].pop(key, None)  # remove only circular credential fields
    record_sha = hashlib.sha256(canonical(semantic)).hexdigest()
    evidence = manifest.parent / record['review']['evidence_path']
    ledger = json.loads(evidence.read_bytes())
    ledger['record_sha256'] = record_sha
    evidence.write_bytes(canonical(ledger))
    record['review']['evidence_sha256'] = hashlib.sha256(evidence.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(dict(schema_version=1, records=[record])))
    registry = dict(schema_version=1, scope='synthetic_test_only',
                    reviews={record['record_id']: dict(record_sha256=record_sha,
                             evidence_sha256=record['review']['evidence_sha256'])},
                    assignments={record['group_id']: dict(split=record['split'],
                                 documents={record['document_id']: record['document_sha256']})})
    operator = tmp_path / 'operator-only'
    operator.mkdir(exist_ok=True)
    target = operator / 'SYNTHETIC_ONLY_registry.json'
    target.write_bytes(canonical(registry))
    return target


@pytest.mark.parametrize('field,value', [
    ('material_id', 'SYNTHETIC_ONLY_Ge'), ('source_kind', 'public_arpes'),
    ('document_id', 'SYNTHETIC_ONLY_OTHER_DOC'), ('page', 2), ('figure', 'SYNTHETIC_ONLY_OTHER_FIG'),
    ('group_id', 'SYNTHETIC_ONLY_OTHER_GROUP'), ('split', 'evaluation'),
    ('fixture_notice', 'SYNTHETIC ONLY modified extra semantics')])
def test_audit_cannot_reuse_authenticated_ledger_after_record_semantics_change(tmp_path, field, value):
    manifest, record, reviewer, canonical = _synthetic_audit_claim(tmp_path)
    registry = _synthetic_operator_registry(tmp_path, manifest, record, canonical)
    before = registry.read_bytes()
    original = retrieval.audit_real_image_manifest(manifest, trusted_reviewers=[reviewer], operator_registry_path=registry)
    assert {e['reason'] for e in original['errors']} == {'synthetic_registry_not_real'}
    assert original['qualified_records'] == 0  # synthetic NEVER fills the real quota
    record[field] = value
    manifest.write_text(json.dumps(dict(schema_version=1, records=[record])))
    result = retrieval.audit_real_image_manifest(manifest, trusted_reviewers=[reviewer], operator_registry_path=registry)
    assert 'review_record_binding_mismatch' in {e['reason'] for e in result['errors']}
    assert result['qualified_records'] == 0 and result['gate_passed'] is False
    assert registry.read_bytes() == before


@pytest.mark.parametrize('case', ['missing', 'split', 'document_id', 'document_sha', 'duplicate_document_group'])
def test_audit_requires_independent_frozen_assignment_even_with_reauthenticated_record(tmp_path, case):
    manifest, record, reviewer, canonical = _synthetic_audit_claim(tmp_path)
    registry = _synthetic_operator_registry(tmp_path, manifest, record, canonical)
    frozen = json.loads(registry.read_bytes())['assignments']
    if case == 'missing': frozen = {}
    if case == 'split': record['split'] = 'evaluation'  # entire submitted group moves together
    if case == 'document_id': record['document_id'] = 'SYNTHETIC_ONLY_RELABELED_DOC'
    if case == 'document_sha': frozen[record['group_id']]['documents'][record['document_id']] = '0' * 64
    if case == 'duplicate_document_group':
        frozen['SYNTHETIC_ONLY_OTHER_GROUP'] = dict(split='evaluation', documents={record['document_id']: record['document_sha256']})
    # Even a new independently pinned review cannot rewrite the frozen assignment.
    registry = _synthetic_operator_registry(tmp_path, manifest, record, canonical)
    value = json.loads(registry.read_bytes())
    value['assignments'] = frozen
    registry.write_bytes(canonical(value))
    result = retrieval.audit_real_image_manifest(manifest, trusted_reviewers=[reviewer], operator_registry_path=registry)
    assert 'review_record_binding_mismatch' not in {e['reason'] for e in result['errors']}
    assert 'frozen_assignment_mismatch' in {e['reason'] for e in result['errors']}
    assert result['qualified_records'] == 0 and result['status'] == 'BLOCKED_REAL_HUMAN_EVIDENCE'


@pytest.mark.parametrize('case', ['submission_path', 'missing_path', 'corrupt', 'unregistered', 'changed_ledger'])
def test_audit_operator_authentication_is_not_a_manifest_claim(tmp_path, case):
    manifest, record, reviewer, canonical = _synthetic_audit_claim(tmp_path)
    registry = _synthetic_operator_registry(tmp_path, manifest, record, canonical)
    if case == 'submission_path':
        target = manifest.parent / registry.name
        target.write_bytes(registry.read_bytes())
        registry = target
    if case == 'missing_path': registry = registry.with_name('missing.json')
    if case == 'corrupt': registry.write_text('{broken')
    if case == 'unregistered':
        data = json.loads(registry.read_bytes()); data['reviews'] = {}; registry.write_bytes(canonical(data))
    if case == 'changed_ledger':
        ledger = manifest.parent / record['review']['evidence_path']
        ledger.write_bytes(ledger.read_bytes() + b' ')
        record['review']['evidence_sha256'] = hashlib.sha256(ledger.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(dict(schema_version=1, records=[record], authenticated=True,
                                        operator_registry_path=str(registry))))
    result = retrieval.audit_real_image_manifest(manifest, trusted_reviewers=[reviewer], operator_registry_path=registry)
    assert 'missing_operator_authentication' in {e['reason'] for e in result['errors']}
    assert result['qualified_records'] == 0


def test_real_image_audit_empty_manifest_keeps_external_gate_closed(tmp_path):
    assert hasattr(retrieval, 'audit_real_image_manifest'), 'independent real-image audit gate is missing'
    manifest = tmp_path / 'real_image_audit.json'
    manifest.write_text(json.dumps({'schema_version': 1, 'records': []}))
    result = retrieval.audit_real_image_manifest(manifest)
    assert result['valid_records'] == 0
    assert result['gate_passed'] is False
    assert result['required_min'] == 200 and result['target_max'] == 500
    assert result['status'] == 'BLOCKED_REAL_HUMAN_EVIDENCE'
    assert result['manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()


@pytest.mark.parametrize('case,reason', [
    ('synthetic', 'synthetic_not_real'), ('review', 'missing_human_review'),
    ('image_sha', 'image_sha_mismatch'), ('document_sha', 'document_sha_mismatch'),
    ('duplicate_id', 'duplicate_record_id'), ('duplicate_image', 'duplicate_image_sha'),
    ('group_split', 'group_split_leakage'), ('group_document', 'document_group_conflict')])
def test_real_image_audit_rejects_untraceable_synthetic_or_ungrouped_evidence(tmp_path, case, reason):
    # No invented reviewed_by. These deliberately unreviewed synthetic files
    # test rejection/traceability only, never real-image acceptance or quotas.
    import copy
    from PIL import Image
    image, document = tmp_path / 'SYNTHETIC.png', tmp_path / 'SYNTHETIC-document.txt'
    Image.new('RGB', (200,200), 'white').save(image)
    document.write_text('SYNTHETIC test witness, NOT a paper or human review')
    record = {'record_id': 'synthetic-test-record', 'source_kind': 'synthetic',
              'document_id': 'SYNTHETIC-DOC', 'document_path': document.name,
              'document_sha256': hashlib.sha256(document.read_bytes()).hexdigest(),
              'image_path': image.name, 'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
              'page': 1, 'figure': 'synthetic figure', 'material_id': 'SYNTHETIC-Si',
              'group_id': 'SYNTHETIC-DOC', 'split': 'evaluation', 'review': {}}
    if case == 'image_sha': record['image_sha256'] = '0'*64
    if case == 'document_sha': record['document_sha256'] = '0'*64
    records = [record]
    if case in ('duplicate_id', 'duplicate_image', 'group_split', 'group_document'):
        records.append(copy.deepcopy(record))
        if case != 'duplicate_id': records[1]['record_id'] = 'synthetic-test-other'
        if case == 'group_split': records[1]['split'] = 'validation'
        if case == 'group_document': records[1]['group_id'] = 'SYNTHETIC-OTHER-GROUP'
    path = tmp_path / 'audit.json'
    path.write_text(json.dumps({'schema_version': 1, 'records': records}))
    result = retrieval.audit_real_image_manifest(path)
    assert 'errors' in result, 'audit does not inspect per-record evidence'
    assert reason in {e['reason'] for e in result['errors']}
    assert result['valid_records'] == 0 and result['gate_passed'] is False
    assert result['total_records'] == len(records)


@pytest.mark.parametrize('payload', ['{broken', '[]', '{"schema_version":1,"records":{}}'])
def test_real_image_audit_rejects_corrupt_manifest(tmp_path, payload):
    path = tmp_path / 'broken.json'
    path.write_text(payload)
    with pytest.raises(ValueError, match='manifest'):
        retrieval.audit_real_image_manifest(path)


def test_ann_never_turns_native_missing_neighbor_into_last_material(tmp_path, monkeypatch):
    gallery = retrieval.LocalHNSWStore(tmp_path).build('ok', np.eye(2), _ann_records(), **_ann_contract())
    # FAISS documents -1/-inf when it cannot return all requested neighbors.
    monkeypatch.setattr(gallery.index, 'search', lambda *_: (
        np.array([[1., -np.inf]], dtype=np.float32), np.array([[0, -1]], dtype=np.int64)))
    with pytest.raises(ValueError, match='ANN.*neighbor'):
        gallery.search(np.array([[1., 0.]]), k=2)


def test_normalize_embeddings_unit_norm():
    emb = np.array([[3.0, 4.0], [0.0, 5.0]])
    normed = normalize_embeddings(emb)
    assert np.allclose(np.linalg.norm(normed, axis=1), 1.0)


def test_cosine_similarity_perfect_diagonal():
    emb = normalize_embeddings(np.array([[1, 0], [0, 1], [1, 1]], dtype=float))
    sim = cosine_similarity_matrix(emb, emb)
    assert np.allclose(np.diag(sim), 1.0)
    assert sim.shape == (3, 3)


def test_recall_at_k_perfect_identity():
    # identity embeddings -> each row's true match (col i) ranks first
    sim = cosine_similarity_matrix(np.eye(4), np.eye(4))
    assert recall_at_k(sim, 1) == 1.0
    assert recall_at_k(sim, 5) == 1.0


def test_recall_at_k_random_is_low():
    # cross-modal: query and gallery are DIFFERENT random matrices, so there
    # is no trivial self-match (diagonal is not forced to 1).
    rng = np.random.default_rng(0)
    q = rng.normal(size=(200, 32))
    g = rng.normal(size=(200, 32))
    sim = cosine_similarity_matrix(q, g)
    assert recall_at_k(sim, 1) < 0.1


def test_map_perfect_is_one():
    sim = cosine_similarity_matrix(np.eye(5), np.eye(5))
    assert mean_average_precision(sim) == pytest.approx(1.0)


def test_median_rank_perfect_is_one():
    sim = cosine_similarity_matrix(np.eye(5), np.eye(5))
    assert median_rank(sim) == 1.0


def test_retrieval_metrics_keys():
    sim = cosine_similarity_matrix(np.eye(8), np.eye(8))
    m = retrieval_metrics(sim)
    assert "recall@1" in m and "recall@5" in m and "recall@10" in m
    assert "map" in m and "median_rank" in m


def test_info_nce_decreases_with_alignment():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(64, 16))
    b = rng.normal(size=(64, 16))
    loss_random = info_nce_temperature(a, b)
    loss_aligned = info_nce_temperature(a, a)
    assert loss_aligned < loss_random


def test_info_nce_perfect_alignment_is_floor():
    a = normalize_embeddings(np.eye(4))
    loss = info_nce_temperature(a, a, temperature=1.0)
    # perfect one-hot alignment: logits diag=1, off-diag=0 ->
    # loss = log(1 + (N-1) * exp(-1))
    expected = np.log(1.0 + 3.0 * np.exp(-1.0))
    assert loss == pytest.approx(expected, abs=0.01)
