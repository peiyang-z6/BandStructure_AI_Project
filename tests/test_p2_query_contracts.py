"""P2 engineering witnesses only: synthetic fixtures NEVER count as human audits."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from src.evaluation import retrieval
from src.vision import brain_invoker
from src.vision.physics_reconstructor import PhysicsReconstructor
from src.data import band_structure_dataset as mbm
from src.models import SSLEncoder, load_ssl_encoder


SCHEMA = {'name': 'mbm_flat6d', 'seq_len': 128,
          'channels': ['VBM_E', 'VBM_curv', 'VBM_k_dist', 'CBM_E', 'CBM_curv', 'CBM_k_dist'],
          'energy_reference': 'fermi_zero', 'k_axis': 'normalized_0_1', 'segment_policy': 'single_segment'}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _manual():
    ann = {'panel': [10., 10., 180., 180.], 'fermi_y': 100.,
           'xaxis_pts': [[10., 190.], [190., 190.]], 'yaxis_pts': [[10., 190.], [10., 10.]],
           'vbm': [100., 130.], 'cbm': [100., 70.],
           'vb_strokes': [[[10., 155.], [60., 135.], [100., 130.], [150., 135.], [190., 155.]]],
           'cb_strokes': [[[10., 45.], [60., 65.], [100., 70.], [150., 65.], [190., 45.]]]}
    cal = {'x_values': [0., 1.], 'y_values': [-3., 3.]}
    return ann, cal


def _reconstruct(image, ann, cal):
    return PhysicsReconstructor().reconstruct_from_manual(
        source_path=str(image), panel_bbox=[10., 10., 190., 190.],
        y_calibration=[{'y': p[1], 'value': v} for p, v in zip(ann['yaxis_pts'], cal['y_values'])],
        x_calibration=[{'x': p[0], 'value': v} for p, v in zip(ann['xaxis_pts'], cal['x_values'])],
        vbm_pixel=dict(zip(('x','y'), ann['vbm'])), cbm_pixel=dict(zip(('x','y'), ann['cbm'])),
        valence_points=[dict(zip(('x','y'), p)) for s in ann['vb_strokes'] for p in s],
        conduction_points=[dict(zip(('x','y'), p)) for s in ann['cb_strokes'] for p in s],
        fermi_y_pixel=ann['fermi_y'])


def _unaugmented_energies(ann, cal):
    # Independent oracle for these complete, distinct-k synthetic traces.
    # Match the established float32 knots, linear/PCHIP policy; never add labels.
    from scipy.interpolate import PchipInterpolator
    target = np.linspace(0., 1., 128, dtype=np.float32)
    xs = (cal['x_values'][1] - cal['x_values'][0]) / (ann['xaxis_pts'][1][0] - ann['xaxis_pts'][0][0])
    ys = (cal['y_values'][1] - cal['y_values'][0]) / (ann['yaxis_pts'][1][1] - ann['yaxis_pts'][0][1])
    result = []
    for field in ('vb_strokes', 'cb_strokes'):
        p = np.asarray([p for stroke in ann[field] for p in stroke], dtype=float)
        k = (cal['x_values'][0] + (p[:, 0] - ann['xaxis_pts'][0][0]) * xs).astype(np.float32)
        e = ((p[:, 1] - ann['fermi_y']) * ys).astype(np.float32)
        order = np.argsort(k)
        k, e = k[order], e[order]
        assert len(np.unique(k)) == len(k) and k[0] == 0. and k[-1] == 1.
        values = PchipInterpolator(k, e)(target) if len(k) >= 4 else np.interp(target, k, e)
        result.append(values.astype(np.float32))
    return np.stack(result, axis=-1)


@pytest.fixture
def query_assets(tmp_path):
    tf.keras.utils.set_random_seed(42)
    image = tmp_path / 'SYNTHETIC_not_external_validation.png'
    Image.new('RGB', (200, 200), 'white').save(image)
    ann, cal = _manual()
    raw = _reconstruct(image, ann, cal).flat_tensor
    norm_stats = {'mean': [2., 1., .25, -1., .5, .25], 'std': [3., 2., .5, 4., 2., .5]}
    norm = tmp_path / 'synthetic_norm.json'
    norm.write_text(json.dumps(norm_stats))
    encoder = SSLEncoder(num_features=6, seq_len=128, d_model=8, num_heads=2,
                         num_layers=1, dff=16, projection_dim=4, dropout_rate=.2)
    x_norm = mbm.normalize_mbm_inputs(raw, norm_stats)
    encoder(x_norm, training=False)
    encoder.trainable = False
    model = tmp_path / 'tiny_engineering_encoder.keras'
    encoder.save(model)
    frozen = load_ssl_encoder(model, compile=False)
    frozen.trainable = False
    expected = frozen(x_norm, training=False, return_features=True).numpy()
    twice = frozen(mbm.normalize_mbm_inputs(x_norm, norm_stats), training=False, return_features=True).numpy()
    assert not np.allclose(expected, twice, atol=1e-5), 'fixture must detect double normalization'
    records = [{'material_id': mid, 'formula': element,
                'structure': {'lattice': np.eye(3).tolist(), 'species': [element],
                              'fractional_coordinates': [[0., 0., 0.]]}}
               for mid, element in [('SYNTHETIC-Si', 'Si'), ('SYNTHETIC-Ge', 'Ge')]]
    store = retrieval.LocalHNSWStore(tmp_path / 'local-trusted')
    store.build('tiny', np.concatenate([expected, -expected]), records,
                encoder_sha256=_sha(model), norm_sha256=_sha(norm), feature_schema=SCHEMA)
    return dict(image=image, annotations=ann, calibration=cal, raw=raw, expected=expected,
                norm_stats=norm_stats, model=model, norm=norm, store=store)


def _invoker(assets):
    return brain_invoker.P2RetrievalInvoker(
        store_root=assets['store'].root, index_name='tiny',
        encoder_path=assets['model'], norm_path=assets['norm'])


def test_manual_image_raw6d_same_mbm_frozen_encoder_ann_roundtrip(query_assets, monkeypatch):
    assert hasattr(brain_invoker, 'P2RetrievalInvoker'), 'existing brain lacks ANN query entry'
    a = query_assets
    actual_calls = []
    original = mbm.normalize_mbm_inputs
    def observe_raw(value, stats):
        actual_calls.append(np.array(value, copy=True))
        return original(value, stats)
    monkeypatch.setattr(mbm, 'normalize_mbm_inputs', observe_raw)
    invoker = _invoker(a)
    before = [v.numpy().copy() for v in invoker.encoder.weights]
    result = invoker.query_manual(a['image'], a['annotations'], a['calibration'], manual_confirmed=True)
    assert len(actual_calls) == 1
    np.testing.assert_array_equal(actual_calls[0], a['raw'])
    assert result['candidates'][0]['material_id'] == 'SYNTHETIC-Si'
    assert result['candidates'][0]['similarity'] == pytest.approx(1., abs=1e-6)
    assert result['candidates'][0]['structure']['species'] == ['Si']
    assert result['retrieval_kind'] == 'band-band'
    assert 'confidence' not in json.dumps(result)
    assert result['score_semantics'] == retrieval.SCORE_SEMANTICS
    assert result['query_provenance']['normalization_count'] == 1
    assert result['query_provenance']['encoder_sha256'] == _sha(a['model'])
    assert result['query_provenance']['norm_sha256'] == _sha(a['norm'])
    assert result['query_provenance']['image_sha256'] == _sha(a['image'])
    assert invoker.encoder.trainable is False
    for old, new in zip(before, invoker.encoder.weights): np.testing.assert_array_equal(old, new.numpy())
    repeated = _invoker(a).query_manual(a['image'], a['annotations'], a['calibration'], manual_confirmed=True)
    assert repeated == result


@pytest.mark.parametrize('case', ['translated_off_image', 'panel_overhang', 'xaxis_off_image',
                                  'yaxis_off_panel', 'narrow_k_interval', 'extremum_outside_k',
                                  'fermi_outside_image'])
def test_query_rejects_geometry_outside_decoded_raster_or_calibrated_k(query_assets, case):
    import copy
    a = query_assets
    ann, cal = copy.deepcopy(a['annotations']), copy.deepcopy(a['calibration'])
    if case == 'translated_off_image':
        ann['panel'][0] += 1000.; ann['panel'][1] += 1000.; ann['fermi_y'] += 1000.
        for field in ('vbm', 'cbm'): ann[field] = [v + 1000. for v in ann[field]]
        for field in ('xaxis_pts', 'yaxis_pts'): ann[field] = [[v + 1000. for v in p] for p in ann[field]]
        for field in ('vb_strokes', 'cb_strokes'): ann[field] = [[[v + 1000. for v in p] for p in s] for s in ann[field]]
    if case == 'panel_overhang': ann['panel'] = [10., 10., 200., 180.]
    if case == 'xaxis_off_image': ann['xaxis_pts'][1][1] = 900.
    if case == 'yaxis_off_panel': ann['yaxis_pts'][1][0] = 0.
    if case in ('narrow_k_interval', 'extremum_outside_k'):
        ann['xaxis_pts'] = [[60., 190.], [150., 190.]]
        ann['yaxis_pts'] = [[60., 190.], [60., 10.]]
    if case == 'extremum_outside_k':
        for field in ('vb_strokes', 'cb_strokes'): ann[field] = [ann[field][0][1:-1]]
        ann['vbm'] = [10., 155.]
    if case == 'fermi_outside_image':
        ann['panel'][3] = 300.; ann['fermi_y'] = 250.
    invoker = _invoker(a)
    with pytest.raises(ValueError, match='image|raster|panel|calibrated k'):
        invoker.query_manual(a['image'], ann, cal, manual_confirmed=True)


@pytest.mark.parametrize('case', ['vbm_not_global', 'cbm_not_global', 'vbm_off_curve',
                                  'cbm_off_curve', 'narrow_missed_extremum'])
def test_query_rejects_extrema_inconsistent_with_complete_manual_curve(query_assets, case):
    import copy
    a = query_assets
    ann, cal = copy.deepcopy(a['annotations']), copy.deepcopy(a['calibration'])
    if case == 'vbm_not_global': ann['vbm'] = [10., 155.]
    if case == 'cbm_not_global': ann['cbm'] = [190., 45.]
    if case == 'vbm_off_curve': ann['vbm'] = [100., 110.]
    if case == 'cbm_off_curve': ann['cbm'] = [100., 90.]
    if case == 'narrow_missed_extremum':
        ann['vb_strokes'][0] += [[110.1, 150.], [110.2, 105.], [110.3, 150.]]
    invoker = _invoker(a)
    with pytest.raises(ValueError, match='extremum|extrema|curve'):
        invoker.query_manual(a['image'], ann, cal, manual_confirmed=True)


@pytest.mark.parametrize('case', ['flat', 'tied', 'near_k_only', 'near_energy_only',
                                  'flat_near_marker', 'flat_energy_boundary', 'flat_pixel_boundary',
                                  'reverse_k', 'reverse_energy_axis'])
def test_query_preserves_valid_flat_tied_and_reversed_calibration(query_assets, case):
    import copy
    a = query_assets
    ann, cal = copy.deepcopy(a['annotations']), copy.deepcopy(a['calibration'])
    if case == 'flat':
        for field, y in [('vb_strokes', 130.), ('cb_strokes', 70.)]:
            ann[field] = [[[10., y], [190., y]]]
        ann['vbm'] = [150., 130.]; ann['cbm'] = [60., 70.]
    if case == 'tied':
        ann['vb_strokes'] = [[[10., 155.], [60., 130.], [100., 155.], [150., 130.], [190., 155.]]]
        ann['cb_strokes'] = [[[10., 45.], [60., 70.], [100., 45.], [150., 70.], [190., 45.]]]
        ann['vbm'] = [150., 130.]; ann['cbm'] = [60., 70.]
    if case == 'near_k_only': ann['vbm'] = [101., 130.]; ann['cbm'] = [101., 70.]
    if case == 'near_energy_only': ann['vbm'] = [100., 129.]; ann['cbm'] = [100., 71.]
    if case.startswith('flat_'):
        ann['vb_strokes'] = [[[10., 130.], [190., 130.]]]
        ann['cb_strokes'] = [[[10., 70.], [190., 70.]]]
        if case == 'flat_near_marker': ann['vbm'] = [101., 129.]; ann['cbm'] = [101., 71.]
        if case == 'flat_energy_boundary': ann['vbm'] = [100., 128.5]; ann['cbm'] = [100., 71.5]
        if case == 'flat_pixel_boundary':
            cal['y_values'] = [-.9, .9]
            ann['vbm'] = [100., 128.]; ann['cbm'] = [100., 72.]
    if case == 'reverse_k': cal['x_values'].reverse()
    if case == 'reverse_energy_axis':
        for field in ('vbm', 'cbm'): ann[field][1] = 200. - ann[field][1]
        for field in ('vb_strokes', 'cb_strokes'):
            ann[field] = [[[x, 200. - y] for x, y in s] for s in ann[field]]
        cal['y_values'].reverse()
    expected = _unaugmented_energies(ann, cal)
    raw = _reconstruct(a['image'], ann, cal).flat_tensor[0][:, [0, 3]]
    e_tol = min(.05, 2. * abs((cal['y_values'][1] - cal['y_values'][0]) / 180.))
    assert np.all(abs(raw - expected) <= e_tol + 1.e-6), 'positive must satisfy the FULL energy contract'
    if case.endswith('_boundary'): assert np.max(abs(raw - expected)) == pytest.approx(e_tol, abs=1.e-6)
    result = _invoker(a).query_manual(a['image'], ann, cal, manual_confirmed=True)
    assert result['retrieval_kind'] == 'band-band'
    assert result['query_provenance']['normalization_count'] == 1


@pytest.mark.parametrize('case', ['distance_channel', 'cb_distance_channel', 'energy_channel'])
def test_p2_rejects_reconstructor_channel_conflict_before_normalization(query_assets, monkeypatch, case):
    # Fault injection into a REAL reconstructed tensor; not a fabricated passing
    # ANN/TF result. P2 must guard the consumed raw channels, not trust metadata.
    a = query_assets
    original = PhysicsReconstructor.reconstruct_from_manual
    def corrupt(self, *args, **kwargs):
        tensor = original(self, *args, **kwargs)
        if case == 'distance_channel': tensor.flat_tensor[0, :, 2] += .25
        if case == 'cb_distance_channel': tensor.flat_tensor[0, :, 5] += .25
        if case == 'energy_channel': tensor.flat_tensor[0, 0, 0] = 10.
        return tensor
    monkeypatch.setattr(PhysicsReconstructor, 'reconstruct_from_manual', corrupt)
    with pytest.raises(ValueError, match='extremum|channel'):
        _invoker(a).query_manual(a['image'], a['annotations'], a['calibration'], manual_confirmed=True)


@pytest.mark.parametrize('case', ['three_knots_near_marker', 'five_knots_near_marker',
                                  'curved_marker_boundary', 'non_extremal_vb', 'non_extremal_cb'])
def test_full_consumed_energy_matches_unaugmented_trace_before_normalization(
        query_assets, monkeypatch, tmp_path, request, case):
    # One contract: no-fault interpolation branch/slope/nearest-grid changes,
    # and one non-extremal raw6d corruption per energy channel. No model/ANN mock.
    import copy
    a = query_assets
    ann, cal = copy.deepcopy(a['annotations']), copy.deepcopy(a['calibration'])
    if case == 'three_knots_near_marker':
        ann['vb_strokes'] = [[[10., 155.], [100., 130.], [190., 155.]]]
        ann['vbm'] = [101., 129.]
    if case == 'five_knots_near_marker':
        # Preserve the former "within_tolerance" input: only its marker was
        # within tolerance; the consumed curves each have EIGHT invalid points.
        ann['vbm'] = [101., 129.]; ann['cbm'] = [101., 71.]
    if case == 'curved_marker_boundary': ann['vbm'] = [100., 128.5]
    # The marker itself is legitimate; rejecting every near marker is not a fix.
    retrieval.validate_manual_calibration(ann, cal, image_size=(200, 200))
    raw = _reconstruct(a['image'], ann, cal).flat_tensor
    if not case.startswith('non_extremal'):
        reference = _unaugmented_energies(ann, cal)[:, 0]
        difference = abs(raw[0, :, 0] - reference)
        count, maximum = {'three_knots_near_marker': (116, .3518136739730835),
                          'five_knots_near_marker': (8, .058260202407836914),
                          'curved_marker_boundary': (1, .05007833242416382)}[case]
        assert int(np.count_nonzero(difference > .05 + 1.e-6)) == count
        assert float(difference.max()) == pytest.approx(maximum, abs=1.e-7)
        np.savez(tmp_path / (case + '_energy.npz'), raw=raw, reference=reference)
        request.node.user_properties.extend([
            ('fault_injection', False), ('points_outside_tolerance', count),
            ('max_energy_difference_ev', float(difference.max()))])
    else:
        channel, delta, extreme = (0, -1., np.max) if case == 'non_extremal_vb' else (3, 1., np.min)
        original_reconstruct = PhysicsReconstructor.reconstruct_from_manual
        def corrupt(self, *args, **kwargs):
            tensor = original_reconstruct(self, *args, **kwargs)
            before = tensor.flat_tensor.copy()
            index = tensor.vbm_index if channel == 0 else tensor.cbm_index
            assert index != 0 and before[0, 0, channel] != extreme(before[0, :, channel])
            tensor.flat_tensor[0, 0, channel] += delta
            assert extreme(tensor.flat_tensor[0, :, channel]) == extreme(before[0, :, channel])
            assert tensor.flat_tensor[0, index, channel] == before[0, index, channel]
            np.testing.assert_array_equal(tensor.flat_tensor[:, :, [2, 5]], before[:, :, [2, 5]])
            return tensor
        monkeypatch.setattr(PhysicsReconstructor, 'reconstruct_from_manual', corrupt)
        request.node.user_properties.extend([('fault_channel', channel), ('fault_delta_ev', delta)])
    invoker = _invoker(a)
    weights_before = invoker._weights_sha()
    norm_calls, encoder_calls = [], []
    original_norm, original_call = mbm.normalize_mbm_inputs, invoker.encoder.call
    def observe_norm(value, stats):
        norm_calls.append(np.array(value, copy=True))
        return original_norm(value, stats)
    def observe_encoder(*args, **kwargs):
        encoder_calls.append(kwargs.get('training'))
        return original_call(*args, **kwargs)
    monkeypatch.setattr(mbm, 'normalize_mbm_inputs', observe_norm)
    monkeypatch.setattr(invoker.encoder, 'call', observe_encoder)
    try:
        with pytest.raises(ValueError, match='reconstructed.*energy'):
            invoker.query_manual(a['image'], ann, cal, manual_confirmed=True)
    finally:
        request.node.user_properties.extend([
            ('normalization_calls', len(norm_calls)), ('encoder_calls', len(encoder_calls)),
            ('weights_unchanged', invoker._weights_sha() == weights_before)])
    assert not norm_calls and not encoder_calls
    assert invoker._weights_sha() == weights_before


@pytest.mark.parametrize('channel,sign', [(0, -1.), (3, 1.)])
@pytest.mark.parametrize('energy_range,e_tol', [(3., .05), (.9, .02)])
def test_consumed_energy_tolerance_caps_at_non_extremal_k(
        query_assets, monkeypatch, channel, sign, energy_range, e_tol):
    # Boundary and just-outside controls at several non-extremal k, in BOTH
    # channels. The 2 px cap is stricter than 0.05 eV in the small-energy plot.
    import copy
    a = query_assets
    cal = copy.deepcopy(a['calibration'])
    cal['y_values'] = [-energy_range, energy_range]
    invoker = _invoker(a)
    original_reconstruct, original_norm = PhysicsReconstructor.reconstruct_from_manual, mbm.normalize_mbm_inputs
    for index in (0, 17, 96, 127):
        for excess in (0., 1.e-4):
            calls = []
            def perturb(self, *args, **kwargs):
                tensor = original_reconstruct(self, *args, **kwargs)
                assert index not in (tensor.vbm_index, tensor.cbm_index)
                tensor.flat_tensor[0, index, channel] += sign * (e_tol + excess)
                return tensor
            def observe(value, stats):
                calls.append(True)
                return original_norm(value, stats)
            with monkeypatch.context() as m:
                m.setattr(PhysicsReconstructor, 'reconstruct_from_manual', perturb)
                m.setattr(mbm, 'normalize_mbm_inputs', observe)
                if excess:
                    with pytest.raises(ValueError, match='reconstructed.*energy'):
                        invoker.query_manual(a['image'], a['annotations'], cal, manual_confirmed=True)
                    assert not calls
                else:
                    result = invoker.query_manual(a['image'], a['annotations'], cal, manual_confirmed=True)
                    assert result['retrieval_kind'] == 'band-band' and len(calls) == 1


@pytest.mark.parametrize('case', ['unconfirmed', 'fermi', 'nan_point', 'no_curves', 'x_range',
                                  'outside_panel', 'multisegment', 'normalized', 'missing_image'])
def test_manual_query_rejects_incomplete_or_incompatible_calibration_before_encoder(query_assets, monkeypatch, case):
    import copy
    a = query_assets
    ann, cal = copy.deepcopy(a['annotations']), copy.deepcopy(a['calibration'])
    confirmed, image = True, a['image']
    if case == 'unconfirmed': confirmed = False
    if case == 'fermi': ann['fermi_y'] = None
    if case == 'nan_point': ann['vb_strokes'][0][0][1] = float('nan')
    if case == 'no_curves': ann['vb_strokes'] = []
    if case == 'x_range': cal['x_values'] = [2., 3.]
    if case == 'outside_panel': ann['vbm'] = [300., 130.]
    if case == 'multisegment': ann['segment_boundaries'] = [100.]
    if case == 'normalized': ann['input_space'] = 'mbm_normalized'
    if case == 'missing_image': image = a['image'].with_name('missing.png')
    invoker = _invoker(a)
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid manual input reached MBM normalization')
    monkeypatch.setattr(mbm, 'normalize_mbm_inputs', forbidden)
    with pytest.raises((ValueError, FileNotFoundError), match='manual|calibration|raw|segment|image|point|curve|missing'):
        invoker.query_manual(image, ann, cal, manual_confirmed=confirmed)


@pytest.mark.parametrize('case', ['model_weights', 'norm_memory', 'norm_file'])
def test_query_rejects_changed_frozen_assets(query_assets, case):
    a = query_assets
    invoker = _invoker(a)
    if case == 'model_weights': invoker.encoder.weights[0].assign_add(tf.ones_like(invoker.encoder.weights[0]))
    if case == 'norm_memory': invoker.norm_stats['mean'][0] += 1.
    if case == 'norm_file': a['norm'].write_text('{}')
    with pytest.raises(ValueError, match='frozen|SHA|changed'):
        invoker.query_manual(a['image'], a['annotations'], a['calibration'], manual_confirmed=True)


@pytest.mark.parametrize('case', ['model', 'index', 'norm_mismatch'])
def test_query_missing_artifacts_fail_closed(query_assets, case):
    a = query_assets
    if case == 'model': a['model'].unlink()
    if case == 'index': (a['store'].root / 'tiny' / 'index.faiss').unlink()
    if case == 'norm_mismatch': a['norm'].write_text(json.dumps({'mean':[0.]*6,'std':[1.]*6}))
    with pytest.raises((ValueError, FileNotFoundError)):
        _invoker(a)


def test_real_tk_button_runs_manual_image_ann_not_supervised(query_assets, monkeypatch, tmp_path):
    from scripts import gui_workbench as gui
    a = query_assets
    config = tmp_path / 'local_query_config.json'
    config.write_text(json.dumps({'store_root': str(a['store'].root), 'index_name': 'tiny',
                                  'encoder_path': str(a['model']), 'norm_path': str(a['norm'])}))
    monkeypatch.setenv('BANDSTRUCTURE_P2_QUERY_CONFIG', str(config))
    monkeypatch.setattr(gui, 'WORKBENCH_STATE_DIR', tmp_path / 'state')
    def forbidden_supervised():
        pytest.fail('ANN query entered original supervised brain')
    monkeypatch.setattr(gui, '_get_brain', forbidden_supervised)
    app = gui.BandStructureWorkbench()  # real WSLg Tk root, never a mock
    try:
        assert hasattr(app, '_retrieval_button'), 'existing Tk workbench has no ANN query button'
        app._file_var.set(str(a['image']))
        app._load_file()  # actual existing GUI loader and state path
        app._canvas.restore_annotations(a['annotations'])
        app._x1_var.set(a['calibration']['x_values'][0]); app._x2_var.set(a['calibration']['x_values'][1])
        app._y1_var.set(a['calibration']['y_values'][0]); app._y2_var.set(a['calibration']['y_values'][1])
        # Programmatic synthetic test confirmation, NOT an external human audit.
        app._retrieval_manual_var.set(True)
        app._retrieval_button.invoke()  # real Tcl command -> production callback
        app.root.update_idletasks()
        result = app._last_retrieval_result
        assert result['candidates'][0]['material_id'] == 'SYNTHETIC-Si'
        text = app._retrieval_result.get('1.0', 'end')
        assert 'band-band' in text and 'uncalibrated' in text
        assert 'not probability' in text and 'SYNTHETIC-Si' in text and 'lattice' in text
        assert 'confidence' not in text.lower() and '%' not in text
        Image.new('RGB', (200, 200), 'blue').save(a['image'])
        app._retrieval_button.invoke()
        assert app._last_retrieval_result is None, 'disk image changed but old pixel annotations were queried'
        assert 'image changed' in app._retrieval_result.get('1.0', 'end')
    finally:
        for timer in app.root.tk.call('after', 'info'): app.root.after_cancel(timer)
        app.root.destroy()


@pytest.mark.parametrize('change', ['calibration', 'annotations', 'image'])
def test_tk_manual_confirmation_is_invalidated_when_input_changes(tmp_path, monkeypatch, change):
    from scripts import gui_workbench as gui
    monkeypatch.setattr(gui, 'WORKBENCH_STATE_DIR', tmp_path / 'state')
    app = gui.BandStructureWorkbench()
    try:
        app._retrieval_manual_var.set(True)
        if change == 'calibration': app._y1_var.set(20.)
        if change == 'annotations': app._canvas.restore_annotations(_manual()[0])
        if change == 'image': app._canvas.load_image(Image.new('RGB', (200,200), 'white'))
        assert not app._retrieval_manual_var.get(), 'stale human confirmation reused for different input'
        assert app._last_retrieval_result is None
        app._retrieval_button.invoke()
        assert 'manual calibration confirmation required' in app._retrieval_result.get('1.0','end')
    finally:
        for timer in app.root.tk.call('after', 'info'): app.root.after_cancel(timer)
        app.root.destroy()


def test_query_binds_calibration_provenance(query_assets):
    a = query_assets
    invoker = _invoker(a)
    result = invoker.query_manual(a['image'], a['annotations'], a['calibration'], manual_confirmed=True)
    expected = hashlib.sha256(json.dumps({'annotations':a['annotations'], 'calibration':a['calibration']},
                                          sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
    assert result['query_provenance'].get('annotation_sha256') == expected


@pytest.mark.parametrize('fraction', [None, .9, .8])
def test_query_rejects_truncated_jpeg_that_pillow_verify_accepts(query_assets, fraction):
    import io
    a = query_assets
    buffer = io.BytesIO()
    Image.new('RGB', (200, 200), 'white').save(buffer, format='JPEG')
    complete = buffer.getvalue()
    payload = complete[:-2] if fraction is None else complete[:int(len(complete) * fraction)]
    # Regression premise: header verification succeeds, actual raster fails.
    with Image.open(io.BytesIO(payload)) as image: image.verify()
    with pytest.raises(OSError):
        with Image.open(io.BytesIO(payload)) as image: image.load()
    path = a['image'].with_name('SYNTHETIC_ONLY_truncated.jpg')
    path.write_bytes(payload)
    with pytest.raises(ValueError, match='raster|image|decode'):
        _invoker(a).query_manual(path, a['annotations'], a['calibration'], manual_confirmed=True)


def test_p2_refuses_process_wide_truncated_image_recovery(query_assets, monkeypatch):
    from PIL import ImageFile
    a = query_assets
    monkeypatch.setattr(ImageFile, 'LOAD_TRUNCATED_IMAGES', True)
    with pytest.raises(ValueError, match='strict|truncated|raster'):
        _invoker(a).query_manual(a['image'], a['annotations'], a['calibration'], manual_confirmed=True)
    assert ImageFile.LOAD_TRUNCATED_IMAGES is True  # no unsafe global flag toggling


def test_query_rejects_non_image_source(query_assets):
    a = query_assets
    invoker = _invoker(a)
    a['image'].write_bytes(b'not an image')
    with pytest.raises(ValueError, match='image'):
        invoker.query_manual(a['image'], a['annotations'], a['calibration'], manual_confirmed=True)


def test_real_tk_no_config_reports_failure_without_fabricated_neighbors(tmp_path, monkeypatch):
    from scripts import gui_workbench as gui
    monkeypatch.setattr(gui, 'WORKBENCH_STATE_DIR', tmp_path / 'state')
    monkeypatch.delenv('BANDSTRUCTURE_P2_QUERY_CONFIG', raising=False)
    app = gui.BandStructureWorkbench()
    try:
        app._retrieval_manual_var.set(True)
        app._retrieval_button.invoke()
        assert app._last_retrieval_result is None
        assert 'No valid ANN index configured' in app._retrieval_result.get('1.0','end')
    finally:
        for timer in app.root.tk.call('after', 'info'): app.root.after_cancel(timer)
        app.root.destroy()
