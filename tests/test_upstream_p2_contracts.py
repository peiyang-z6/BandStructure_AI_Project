"""Synthetic-only upstream contracts. Positive witnesses always build/train natively."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def write_raw(path, count=16):
    """Engineered bands, not material measurements or reviewer evidence."""
    with h5py.File(path, 'w') as f:
        f.attrs['scope'] = 'test'
        k = np.linspace(0., 1., 24, dtype=np.float32)
        for i in range(count):
            g = f.create_group(f'synthetic-{i:03d}')
            val = -.2 - (k - .3)**2 - i * .01
            cond = .5 + (k - .6)**2 + i * .01
            g.create_dataset('energies', data=np.stack([val-1, val, cond, cond+1])+5.)
            g.create_dataset('k_distances', data=k)
            g.attrs.update(spacegroup_number=i//2+1, band_gap=.7+i*.02,
                           source='materials_project', efermi=5., energy_reference='absolute',
                           energy_quantity='band_energy_eV',
                           num_kpoints=len(k), is_metal=False, is_direct=False,
                           kpath_labels=json.dumps([{'index':0,'label':'G'}, {'index':23,'label':'X'}]))


def build(root, count=16, target_k=128):
    root.mkdir(parents=True, exist_ok=True)
    raw = root/'synthetic.h5'
    write_raw(raw, count)
    out = root/'tensors'
    cmd = [sys.executable, str(ROOT/'scripts/build_ood_tensors.py'), '--h5', str(raw),
           '--metadata', '', '--output', str(out), '--target-k', str(target_k), '--test-scope']
    run = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    assert run.returncode == 0, run.stdout + run.stderr
    return out


@pytest.fixture(scope='module')
def pristine_built(tmp_path_factory):
    return build(tmp_path_factory.mktemp('native-builder'))


@pytest.fixture
def built(pristine_built):
    originals = {p: p.read_bytes() for p in pristine_built.iterdir() if p.is_file()}
    yield pristine_built
    for p, content in originals.items():
        p.write_bytes(content)



def test_real_builder_emits_split_local_raw_credentials(tmp_path):
    built = build(tmp_path)
    tensor = built/'band_tensors_ood_split.npz'
    with np.load(tensor, allow_pickle=False) as z:
        for split in ('train', 'test'):
            c = json.loads((built/f'band_tensors_ood_split.{split}.tensor_contract.json').read_text())
            assert c['kind'] == 'raw_tensor'
            assert c['scope'] == 'test'
            assert c['split'] == split
            assert c['input_space'] == 'raw_canonical_6d'
            assert c['raw_semantics']['energy'] == 'E-E_F'
            assert c['material_ids'] == z[f'material_ids_{split}'].tolist()
            assert c['groups'] == z[f'groups_{split}'].tolist()
            assert c['tensor_ref']['bytes'] == tensor.stat().st_size
            assert np.all(z[f'X_{split}'][:,0,:,0] <= 0)
            assert np.all(z[f'X_{split}'][:,1,:,0] >= 0)
            assert not any(key in c for key in ('samples', 'test_material_ids', 'train_material_ids'))
            assert len(c['source_qualifications']) == len(c['material_ids'])


@pytest.mark.parametrize('damage', [None, 'schema_version', 'kind', 'scope', 'input_space',
                                    'feature_schema', 'material_ids', 'groups', 'tensor_ref',
                                    'raw_semantics', 'source', 'source_qualifications', 'code',
                                    'construction_id'])
def test_raw_validator_requires_bound_schema(built, damage):
    from src.utils import selection_manifest as sm
    validator = getattr(sm, 'validate_raw_tensor_contract', None)
    assert callable(validator), 'shared raw validator is missing'
    path = built/'band_tensors_ood_split.train.tensor_contract.json'
    if damage:
        c = json.loads(path.read_text()); del c[damage]
        path.write_text(json.dumps(c))
        with pytest.raises(ValueError):
            validator(path, tensor_path=built/'band_tensors_ood_split.npz', split='train', allow_test_scope=True)
    else:
        result = validator(path, tensor_path=built/'band_tensors_ood_split.npz', split='train', allow_test_scope=True)
        assert result['contract_ref'] == sm.p2_file_ref(path)
        assert result['scope'] == 'test'
        assert result['material_ids']
        probe = subprocess.run([sys.executable, '-c',
            'import sys; from src.utils.selection_manifest import validate_raw_tensor_contract, validate_inner_selection_manifest; '
            'assert "tensorflow" not in sys.modules and "torch" not in sys.modules'], cwd=ROOT, capture_output=True)
        assert probe.returncode == 0, probe.stderr


@pytest.mark.parametrize('damage', ['input_space', 'feature_schema', 'raw_semantics', 'ids',
                                    'groups', 'expected_ids', 'expected_groups', 'split', 'test_scope',
                                    'tensor_bytes', 'normalized_npz'])
def test_raw_validator_rejects_wrong_space_split_identity(built, damage):
    from src.utils import selection_manifest as sm
    path = built/'band_tensors_ood_split.train.tensor_contract.json'
    tensor = built/'band_tensors_ood_split.npz'
    c = json.loads(path.read_text())
    kwargs = dict(tensor_path=tensor, split='train', allow_test_scope=True)
    if damage == 'input_space': c['input_space'] = 'mbm_normalized'
    if damage == 'feature_schema': c['feature_schema']['channels'].reverse()
    if damage == 'raw_semantics': c['raw_semantics']['energy'] = 'unknown'
    if damage == 'ids': c['material_ids'].reverse()
    if damage == 'groups': c['groups'][0] = 230
    if damage == 'expected_ids': kwargs['expected_ids'] = ['synthetic-wrong']
    if damage == 'expected_groups': kwargs['expected_groups'] = [230]*len(c['groups'])
    if damage == 'split': kwargs['split'] = 'test'
    if damage == 'test_scope': kwargs['allow_test_scope'] = False
    if damage == 'tensor_bytes':
        with tensor.open('ab') as f: f.write(b'tamper')
    if damage == 'normalized_npz':
        with np.load(tensor) as z: arrays = dict(z)
        arrays['input_space'] = np.asarray('mbm_normalized')
        np.savez_compressed(tensor, **arrays)
        c['tensor_ref'] = sm.p2_file_ref(tensor)
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        sm.validate_raw_tensor_contract(path, **kwargs)


@pytest.mark.parametrize('damage', ['energy_reference', 'efermi', 'energy_quantity', 'source', 'k_distances'])
def test_new_builder_blocks_ambiguous_raw_semantics(tmp_path, damage):
    raw = tmp_path/'synthetic.h5'
    write_raw(raw)
    with h5py.File(raw, 'a') as f:
        for g in f.values():
            if damage == 'k_distances': del g['k_distances']
            else: del g.attrs[damage]
    out = tmp_path/'tensors'
    result = subprocess.run([sys.executable, str(ROOT/'scripts/build_ood_tensors.py'),
        '--h5', str(raw), '--metadata', '', '--output', str(out), '--test-scope'],
        cwd=ROOT, text=True, capture_output=True)
    assert result.returncode != 0, 'ambiguous raw source was certified'
    assert not list(out.glob('*.tensor_contract.json'))


@pytest.mark.parametrize('damage', ['missing', 'efermi', 'reference', 'protocol', 'source', 'quantity'])
def test_raw_validator_preserves_qualifications(built, damage):
    from src.utils import selection_manifest as sm
    path = built/'band_tensors_ood_split.train.tensor_contract.json'
    c = json.loads(path.read_text())
    q = c['source_qualifications'][0]
    if damage == 'missing': c['source_qualifications'] = []
    if damage == 'efermi': q['efermi'] = None
    if damage == 'reference': q['energy_reference'] = 'unknown'
    if damage == 'protocol': del q['protocol']
    if damage == 'source': q['source'] = 'unknown'
    if damage == 'quantity': q['energy_quantity'] = 'occupation'
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        sm.validate_raw_tensor_contract(path, tensor_path=built/'band_tensors_ood_split.npz', split='train', allow_test_scope=True)


@pytest.mark.parametrize('damage', ['nonempty', 'oversize', 'unmarked', 'formal_synthetic'])
def test_builder_scope_and_fresh_directory_preflight(tmp_path, damage):
    raw = tmp_path/'synthetic.h5'; write_raw(raw, 33 if damage == 'oversize' else 16)
    if damage == 'unmarked':
        with h5py.File(raw, 'a') as f: del f.attrs['scope']
    out = tmp_path/'tensors'
    if damage == 'nonempty':
        out.mkdir(); (out/'sentinel').write_text('immutable synthetic previous run')
    cmd = [sys.executable, str(ROOT/'scripts/build_ood_tensors.py'), '--h5', str(raw),
           '--metadata', '', '--output', str(out)]
    if damage != 'formal_synthetic': cmd.append('--test-scope')
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    assert result.returncode != 0, 'scope/freshness boundary bypassed'
    assert not list(out.glob('*.tensor_contract.json'))


@pytest.mark.parametrize('damage', ['construction_id', 'input_space_train', 'scope_train', 'feature_schema_train',
                                    'bad_tensor', 'bad_segments', 'source_rows', 'source_kind'])
def test_raw_requires_embedded_construction_evidence(built, damage):
    from src.utils import selection_manifest as sm
    path = built/'band_tensors_ood_split.train.tensor_contract.json'
    tensor = built/'band_tensors_ood_split.npz'
    c = json.loads(path.read_text())
    with np.load(tensor) as z: arrays = dict(z)
    if damage.startswith('source_'):
        c['source']['raw_rows' if damage == 'source_rows' else 'kind'] = 100 if damage == 'source_rows' else 'legacy'
    elif damage == 'bad_tensor': arrays['X_train'][0,0,0,0] = 42.
    elif damage == 'bad_segments': arrays['segment_ids_train'] = arrays['segment_ids_train'][:, :2]
    else: arrays.pop(damage, None)
    np.savez_compressed(tensor, **arrays); c['tensor_ref'] = sm.p2_file_ref(tensor)
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        sm.validate_raw_tensor_contract(path, tensor_path=tensor, split='train', allow_test_scope=True)


@pytest.mark.parametrize('damage', ['missing_required', 'empty', 'absolute', 'traversal', 'other_stage',
                                    'wrong_path', 'wrong_sha', 'source_ref'])
def test_raw_code_is_stage_allowlisted_not_arbitrary_provenance(built, damage):
    from src.utils import selection_manifest as sm
    path = built/'band_tensors_ood_split.train.tensor_contract.json'
    c = json.loads(path.read_text())
    first = next(iter(c['code']))
    if damage == 'missing_required': del c['code'][first]
    if damage == 'empty': c['code'] = {}
    if damage == 'absolute': c['code'][str(ROOT/first)] = c['code'].pop(first)
    if damage == 'traversal': c['code']['../escape.py'] = c['code'].pop(first)
    if damage == 'other_stage': c['code']['scripts/train_ssl.py'] = sm.p2_file_ref(ROOT/'scripts/train_ssl.py')
    if damage == 'wrong_path': c['code'][first]['path'] = '/not-allowed/outer.h5'
    if damage == 'wrong_sha': c['code'][first]['sha256'] = '0'*64
    if damage == 'source_ref': c['source']['h5_ref'] = {}
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        sm.validate_raw_tensor_contract(path, tensor_path=built/'band_tensors_ood_split.npz', split='train', allow_test_scope=True)



def train_native(root):
    tensors = build(root/'raw-build')
    out = root/'model'
    cmd = [sys.executable, str(ROOT/'scripts/train_ssl.py'),
        '--tensor-npz', str(tensors/'band_tensors_ood_split.npz'),
        '--tensor-contract', str(tensors/'band_tensors_ood_split.train.tensor_contract.json'),
        '--test-scope', '--epochs', '2', '--batch-size', '4', '--warmup-epochs', '0',
        '--d-model', '16', '--num-heads', '2', '--num-layers', '1', '--dff', '32', '--projection-dim', '8',
        '--model-dir', str(out), '--checkpoint-dir', str(root/'checkpoints'), '--log-dir', str(root/'logs')]
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    (root/'native-command.json').write_text(json.dumps(cmd))
    (root/'native-stdout.log').write_text(result.stdout+result.stderr)
    assert result.returncode == 0, result.stdout+result.stderr
    return out


def test_actual_ssl_training_emits_anchor_lineage(tmp_path):
    out = train_native(tmp_path)
    c = json.loads((out/'ssl_anchor_contract.json').read_text())
    assert c['scope'] == 'test' and c['outer_test_accessed'] is False
    assert c['fit_material_ids'] and c['selection_material_ids']
    assert not set(c['fit_groups']) & set(c['selection_groups'])
    assert set(c['material_ids']) == set(c['fit_material_ids']+c['selection_material_ids'])
    assert len(c['history']['epochs']) == 2
    assert c['optimizer_iterations'] > 0
    assert c['best_epoch'] in (1, 2)
    assert c['states']['accepted']['sha256'] == c['states']['best']['sha256']
    assert len({c['states'][s]['path'] for s in ('best','last','accepted')}) == 3
    norm = json.loads((out/'ssl_mbm_norm_stats.json').read_text())
    assert norm['fit_material_ids'] == c['fit_material_ids']
    assert norm['fit_groups'] == c['fit_groups']
    assert norm['run_id'] == c['run_id']
    assert norm['encoder_ref'] == c['encoder_ref']
    with np.load(c['source_tensor_ref']['path']) as z:
        ids = z['material_ids_train'].tolist()
        fit = [ids.index(mid) for mid in c['fit_material_ids']]
        x = z['X_train'][fit].transpose(0,2,1,3).reshape(len(fit),128,6)
        np.testing.assert_allclose(norm['mean'], x.mean(axis=(0,1)), rtol=0, atol=0)
    actual_history = json.loads((tmp_path/'logs/ssl_history.json').read_text())
    assert c['history'] == actual_history


@pytest.fixture(scope='module')
def pristine_native(tmp_path_factory):
    return train_native(tmp_path_factory.mktemp('native-ssl'))


@pytest.fixture
def native(pristine_native):
    originals = {p: p.read_bytes() for p in pristine_native.iterdir() if p.is_file()}
    yield pristine_native
    for p, data in originals.items(): p.write_bytes(data)


@pytest.mark.parametrize('damage', [None, 'schema_version', 'scope', 'kind', 'feature_schema', 'encoder_ref',
    'norm_ref', 'run_id', 'material_ids', 'groups', 'fit_material_ids', 'fit_groups',
    'selection_material_ids', 'selection_groups', 'source_tensor_ref', 'source_tensor_contract_ref',
    'history', 'history_ref', 'best_epoch', 'completed_epochs', 'optimizer_iterations',
    'states', 'checkpoints', 'code', 'config_ref', 'runtime', 'outer_test_accessed', 'selection_monitor', 'accepted_source'])
def test_anchor_validator_requires_producer_contract(native, damage):
    from src.utils import selection_manifest as sm
    validate = getattr(sm, 'validate_ssl_anchor_contract', None)
    assert callable(validate), 'shared anchor validator missing'
    path = native/'ssl_anchor_contract.json'
    if damage:
        c = json.loads(path.read_text()); del c[damage]; path.write_text(json.dumps(c))
        with pytest.raises(ValueError):
            validate(path, encoder_path=native/'ssl_mbm_pretrained.keras', norm_path=native/'ssl_mbm_norm_stats.json', allow_test_scope=True)
    else:
        c = validate(path, encoder_path=native/'ssl_mbm_pretrained.keras', norm_path=native/'ssl_mbm_norm_stats.json', allow_test_scope=True)
        assert c['contract_ref'] == sm.p2_file_ref(path)
        assert c['source_tensor_contract_ref']


@pytest.mark.parametrize('damage', ['wrong_norm_path', 'wrong_encoder_path', 'unbound_norm',
    'norm_run', 'norm_encoder', 'norm_mean', 'norm_fit_ids', 'norm_fit_groups',
    'fit_overlap', 'fit_groups', 'selection_groups', 'pool_coverage', 'source_tensor', 'source_contract'])
def test_anchor_norm_and_real_allocation_cannot_be_relabelled(native, damage):
    from src.utils import selection_manifest as sm
    path = native/'ssl_anchor_contract.json'
    c = json.loads(path.read_text())
    norm_path = native/'ssl_mbm_norm_stats.json'; encoder_path = native/'ssl_mbm_pretrained.keras'
    norm = json.loads(norm_path.read_text())
    if damage == 'wrong_norm_path': norm_path = native/'ssl_run_config.json'
    if damage == 'wrong_encoder_path': encoder_path = native/'ssl_mbm_final_epoch2.keras'
    if damage == 'unbound_norm': norm = {'mean': norm['mean'], 'std': norm['std']}
    if damage == 'norm_run': norm['run_id'] = 'wrong'
    if damage == 'norm_encoder': norm['encoder_ref'] = c['states']['last']
    if damage == 'norm_mean': norm['mean'][0] += 1
    if damage == 'norm_fit_ids': norm['fit_material_ids'].reverse()
    if damage == 'norm_fit_groups': norm['fit_groups'][0] = 230
    if damage == 'fit_overlap': c['selection_material_ids'][0] = c['fit_material_ids'][0]
    if damage == 'fit_groups': c['fit_groups'][0] = 230
    if damage == 'selection_groups': c['selection_groups'][0] = c['fit_groups'][0]
    if damage == 'pool_coverage': c['material_ids'] = c['material_ids'][:-1]; c['groups'] = c['groups'][:-1]
    if damage == 'source_tensor': c['source_tensor_ref']['path'] = '/forbidden/raw/outer.h5'
    if damage == 'source_contract': c['source_tensor_contract_ref']['sha256'] = '0'*64
    if damage.startswith('norm_') or damage == 'unbound_norm':
        norm_path.write_text(json.dumps(norm)); c['norm_ref'] = sm.p2_file_ref(norm_path)
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        sm.validate_ssl_anchor_contract(path, encoder_path=encoder_path, norm_path=norm_path, allow_test_scope=True)


@pytest.mark.parametrize('damage', ['empty_history', 'history_file', 'history_metric', 'best_epoch',
    'zero_updates', 'last', 'accepted', 'states_alias', 'checkpoints', 'config', 'code_required',
    'code_escape', 'outer_access', 'monitor', 'archive_origin'])
def test_anchor_requires_real_history_and_frozen_states(native, damage):
    from src.utils import selection_manifest as sm
    import zipfile
    path = native/'ssl_anchor_contract.json'
    c = json.loads(path.read_text())
    if damage == 'empty_history': c['history']['epochs'] = []
    if damage == 'history_file': (native/'ssl_history.json').unlink()
    if damage == 'history_metric': c['history']['epochs'][0]['val']['total'] += 2
    if damage == 'best_epoch': c['best_epoch'] = 999
    if damage == 'zero_updates': c['optimizer_iterations'] = 0
    if damage == 'last': c['states']['last']['sha256'] = '0'*64
    if damage == 'accepted': c['accepted_source'] = 'last'
    if damage == 'states_alias': c['states']['best'] = copy.deepcopy(c['states']['accepted'])
    if damage == 'checkpoints': c['checkpoints'] = {'best': [], 'last': []}
    if damage == 'config': c['config_ref']['sha256'] = '0'*64
    if damage == 'code_required': c['code'].pop('src/engine/ssl_trainer.py')
    if damage == 'code_escape': c['code']['../escape.py'] = c['code'].pop('scripts/train_ssl.py')
    if damage == 'outer_access': c['outer_test_accessed'] = True
    if damage == 'monitor': c['selection_monitor'] = 'outer_loss'
    if damage == 'archive_origin':
        encoder = native/'ssl_mbm_pretrained.keras'
        with zipfile.ZipFile(encoder) as z:
            contents = {key: z.read(key) for key in z.namelist() if key != 'assets/ssl_origin.json'}
        with zipfile.ZipFile(encoder, 'w') as z:
            for key, value in contents.items(): z.writestr(key, value)
        c['encoder_ref'] = sm.p2_file_ref(encoder); c['states']['accepted'] = c['encoder_ref']
        norm_path = native/'ssl_mbm_norm_stats.json'; norm = json.loads(norm_path.read_text())
        norm['encoder_ref'] = c['encoder_ref']; norm_path.write_text(json.dumps(norm))
        c['norm_ref'] = sm.p2_file_ref(norm_path)
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        sm.validate_ssl_anchor_contract(path, encoder_path=native/'ssl_mbm_pretrained.keras',
            norm_path=native/'ssl_mbm_norm_stats.json', allow_test_scope=True)



def test_ssl_defaults_to_formal_gpu(monkeypatch):
    from scripts import train_ssl
    monkeypatch.setattr(sys, 'argv', ['train_ssl.py'])
    args = train_ssl.parse_args()
    assert not args.test_scope and args.require_gpu, 'CPU cannot silently claim formal SSL'


@pytest.mark.parametrize('damage', ['model_nonempty', 'checkpoint_nonempty', 'log_nonempty', 'epochs', 'resume', 'cpu_formal'])
def test_ssl_preflight_blocks_before_data_or_training(built, tmp_path, monkeypatch, damage):
    from scripts import train_ssl
    monkeypatch.setattr(sys, 'argv', ['train_ssl.py', '--test-scope'])
    args = train_ssl.parse_args()
    args.tensor_npz = str(built/'band_tensors_ood_split.npz')
    args.tensor_contract = str(built/'band_tensors_ood_split.train.tensor_contract.json')
    args.model_dir = str(tmp_path/'model'); args.checkpoint_dir = str(tmp_path/'checkpoint'); args.log_dir = str(tmp_path/'log')
    args.epochs = 2
    if damage.endswith('_nonempty'):
        attr = {'model_nonempty':'model_dir','checkpoint_nonempty':'checkpoint_dir','log_nonempty':'log_dir'}[damage]
        p = Path(getattr(args, attr)); p.mkdir(); (p/'sentinel').write_text('prior synthetic evidence')
    if damage == 'epochs': args.epochs = 3
    if damage == 'resume': args.resume = True
    if damage == 'cpu_formal': args.test_scope = False; args.require_gpu = False
    calls = []
    def forbidden_loader(*a, **k):
        calls.append('loaded'); raise AssertionError('training/data boundary crossed before preflight')
    monkeypatch.setattr(train_ssl, 'load_ood_tensor_data', forbidden_loader)
    with pytest.raises((ValueError, FileExistsError, RuntimeError)):
        train_ssl.run_tensor_mbm(args)
    assert not calls


@pytest.mark.parametrize('damage', ['missing_backward', 'test_epoch_cap', 'scope_config', 'formal_cpu'])
def test_ssl_runtime_evidence_cannot_promote_cpu_to_formal(native, damage):
    from src.utils import selection_manifest as sm
    path = native/'ssl_anchor_contract.json'; c = json.loads(path.read_text())
    if damage == 'missing_backward': c['runtime'].pop('backward_devices', None)
    if damage in ('test_epoch_cap', 'scope_config'):
        p = native/'ssl_run_config.json'; cfg = json.loads(p.read_text())
        if damage == 'test_epoch_cap': cfg['epochs'] = 3
        else: cfg['test_scope'] = False
        p.write_text(json.dumps(cfg)); c['config_ref'] = sm.p2_file_ref(p)
    if damage == 'formal_cpu': c['scope'] = 'formal'
    with pytest.raises(ValueError):
        sm._p2_ssl_training_evidence(c, native)



def test_actual_encoder_contains_observed_training_origin(native):
    import zipfile
    with zipfile.ZipFile(native/'ssl_mbm_pretrained.keras') as z:
        assert 'assets/ssl_origin.json' in z.namelist(), 'encoder is not bound to actual trainer observations'
        origin = json.loads(z.read('assets/ssl_origin.json'))
    c = json.loads((native/'ssl_anchor_contract.json').read_text())
    assert origin['fit_material_ids'] == c['fit_material_ids']
    assert origin['history_ref'] == c['history_ref']
    assert origin['initial_state_sha256'] != origin['state_sha256']


def test_arbitrary_random_encoder_cannot_borrow_real_run_history(native, tmp_path):
    from src.utils import selection_manifest as sm
    import shutil
    random = tmp_path/'random_untrained.keras'
    command = [sys.executable, '-c',
        'import tensorflow as tf; from src.models.band_structure_encoder import SSLEncoder; '
        'm=SSLEncoder(num_features=6,seq_len=128,d_model=16,num_heads=2,num_layers=1,dff=32,projection_dim=8); '
        'x=tf.zeros((1,128,6)); m(x); m.reconstruct(x); m.save(__import__("sys").argv[1])', str(random)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True)
    assert result.returncode == 0, result.stderr
    path = native/'ssl_anchor_contract.json'; c = json.loads(path.read_text())
    for key in ('best', 'last', 'accepted'):
        target = Path(c['states'][key]['path']); shutil.copyfile(random, target)
        c['states'][key] = sm.p2_file_ref(target)
    c['encoder_ref'] = c['states']['accepted']
    p = native/'ssl_mbm_norm_stats.json'; norm = json.loads(p.read_text())
    norm['encoder_ref'] = c['encoder_ref']; p.write_text(json.dumps(norm)); c['norm_ref'] = sm.p2_file_ref(p)
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        sm.validate_ssl_anchor_contract(path, encoder_path=native/'ssl_mbm_pretrained.keras', norm_path=p, allow_test_scope=True)


@pytest.mark.parametrize('damage', ['fermi', 'protocol', 'source_sha'])
def test_raw_observed_qualifications_cannot_be_rewritten(built, damage):
    from src.utils import selection_manifest as sm
    p = built/'band_tensors_ood_split.train.tensor_contract.json'; c = json.loads(p.read_text())
    if damage == 'fermi': c['source_qualifications'][0]['efermi'] = 999.
    if damage == 'protocol': c['source_qualifications'][0]['protocol'] = 'claimed verified PBE'
    if damage == 'source_sha': c['source']['h5_ref']['sha256'] = '0'*64
    p.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        sm.validate_raw_tensor_contract(p, tensor_path=built/'band_tensors_ood_split.npz', split='train', allow_test_scope=True)


@pytest.mark.parametrize('damage', ['kpath_labels', 'nonfinite_k', 'reversed_k', 'short_k', 'group'])
def test_new_builder_does_not_certify_inferred_path_or_invalid_group(tmp_path, damage):
    from src.data.ood_tensor_builder import process_band_data
    raw = tmp_path/'synthetic.h5'; write_raw(raw)
    with h5py.File(raw, 'a') as f:
        for g in f.values():
            if damage == 'kpath_labels': del g.attrs['kpath_labels']
            if damage == 'nonfinite_k': g['k_distances'][0] = float('nan')
            if damage == 'reversed_k': g['k_distances'][:] = g['k_distances'][:][::-1]
            if damage == 'short_k': del g['k_distances']; g.create_dataset('k_distances', data=[0., 1.])
            if damage == 'group': g.attrs['spacegroup_number'] = 0
    with pytest.raises(ValueError):
        process_band_data(str(raw), target_k_points=16, canonical_contract=True)


@pytest.mark.parametrize('damage', ['overlap', 'groups', 'missing', 'duplicate_ids'])
def test_tensor_save_checks_actual_split_before_publication(tmp_path, damage):
    from src.data.ood_tensor_builder import process_band_data, save_ood_tensors
    raw = tmp_path/'synthetic.h5'; write_raw(raw, 4)
    x,y,g,ids,t,meta = process_band_data(str(raw), target_k_points=16, canonical_contract=True)
    train, test = np.array([0,1]), np.array([2,3])
    if damage == 'overlap': test = np.array([1,2,3])
    if damage == 'groups': g[2] = g[0]
    if damage == 'missing': test = np.array([2])
    if damage == 'duplicate_ids': ids[2] = ids[0]
    out = tmp_path/'out'
    with pytest.raises(ValueError):
        save_ood_tensors(str(out), x,y,g,ids,t,train,test,meta,16,.8,42)
    assert not (out/'band_tensors_ood_split.npz').exists()



def test_builder_source_change_during_construction_is_not_certified(tmp_path, monkeypatch):
    from src.data import ood_tensor_builder as builder
    raw = tmp_path/'synthetic.h5'; write_raw(raw)
    original = builder.process_band_data
    def change_after_read(*args, **kwargs):
        result = original(*args, **kwargs)
        with h5py.File(raw, 'a') as f: f['synthetic-000']['energies'][0,0] += 1
        return result
    monkeypatch.setattr(builder, 'process_band_data', change_after_read)
    out = tmp_path/'out'
    monkeypatch.setattr(sys, 'argv', ['build', '--h5', str(raw), '--metadata', '', '--output', str(out), '--test-scope'])
    with pytest.raises(ValueError): builder.main()
    assert not list(out.glob('*.tensor_contract.json'))


def test_ssl_source_change_after_load_blocks_before_training(built, tmp_path, monkeypatch):
    from scripts import train_ssl
    original = train_ssl.load_ood_tensor_data
    def change_after_read(*args, **kwargs):
        result = original(*args, **kwargs)
        with Path(args[0]).open('ab') as f: f.write(b'synthetic race')
        return result
    monkeypatch.setattr(train_ssl, 'load_ood_tensor_data', change_after_read)
    monkeypatch.setattr(sys, 'argv', ['ssl', '--tensor-npz', str(built/'band_tensors_ood_split.npz'),
        '--tensor-contract', str(built/'band_tensors_ood_split.train.tensor_contract.json'), '--test-scope',
        '--epochs', '1', '--d-model', '16', '--num-heads', '2', '--num-layers', '1', '--dff', '32',
        '--model-dir', str(tmp_path/'model'), '--checkpoint-dir', str(tmp_path/'checkpoints'), '--log-dir', str(tmp_path/'logs')])
    with pytest.raises(ValueError): train_ssl.run_tensor_mbm(train_ssl.parse_args())
    assert not (tmp_path/'model/ssl_anchor_contract.json').exists()


@pytest.mark.parametrize('damage', ['group_dtype', 'extremum_distance'])
def test_raw_array_types_and_feature_channels_are_not_inferred(built, damage):
    from src.utils import selection_manifest as sm
    p = built/'band_tensors_ood_split.train.tensor_contract.json'; c = json.loads(p.read_text())
    tensor = built/'band_tensors_ood_split.npz'
    with np.load(tensor) as z: arrays = dict(z)
    if damage == 'group_dtype': arrays['groups_train'] = arrays['groups_train'].astype(float)
    else: arrays['X_train'][:,:,:,2] += .5
    np.savez_compressed(tensor, **arrays); c['tensor_ref'] = sm.p2_file_ref(tensor); p.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        sm.validate_raw_tensor_contract(p, tensor_path=tensor, split='train', allow_test_scope=True)


@pytest.mark.parametrize('field,value', [('fit_material_ids', None), ('fit_groups', [True]), ('material_ids', 1)])
def test_anchor_bad_schema_types_fail_closed(native, field, value):
    from src.utils import selection_manifest as sm
    p = native/'ssl_anchor_contract.json'; c = json.loads(p.read_text()); c[field] = value; p.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        sm.validate_ssl_anchor_contract(p, encoder_path=native/'ssl_mbm_pretrained.keras',
            norm_path=native/'ssl_mbm_norm_stats.json', allow_test_scope=True)



def test_producer_code_bindings_include_contract_and_gpu_gates(built, native):
    raw = json.loads((built/'band_tensors_ood_split.train.tensor_contract.json').read_text())
    ssl = json.loads((native/'ssl_anchor_contract.json').read_text())
    assert 'src/utils/selection_manifest.py' in raw['code']
    assert {'src/utils/selection_manifest.py', 'src/utils/__init__.py'} <= set(ssl['code'])


def test_shared_anchor_mismatch_diagnostic_identifies_anchor(native):
    from src.utils import selection_manifest as sm
    norm = native/'ssl_mbm_norm_stats.json'
    stats = json.loads(norm.read_text()); stats['mean'][0] += .1
    norm.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match='anchor.*normalization'):
        sm.validate_ssl_anchor_contract(native/'ssl_anchor_contract.json',
            encoder_path=native/'ssl_mbm_pretrained.keras', norm_path=norm, allow_test_scope=True)


@pytest.mark.parametrize('damage', ['formal', 'oversize'])
def test_cpu_ssl_test_scope_does_not_load_a_formal_pool(built, tmp_path, monkeypatch, damage):
    import builtins
    import io
    from scripts import train_ssl
    from src.utils import selection_manifest as sm
    # Destructive negative only: relabel a real synthetic build as formal.
    # Never train it, and never publish it as a formal positive witness.
    tensor = built/'band_tensors_ood_split.npz'
    path = built/'band_tensors_ood_split.train.tensor_contract.json'
    c = json.loads(path.read_text())
    with np.load(tensor) as z: arrays = dict(z)
    arrays['scope_train'] = np.asarray('formal' if damage == 'formal' else 'test')
    np.savez_compressed(tensor, **arrays)
    c['scope'] = 'formal' if damage == 'formal' else 'test'
    if damage == 'oversize': c['source']['raw_rows'] = 33
    c['tensor_ref'] = sm.p2_file_ref(tensor)
    path.write_text(json.dumps(c))
    monkeypatch.setattr(sys, 'argv', ['ssl', '--tensor-npz', str(tensor), '--tensor-contract', str(path),
        '--test-scope', '--epochs', '1', '--model-dir', str(tmp_path/'model'),
        '--checkpoint-dir', str(tmp_path/'checkpoints'), '--log-dir', str(tmp_path/'logs')])
    events = []
    for module in (builtins, io):
        original = module.open
        def guard(file, *a, _original=original, **kw):
            if isinstance(file, (str, bytes, os.PathLike)) and Path(os.fsdecode(file)).resolve() == tensor.resolve():
                events.append('open/hash'); raise AssertionError('NPZ opened before scope/row rejection')
            return _original(file, *a, **kw)
        monkeypatch.setattr(module, 'open', guard)
    def forbidden_decode(self, key):
        events.append(key); raise AssertionError('NPZ decoded before scope/row rejection')
    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', forbidden_decode)
    with pytest.raises(ValueError, match='test scope.*synthetic|source/scope'):
        train_ssl.run_tensor_mbm(train_ssl.parse_args())
    assert events == []
    assert not (tmp_path/'model').exists()


@pytest.mark.parametrize('damage', ['receipt_symlink', 'tensor_symlink', 'directory_symlink', 'traversal', 'relative'])
def test_up_s1_source_paths_rejected_before_open(native, tmp_path, monkeypatch, damage):
    import builtins
    import io
    from src.utils import selection_manifest as sm
    c = json.loads((native/'ssl_anchor_contract.json').read_text())
    source = Path(c['source_tensor_ref']['path'])
    receipt = Path(c['source_tensor_contract_ref']['path'])
    directory = tmp_path/'alias'; directory.mkdir()
    npz = directory/source.name; cp = directory/receipt.name
    if damage == 'directory_symlink':
        directory.rmdir(); directory.symlink_to(source.parent, target_is_directory=True)
    else:
        if damage == 'tensor_symlink': npz.symlink_to(source)
        else: npz.write_bytes(source.read_bytes())
        if damage == 'receipt_symlink': cp.symlink_to(receipt)
        else: cp.write_bytes(receipt.read_bytes())
    if damage == 'traversal':
        sub = directory/'sub'; sub.mkdir()
        npz = sub/'..'/source.name; cp = sub/'..'/receipt.name
    if damage == 'relative':
        npz = Path(os.path.relpath(npz, Path.cwd())); cp = Path(os.path.relpath(cp, Path.cwd()))
    c['source_tensor_ref'] = dict(sm.p2_file_ref(npz), path=str(npz))
    c['source_tensor_contract_ref'] = dict(sm.p2_file_ref(cp), path=str(cp))
    attacked = tmp_path/'negative_anchor.json'; attacked.write_text(json.dumps(c))
    forbidden = {npz.resolve(), cp.resolve()}
    opened = []
    for module in (builtins, io):
        original = module.open
        def watch(file, *args, _original=original, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)) and Path(os.fsdecode(file)).resolve() in forbidden:
                opened.append(str(file))
            return _original(file, *args, **kwargs)
        monkeypatch.setattr(module, 'open', watch)
    with pytest.raises(ValueError):
        sm.validate_ssl_anchor_contract(attacked, encoder_path=native/'ssl_mbm_pretrained.keras',
            norm_path=native/'ssl_mbm_norm_stats.json', allow_test_scope=True)
    assert opened == [], f'upstream path read before rejection: {opened}'


@pytest.mark.parametrize('layer', ['nested', 'external', 'nested_external'])
@pytest.mark.parametrize('field,value', [('efermi', 5.15), ('source', 'aflow'),
    ('energy_reference', 'fermi_shifted_zero'), ('energy_quantity', 'occupation'),
    ('is_metal', True), ('is_direct', True), ('spacegroup_number', 230),
    ('kpath_labels', [{'index': 0, 'label': 'X'}, {'index': 23, 'label': 'L'}])])
def test_up_l1a_metadata_conflict_is_not_last_wins(tmp_path, monkeypatch, layer, field, value):
    from src.data import ood_tensor_builder as builder
    raw = tmp_path/'synthetic.h5'; write_raw(raw, 4)
    external = {}
    with h5py.File(raw, 'a') as f:
        for mid, g in f.items():
            original = g.attrs[field]
            if layer == 'nested_external':
                del g.attrs[field]
                g.create_group('metadata').attrs[field] = original
            if layer == 'nested':
                g.create_group('metadata').attrs[field] = json.dumps(value) if isinstance(value, list) else value
            else:
                external[mid] = {field: value}
    metadata = tmp_path/'external.json'; metadata.write_text(json.dumps(external))
    out = tmp_path/'tensors'
    monkeypatch.setattr(sys, 'argv', ['builder', '--h5', str(raw), '--metadata', str(metadata),
        '--output', str(out), '--test-scope'])
    with pytest.raises(ValueError, match='[Cc]onflict'):
        builder.main()
    assert not list(out.glob('*.tensor_contract.json'))


@pytest.mark.parametrize('damage', ['duplicate_list_id', 'duplicate_json_key', 'efermi_alias',
                                   'source_alias', 'group_alias', 'protocol_alias'])
def test_up_l1a_duplicate_metadata_and_aliases_cannot_hide_conflicts(tmp_path, damage):
    from src.data.ood_tensor_builder import process_band_data
    raw = tmp_path/'synthetic.h5'; write_raw(raw, 4)
    with h5py.File(raw, 'a') as f:
        f['synthetic-000'].attrs['dft_type'] = 'PBE'
    metadata = tmp_path/'metadata.json'
    payload = {'synthetic-000': {}}
    if damage == 'duplicate_list_id':
        payload = [{'material_id': 'synthetic-000', 'efermi': 5.15},
                   {'material_id': 'synthetic-000', 'efermi': 5.}]
    elif damage == 'duplicate_json_key':
        metadata.write_text('{"synthetic-000":{"efermi":5.15,"efermi":5.0}}')
    else:
        field, value = {'efermi_alias': ('Efermi', 5.15), 'source_alias': ('provider', 'aflow'),
                        'group_alias': ('spacegroup', 230), 'protocol_alias': ('protocol', 'LDA')}[damage]
        payload['synthetic-000'][field] = value
    if damage != 'duplicate_json_key': metadata.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='[Cc]onflict'):
        process_band_data(str(raw), str(metadata), canonical_contract=True)


def test_up_l1a_equivalent_and_empty_metadata_preserves_canonical_and_legacy_api(tmp_path):
    from src.data.ood_tensor_builder import process_band_data
    raw = tmp_path/'synthetic.h5'; write_raw(raw, 4)
    metadata = tmp_path/'metadata.json'
    metadata.write_text(json.dumps({'synthetic-000': {'efermi': 5, 'source': None,
        'kpath_labels': [{'index': 0, 'label': 'G'}, {'index': 23, 'label': 'X'}]}}))
    actual = process_band_data(str(raw), str(metadata), canonical_contract=True)
    control = process_band_data(str(raw), canonical_contract=True)
    np.testing.assert_array_equal(actual[0], control[0])
    assert len(actual) == 6 and actual[0].shape == (4, 2, 128, 3)
    legacy = process_band_data(str(raw))
    np.testing.assert_allclose(legacy[0][:, :, :, 0] - 5., actual[0][:, :, :, 0], atol=1e-6)


def write_segment_raw(path, kind):
    write_raw(path, 8)
    k = np.array([0, .02, .04, .06, .08, .10, .25, .40, .55, .70, .85, 1.])
    labels = [{'index': i, 'label': label} for i, label in ((0,'G'),(5,'X'),(11,'L'))]
    v = -.5 - k**2
    if kind == 'disconnected':
        k = np.array([0,.1,.2,.3,.4,.5,.5,.6,.7,.8,.9,1.])
        labels = [{'index': i, 'label': label} for i, label in ((0,'G'),(5,'X'),(6,'L'),(11,'W'))]
        v = np.array([-1.]*6 + [-3.]*6)
    if kind == 'continuous_kink':
        v = np.array([-.5,-.52,-.57,-.59,-.66,-.70,-1.,-1.05,-1.1,-1.2,-1.3,-1.4])
    with h5py.File(path, 'a') as f:
        for g in f.values():
            del g['energies']; del g['k_distances']
            g.create_dataset('energies', data=np.stack([v-1., v, -v, -v+1.]))
            g.create_dataset('k_distances', data=k)
            g.attrs.update(efermi=0., num_kpoints=len(k), kpath_labels=json.dumps(labels))
    return k, v, labels


@pytest.mark.parametrize('layer', ['attrs', 'nested', 'external_string', 'external_structure'])
def test_up_l1a_encoded_path_duplicate_is_rejected(tmp_path, monkeypatch, layer):
    from src.data import ood_tensor_builder as builder
    raw = tmp_path/'synthetic.h5'
    write_segment_raw(raw, 'nonuniform')
    encoded = '[{"index":0,"label":"G"},{"index":2,"index":5,"label":"X"},{"index":11,"label":"L"}]'
    with h5py.File(raw, 'a') as f:
        for g in f.values():
            if layer == 'attrs':
                g.attrs['kpath_labels'] = encoded
            elif layer == 'nested':
                del g.attrs['kpath_labels']
                g.create_group('metadata').attrs['kpath_labels'] = encoded
    metadata = tmp_path/'metadata.json'
    if layer == 'external_string':
        metadata.write_text(json.dumps({'synthetic-000': {'kpath_labels': encoded}}))
    elif layer == 'external_structure':
        metadata.write_text('{"synthetic-000":{"kpath_labels":' + encoded + '}}')
    out = tmp_path/'tensors'
    monkeypatch.setattr(sys, 'argv', ['builder', '--h5', str(raw),
        '--metadata', str(metadata) if metadata.exists() else '',
        '--output', str(out), '--target-k', '128', '--test-scope'])
    with pytest.raises(ValueError, match='[Cc]onflict'):
        builder.main()
    assert not list(out.glob('*.tensor_contract.json'))


def test_up_l1a_encoded_conflict_keeps_explicit_legacy_parser_boundary():
    from src.data.ood_tensor_builder import _parse_json_attr, _canonical_metadata_value
    encoded = b'{"index":2,"index":5}'
    assert _parse_json_attr(encoded, {}) == {'index': 5}
    with pytest.raises(ValueError, match='[Cc]onflict'):
        _canonical_metadata_value({'path': [encoded]})
    assert _canonical_metadata_value('{"index":5,"index":5}') == {'index': 5}
    assert _parse_json_attr('not JSON', 'fallback', canonical_contract=True) == 'fallback'


@pytest.mark.parametrize('kind', ['nonuniform', 'disconnected', 'continuous_kink'])
def test_up_l1b_native_segment_grid_and_interpolation(tmp_path, monkeypatch, kind):
    from src.data import ood_tensor_builder as builder
    from src.utils import selection_manifest as sm
    from scipy.interpolate import PchipInterpolator
    raw = tmp_path/'synthetic.h5'; k, v, labels = write_segment_raw(raw, kind)
    out = tmp_path/'tensors'
    monkeypatch.setattr(sys, 'argv', ['builder', '--h5', str(raw), '--metadata', '',
        '--output', str(out), '--target-k', '128', '--test-scope'])
    builder.main()
    sm.validate_raw_tensor_contract(out/'band_tensors_ood_split.train.tensor_contract.json',
        tensor_path=out/'band_tensors_ood_split.npz', split='train', allow_test_scope=True)
    with np.load(out/'band_tensors_ood_split.npz') as z:
        segments = z['segment_ids_train'][0]; energy = z['X_train'][0,0,:,0]
    target = np.linspace(k[0], k[-1], 128)
    boundary = int(np.searchsorted(target, k[5])) if kind == 'disconnected' else int(np.rint(k[5]*127))
    assert np.flatnonzero(np.diff(segments) != 0).tolist() == [boundary-1]
    if kind == 'disconnected':
        np.testing.assert_allclose(energy[target < .5], -1., rtol=0, atol=1e-6)
        np.testing.assert_allclose(energy[target > .5], -3., rtol=0, atol=1e-6)
    else:
        expected = np.concatenate([
            PchipInterpolator(k[:6], v[:6].astype(np.float32))(np.clip(target[:boundary], k[0], k[5])),
            PchipInterpolator(k[5:], v[5:].astype(np.float32))(np.clip(target[boundary:], k[5], k[-1]))])
        np.testing.assert_allclose(energy, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize('kind', ['nonuniform', 'continuous_kink', 'disconnected'])
@pytest.mark.parametrize('tiny_scale,offset', [(1.e-11, 0.), (1.e-12, 0.),
                                             (1.e-13, 0.), (1.e-14, -5.e-14)])
def test_up_l1b_positive_tiny_axis_retains_source_interpolation(tmp_path, monkeypatch, kind, tiny_scale, offset):
    from src.data import ood_tensor_builder as builder
    from src.utils import selection_manifest as sm
    results = []
    for name, scale, shift in [('control', 1., 0.), ('tiny', tiny_scale, offset)]:
        root = tmp_path/name; root.mkdir()
        raw = root/'synthetic.h5'
        write_segment_raw(raw, kind)
        with h5py.File(raw, 'a') as f:
            for g in f.values():
                g['k_distances'][:] = g['k_distances'][:] * scale + shift
                np.testing.assert_array_equal(
                    builder._source_k_axis(g['k_distances'][:], len(g['k_distances'])),
                    g['k_distances'][:])
        out = root/'tensors'
        monkeypatch.setattr(sys, 'argv', ['builder', '--h5', str(raw), '--metadata', '',
            '--output', str(out), '--target-k', '128', '--test-scope'])
        builder.main()
        sm.validate_raw_tensor_contract(out/'band_tensors_ood_split.train.tensor_contract.json',
            tensor_path=out/'band_tensors_ood_split.npz', split='train',
            allow_test_scope=True, required_scope='test')
        with np.load(out/'band_tensors_ood_split.npz', allow_pickle=False) as z:
            assert np.isfinite(z['X_train']).all()
            results.append((z['material_ids_train'], z['segment_ids_train'], z['X_train'][:,:,:,0]))
    control, tiny = results
    np.testing.assert_array_equal(tiny[0], control[0])
    np.testing.assert_array_equal(tiny[1], control[1])
    np.testing.assert_allclose(tiny[2], control[2], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('damage', ['label_order', 'wrong_distance', 'unlabelled_duplicate', 'collapsed_target_segment'])
def test_up_l1b_unsupported_path_is_not_certified(tmp_path, monkeypatch, damage):
    from src.data import ood_tensor_builder as builder
    raw = tmp_path/'synthetic.h5'; k, v, labels = write_segment_raw(raw, 'nonuniform')
    if damage == 'label_order': labels.insert(2, {'index': 3, 'label': 'Q'})
    if damage == 'wrong_distance': labels[1]['k_distance'] = .9
    if damage == 'collapsed_target_segment': labels.insert(1, {'index': 1, 'label': 'Q'})
    with h5py.File(raw, 'a') as f:
        for g in f.values():
            g.attrs['kpath_labels'] = json.dumps(labels)
            if damage == 'unlabelled_duplicate': g['k_distances'][3] = k[2]
    out = tmp_path/'tensors'
    monkeypatch.setattr(sys, 'argv', ['builder', '--h5', str(raw), '--metadata', '',
        '--output', str(out), '--target-k', '16' if damage == 'collapsed_target_segment' else '128', '--test-scope'])
    with pytest.raises(ValueError, match='[Pp]ath|segment|coordinate'):
        builder.main()
    assert not list(out.glob('*.tensor_contract.json'))


def ssl_test_args(tensors, out, monkeypatch):
    from scripts import train_ssl
    monkeypatch.setattr(sys, 'argv', ['ssl', '--tensor-npz', str(tensors/'band_tensors_ood_split.npz'),
        '--tensor-contract', str(tensors/'band_tensors_ood_split.train.tensor_contract.json'), '--test-scope',
        '--epochs', '1', '--batch-size', '4', '--warmup-epochs', '0', '--d-model', '16', '--num-heads', '2',
        '--num-layers', '1', '--dff', '32', '--projection-dim', '8', '--model-dir', str(out/'model'),
        '--checkpoint-dir', str(out/'checkpoints'), '--log-dir', str(out/'logs')])
    return train_ssl.parse_args()


@pytest.mark.parametrize('k,minimum,maximum,ratio', [(2,5,15,.2), (3,5,15,.2),
    (16,5,15,.2), (64,5,15,.2), (128,5,30,.2), (128,1,15,.2), (128,0,15,.2), (128,5,15,.9)])
def test_up_l2a_unsupported_mask_preflight_before_model(tmp_path, monkeypatch, k, minimum, maximum, ratio):
    from scripts import train_ssl
    tensors = build(tmp_path/'raw-build', target_k=k)
    args = ssl_test_args(tensors, tmp_path, monkeypatch)
    args.min_span, args.max_span, args.mask_ratio = minimum, maximum, ratio
    calls = []
    def forbidden_model(*a, **kw):
        calls.append(True); raise AssertionError('unsupported mask reached model construction')
    monkeypatch.setattr(train_ssl, 'SSLEncoder', forbidden_model)
    with pytest.raises(ValueError, match='mask|span') as exc:
        train_ssl.run_tensor_mbm(args)
    if k in (16, 64):
        assert 'unsupported' in str(exc.value).lower()
        assert 'infeasible' not in str(exc.value).lower()
    assert not calls and not (tmp_path/'model').exists()


def test_up_l2a_short_k_is_feasible_but_default_prefix_can_fail():
    import tensorflow as tf
    from src.engine.ssl_trainer import random_span_mask, span_mask_from_starts
    valid = span_mask_from_starts(tf.constant([[13]]), 5, 16, tf.zeros((1,16), tf.int32))
    assert int(tf.reduce_sum(valid)) == 3
    bad = random_span_mask(2, 16, .2, 5, 15, tf.zeros((2,16), tf.int32), seed=tf.constant([42,0]))
    assert bad.numpy().sum((1,2)).tolist() == [3., 9.]
    masks = [random_span_mask(16, 64, .2, 5, 15, tf.zeros((16,64), tf.int32),
                             seed=tf.constant([42,s])).numpy() for s in range(16)]
    fractions = np.concatenate(masks).mean((1,2))
    assert np.any((fractions < .15) | (fractions > .30)), 'K64 counterexample must stay covered'


@pytest.fixture(scope='module')
def actual_short_run(tmp_path_factory):
    """Defense-in-depth negative: bypass ONLY producer preflight, not MBM.

    The unchanged trainer really executes K16/2 epochs. No metric/history is
    invented. Its output must never receive an anchor after the post-run gate.
    """
    from scripts import train_ssl
    root = tmp_path_factory.mktemp('negative-actual-short-run')
    tensors = build(root/'raw-build', target_k=16)
    with pytest.MonkeyPatch.context() as mp:
        args = ssl_test_args(tensors, root, mp); args.epochs = 2
        mp.setattr(train_ssl, 'validate_p2_mask_configuration', lambda *a: None)
        try:
            train_ssl.run_tensor_mbm(args)
        except ValueError as exc:
            assert 'mask' in str(exc).lower() and '15' in str(exc)
    history = json.loads((root/'logs/ssl_history.json').read_text())
    assert any(e['val']['mask_fraction'] > .30 for e in history['epochs'])
    return root


def test_up_l2a_post_producer_refuses_real_out_of_budget_history(actual_short_run):
    assert not (actual_short_run/'model/ssl_anchor_contract.json').exists(), 'out-of-budget real run was certified'
    assert len(json.loads((actual_short_run/'logs/ssl_history.json').read_text())['epochs']) == 2


def test_up_l2a_validator_refuses_real_out_of_budget_history(native, actual_short_run):
    from src.utils import selection_manifest as sm
    from scripts.run_full_pipeline import validate_ssl_history_schema
    history_path = native/'ssl_history.json'
    history_path.write_bytes((actual_short_run/'logs/ssl_history.json').read_bytes())
    actual = json.loads(history_path.read_text())
    with pytest.raises(RuntimeError, match='mask fraction outside'):
        validate_ssl_history_schema(actual, expected_consistency_weight=0.)
    c = json.loads((native/'ssl_anchor_contract.json').read_text())
    c['history'], c['history_ref'] = actual, sm.p2_file_ref(history_path)
    c['best_epoch'] = [e['epoch'] for e in actual['epochs'] if e['improved']][-1]
    # Destructive negative contract only; no archive origin is altered/resigned.
    with pytest.raises(ValueError, match='mask.*15'):
        sm._p2_ssl_training_evidence(c, native)


@pytest.mark.parametrize('damage', ['revived', 'missing'])
def test_up_l2a_validator_keeps_pipeline_post_adaptation_gate(native, damage):
    from src.utils import selection_manifest as sm
    c = json.loads((native/'ssl_anchor_contract.json').read_text())
    if damage == 'revived': c['history']['epochs'][0]['post_adaptation_weights']['symmetry'] = .01
    else: del c['history']['epochs'][0]['post_adaptation_weights']
    path = native/'ssl_history.json'; path.write_text(json.dumps(c['history']))
    c['history_ref'] = sm.p2_file_ref(path)
    with pytest.raises(ValueError, match='consistency|adaptation'):
        sm._p2_ssl_training_evidence(c, native)


@pytest.mark.parametrize('damage', ['short_k', 'span', 'ratio', 'missing_consistency'])
def test_up_l2a_validator_requires_supported_mask_configuration(native, damage):
    from src.utils import selection_manifest as sm
    c = json.loads((native/'ssl_anchor_contract.json').read_text())
    path = native/'ssl_run_config.json'; config = json.loads(path.read_text())
    if damage == 'short_k': c['feature_schema']['seq_len'] = 64
    if damage == 'span': config['max_span'] = 30
    if damage == 'ratio': config['mask_ratio'] = .9
    if damage == 'missing_consistency': del config['consistency_weight']
    path.write_text(json.dumps(config)); c['config_ref'] = sm.p2_file_ref(path)
    with pytest.raises(ValueError, match='mask|span|consistency'):
        sm._p2_ssl_training_evidence(c, native)
