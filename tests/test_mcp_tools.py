"""MCP service contract tests; no historical research assets are opened."""
import asyncio
import base64
import json
from pathlib import Path
import pytest


def synthetic_numeric():
    return {'energies_eV': [[-2., -1., -2.], [2., 1., 2.]],
            'k_distance': [0., .5, 1.], 'segment_ids': [0, 0, 0],
            'fermi_eV': 0., 'k_unit': 'relative'}


@pytest.fixture(scope='module')
def native_stdio(tmp_path_factory):
    """Send untouched tool arguments to the real SDK1 server over stdio."""
    import os
    import queue
    import subprocess
    import sys
    import threading
    root = Path(__file__).resolve().parents[1]
    out = tmp_path_factory.mktemp('native-boundary')
    messages = queue.Queue()
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               CUDA_VISIBLE_DEVICES='-1')
    with (out / 'stderr.txt').open('wb') as errlog, (out / 'stdout.jsonl').open('wb') as received, \
            (out / 'stdin.jsonl').open('wb') as sent:
        process = subprocess.Popen([sys.executable, '-B', str(root / 'mcp_server/server.py')],
            cwd=root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errlog)
        def read():
            for line in process.stdout:
                received.write(line); received.flush()
                messages.put(json.loads(line))
        thread = threading.Thread(target=read, daemon=True)
        thread.start()
        counter = 0
        def request(method, params):
            nonlocal counter
            counter += 1
            raw = (json.dumps({'jsonrpc': '2.0', 'id': counter, 'method': method,
                               'params': params}, allow_nan=False) + '\n').encode()
            sent.write(raw); sent.flush()
            process.stdin.write(raw); process.stdin.flush()
            response = messages.get(timeout=65)
            assert response.get('id') == counter, response
            return response
        try:
            initialized = request('initialize', {'protocolVersion': '2025-06-18', 'capabilities': {},
                                  'clientInfo': {'name': 'boundary-regression', 'version': '1'}})
            assert initialized['result']['serverInfo']['name'] == 'BandStructure'
            initialized_notice = b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n'
            sent.write(initialized_notice); sent.flush()
            process.stdin.write(initialized_notice); process.stdin.flush()
            yield lambda name, arguments: request('tools/call', {'name': name, 'arguments': arguments})
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
                pytest.fail('server did not shut down after stdin EOF')
            thread.join(timeout=5)
            assert process.returncode == 0


@pytest.mark.parametrize('mutation', ['duplicate', 'nested_duplicate', 'NaN', 'Infinity', '-Infinity',
                                      '1e400', '-1e400', 'oversize'])
def test_native_stdio_rejects_invalid_strict_numerical_json(native_stdio, mutation):
    text = json.dumps(synthetic_numeric())
    if mutation == 'duplicate':
        bad = text.replace('"fermi_eV": 0.0', '"fermi_eV": 0.0, "fermi_eV": 7.0')
    elif mutation == 'nested_duplicate':
        bad = text[:-1] + ', "metadata": {"x": 0, "x": 1}}'
    elif mutation == 'oversize':
        bad = text + ' ' * (10 * 1024 * 1024)
    else:
        bad = text[:-1] + ', "metadata": ' + mutation + '}'
    reply = native_stdio('analyze_band_structure', {'input_type': 'numerical_data', 'data': bad})
    result = reply['result']['structuredContent']
    assert result['status'] == 'refused', result
    assert result['error_code'] == 'INVALID_NUMERICAL_JSON'
    assert result['confidence'] is None and result['ood_flag'] is None
    assert 'line_mode_gap_eV' not in result
    good = native_stdio('analyze_band_structure', {'input_type': 'numerical_data', 'data': text})
    assert good['result']['structuredContent']['line_mode_gap_eV'] == 2.


@pytest.mark.parametrize('budget', [True, False, '1', 1.0, 1.5])
def test_native_stdio_page_budget_is_a_strict_integer(native_stdio, budget):
    import pymupdf
    with pymupdf.open() as doc:
        doc.new_page().insert_text((30, 40), 'Synthetic page budget')
        payload = base64.b64encode(doc.tobytes()).decode('ascii')
    reply = native_stdio('extract_document', {'payload_base64': payload, 'kind': 'pdf', 'max_pages': budget})
    assert reply['result'].get('isError') is True, reply
    assert not reply['result'].get('structuredContent')
    good = native_stdio('extract_document', {'payload_base64': payload, 'kind': 'pdf', 'max_pages': 1})
    assert good['result']['structuredContent']['status'] == 'ok'


@pytest.mark.parametrize('tool,field', [('validate_physics', 'band_data'),
                                       ('analyze_visual_observations', 'observations'),
                                       ('request_missing_evidence', 'observations'),
                                       ('prepare_human_audit_candidate', 'observations'),
                                       ('extract_band_from_image', 'calibration'),
                                       ('extract_band_from_image', 'annotations')])
def test_native_stdio_object_fields_never_undergo_lossy_json_coercion(native_stdio, tool, field):
    if field == 'band_data':
        text = json.dumps(synthetic_numeric())
        text = text.replace('"fermi_eV": 0.0', '"fermi_eV": 7.0, "fermi_eV": 0.0')
        arguments = {field: text}
    elif field == 'observations':
        arguments = {field: '{"source":{"page_number":1,"page_number":7}}'}
    else:
        examples = Path(__file__).resolve().parents[1] / 'mcp_server/examples'
        manual = json.loads((examples / 'synthetic_band_calibration.json').read_text())
        arguments = {'payload_base64': base64.b64encode((examples / 'synthetic_band.png').read_bytes()).decode('ascii'), **manual}
        text = json.dumps(arguments[field])
        duplicate = ', "x_values": [0,1]}' if field == 'calibration' else ', "fermi_y": 240}'
        arguments[field] = text[:-1] + duplicate
    reply = native_stdio(tool, arguments)
    assert reply['result'].get('isError') is True, reply
    assert not reply['result'].get('structuredContent')


@pytest.mark.parametrize('tool', ['analyze_band_structure', 'extract_band_from_image'])
@pytest.mark.parametrize('axis,values', [('x_values', [False, True]), ('y_values', [True, False])])
def test_native_stdio_boolean_calibration_cannot_produce_physical_values(native_stdio, tool, axis, values):
    examples = Path(__file__).resolve().parents[1] / 'mcp_server/examples'
    manual = json.loads((examples / 'synthetic_band_calibration.json').read_text())
    image = base64.b64encode((examples / 'synthetic_band.png').read_bytes()).decode('ascii')
    manual['calibration'][axis] = values
    arguments = ({'input_type': 'image', 'data': image} if tool == 'analyze_band_structure'
                 else {'payload_base64': image})
    reply = native_stdio(tool, {**arguments, **manual})
    result = reply['result']['structuredContent']
    assert result['status'] == 'refused', result
    assert result['error_code'] == 'INVALID_IMAGE_OR_CALIBRATION'
    assert result['confidence'] is None and result['ood_flag'] is None
    assert not result.get('band_data') and 'line_mode_gap_eV' not in result


def test_service_status_distinguishes_delivered_and_blocked_capabilities():
    from mcp_server.server import get_service_status
    result = get_service_status()
    assert result['status'] == 'ok'
    from mcp_server.version import __version__
    assert result['delivery_version'] == __version__
    assert result['confidence'] is None and result['ood_flag'] is None
    assert result['capabilities']['analytic_line_mode'] == 'available'
    assert result['capabilities']['ai_native_observation_planning'] == 'available'
    assert result['capabilities']['ai_native_observation_analysis'] == 'available_unverified'
    assert result['primary_perception_path'] == 'ai_client_native_document_vision_ocr'
    assert result['capabilities']['document_ocr'] == 'optional_local_rapidocr_fallback'
    assert result['capabilities']['upload_worker_isolation'] == (
        'available_subprocess_timeout_minimal_environment')
    assert result['capabilities']['pending_review_workbench'] == (
        'available_nonapproving_ai_assisted')
    assert result['capabilities']['ann_retrieval'] == 'blocked_not_registered_unverified_artifacts'
    assert result['capabilities']['calibrated_uncertainty'] == 'blocked_no_calibration'
    assert result['historical_assets_loaded'] is False


def test_numerical_measurement_is_not_a_prediction():
    from mcp_server.server import analyze_band_structure
    result = analyze_band_structure('numerical_data', synthetic_numeric())
    assert result['analysis_kind'] == 'analytic_measurement'
    assert result['status'] == 'ok'
    assert result['line_mode_gap_eV'] == 2.
    assert result['line_mode_topology'] == 'direct'
    assert result['confidence'] is None and result['uncertainty_interval'] is None
    assert result['provider_global_electronic_type'] is None
    assert result['line_global_disagreement'] is None


def test_strict_json_accepts_valid_data_but_refuses_duplicates_constants_and_wrapping():
    from mcp_server.server import analyze_band_structure
    text = json.dumps(synthetic_numeric())
    assert analyze_band_structure('numerical_data', text)['line_mode_gap_eV'] == 2.
    bad = [text.replace('"fermi_eV": 0.0', '"fermi_eV": 9,"fermi_eV": 0.0'),
           text.replace('"fermi_eV": 0.0', '"fermi_eV": NaN'),
           json.dumps(text), 'x' * (10 * 1024 * 1024 + 1), 'C:/private/band.json']
    for item in bad:
        result = analyze_band_structure('numerical_data', item)
        assert result['status'] == 'refused'
        assert result['confidence'] is None and result['ood_flag'] is None


def test_non_numeric_mode_never_silently_measures_numeric_payload():
    from mcp_server.server import analyze_band_structure
    for mode in ['structure', 'material_id', 'image']:
        result = analyze_band_structure(mode, synthetic_numeric())
        assert result['status'] in ['refused', 'unavailable']
        assert result.get('line_mode_gap_eV') is None


def test_mcp_document_ocr_uses_upload_and_returns_errors_without_file_access():
    from mcp_server.server import extract_document
    from tests.test_mcp_documents import synthetic_text_png
    payload = base64.b64encode(synthetic_text_png()).decode('ascii')
    result = extract_document(payload, 'image', 1)
    assert result['status'] == 'ok' and 'Energy' in result['text']
    assert result['calibration_verified'] is False
    for bad in ['C:/private/file.pdf', '%%%%', 'A' * (14 * 1024 * 1024 + 1)]:
        denied = extract_document(bad, 'pdf', 1)
        assert denied['status'] == 'refused' and denied['confidence'] is None


def test_image_to_measurement_requires_explicit_calibration_and_preserves_audit_status():
    from mcp_server.server import extract_band_from_image, analyze_band_structure
    from tests.test_mcp_documents import synthetic_band_png, synthetic_annotations
    payload = base64.b64encode(synthetic_band_png()).decode('ascii')
    uncalibrated = extract_band_from_image(payload)
    assert uncalibrated['status'] == 'requires_calibration' and uncalibrated['band_data'] is None
    ann, cal = synthetic_annotations()
    result = analyze_band_structure('image', payload, ann, cal)
    assert result['status'] == 'caller_calibrated_unverified'
    assert result['analysis_kind'] == 'image_reconstruction_measurement'
    assert result['human_audited'] is False
    assert result['line_mode_gap_eV'] == pytest.approx(1.5, rel=1e-5, abs=1e-5)
    assert result['confidence'] is None and result['effective_mass_electron_m0'] is None
    assert analyze_band_structure('image', '%%%')['status'] == 'refused'


def test_material_id_reads_only_pinned_public_reference_and_rejects_unknown():
    from mcp_server.server import analyze_band_structure
    result = analyze_band_structure('material_id', 'pymatgen-Cu2O_361')
    assert result['status'] == 'ok'
    assert result['line_mode_gap_eV'] == pytest.approx(.5047, rel=1e-5, abs=1e-5)
    assert result['line_mode_topology'] == 'direct'
    assert result['reference_scope'] == 'public_reference_not_blind_test'
    assert len(result['source']['sha256']) == 64
    for unknown in ['aflow:anything', '../../data/cache', 'pymatgen-Cu2O_361 ']:
        assert analyze_band_structure('material_id', unknown)['status'] == 'unavailable'


def test_physics_validation_reports_limited_consistency_not_universal_physics_pass():
    from mcp_server.server import validate_physics
    result = validate_physics(synthetic_numeric())
    assert result['input_consistent'] is True and result['physics_valid'] is None
    assert result['validation_scope'] == 'sampled_path_consistency_only'
    bad = synthetic_numeric(); bad['k_distance'] = [0., 0., 1.]
    assert validate_physics(bad)['input_consistent'] is False
    assert validate_physics(bad)['status'] == 'refused'


def test_retrieval_refuses_unverified_artifacts_without_claiming_neighbors():
    from mcp_server.server import retrieve_similar_materials
    result = retrieve_similar_materials(synthetic_numeric(), 'band_data', 5)
    assert result['status'] == 'unavailable'
    assert result['results'] == [] and result['confidence'] is None
    assert result['error_code'] == 'RETRIEVAL_NOT_CERTIFIED'
    assert result['historical_assets_loaded'] is False


def test_knowledge_is_cited_and_unknown_system_is_not_fabricated():
    from mcp_server.server import get_material_knowledge
    result = get_material_knowledge('oxide', 'dft_methodology')
    assert result['status'] == 'ok' and result['citations']
    assert result['confidence'] is None
    assert 'global' in result['statements'][0]
    assert get_material_knowledge('imaginary-element-999')['status'] == 'unknown'


def test_application_recommendation_never_invents_device_grade_from_gap():
    from mcp_server.server import recommend_application
    result = recommend_application(synthetic_numeric())
    assert result['status'] == 'insufficient_evidence'
    assert result['application_recommendation'] is None
    assert 'optical_matrix_elements' in result['missing_evidence']
    assert result['measurement']['line_mode_gap_eV'] == 2.


def test_mcp_resources_are_metadata_and_prompts_preserve_unknowns():
    from mcp_server.server import mcp
    async def probe():
        resources = await mcp.list_resources()
        assert {str(r.uri) for r in resources} == {
            'band://model_card', 'band://physics_constraints', 'band://band_database',
            'band://human_audit_protocol', 'band://schemas'}
        parts = await mcp.read_resource('band://model_card')
        card = json.loads(list(parts)[0].content)
        assert card['historical_assets_loaded'] is False
        assert card['confidence'] is None and card['ood_flag'] is None
        prompts = await mcp.list_prompts()
        assert {p.name for p in prompts} == {'band_analysis_guide', 'band_human_audit_guide'}
    asyncio.run(probe())


def test_real_stdio_initialize_list_and_structured_tool_call():
    import sys
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    server = Path(__file__).resolve().parents[1] / 'mcp_server/server.py'
    async def probe():
        params = StdioServerParameters(command=sys.executable, args=['-B', str(server)],
                                      env={'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1'})
        async with stdio_client(params) as streams:
            async with ClientSession(*streams) as session:
                initialized = await session.initialize()
                assert initialized.serverInfo.name == 'BandStructure'
                tools = await session.list_tools()
                assert {x.name for x in tools.tools} == {
                    'get_service_status', 'analyze_band_structure', 'validate_physics',
                    'extract_document', 'extract_band_from_image', 'get_material_knowledge',
                    'plan_band_image_analysis', 'request_missing_evidence',
                    'analyze_visual_observations', 'prepare_human_audit_candidate',
                    'analyze_electronic_data','inspect_attachment','analyze_attachment','export_attachment',
                    'read_result_chunk','validate_axis_calibration','review_material_evidence'}
                result = await session.call_tool('analyze_band_structure',
                    {'input_type': 'numerical_data', 'data': synthetic_numeric()})
                assert not result.isError
                assert result.structuredContent['line_mode_gap_eV'] == 2.
                assert result.structuredContent['confidence'] is None
                malformed = await session.call_tool('analyze_band_structure', {'input_type': 'structure', 'data': {}})
                assert malformed.isError
    asyncio.run(asyncio.wait_for(probe(), timeout=80))


def test_reusable_demo_runs_all_tools_over_real_stdio(tmp_path):
    import sys
    from scripts.demo_mcp_client import run_demo
    observation_path = tmp_path / 'observations.json'
    observation_path.write_text(json.dumps({
        'source': {'document_id': 'test:paper', 'page_number': 7, 'panel_label': '(a)'},
        'energy_unit': 'eV', 'fermi_eV': 0., 'k_unit': 'relative',
        'calibration_evidence': {'energy_tick_count': 2, 'k_anchor_count': 8,
                                 'fermi_reference_observed': True},
        'ambiguities': ['continuous band identity not established'],
    }))
    report = asyncio.run(run_demo(sys.executable, tmp_path / 'demo',
                                  ai_observations_path=observation_path))
    assert report['status'] == 'passed'
    assert len(report['discovered_tools']) == 17
    assert set(report['called_tools']) == set(report['discovered_tools'])
    assert report['cases']['calibrated_image']['line_mode_gap_eV'] == pytest.approx(1.5, rel=1e-5, abs=1e-5)
    assert report['cases']['public_material']['line_mode_gap_eV'] == pytest.approx(.5047, rel=1e-5, abs=1e-5)
    assert report['cases']['ocr']['calibration_verified'] is False
    assert report['cases']['ocr']['worker_isolated'] is True
    assert report['cases']['uncalibrated_image']['worker_isolated'] is True
    assert report['cases']['ai_visual_observations']['line_mode_gap_eV'] == 2.
    assert report['cases']['ai_analysis_plan']['primary_perception_path'] == (
        'ai_client_native_document_vision_ocr')
    assert report['cases']['ai_audit_candidate']['status'] == 'pending_human'
    assert report['cases']['external_ai_missing_evidence']['status'] == 'needs_more_evidence'
    assert report['cases']['external_ai_audit_candidate']['eligible_for_scientific_acceptance'] is False
    assert report['pending_human_queue']['record_count'] == 1
    assert report['pending_human_queue']['eligible_for_scientific_acceptance'] is False
    assert (tmp_path / 'demo/pending_human_candidates.json').is_file()
    assert (tmp_path / 'demo/report.json').is_file()
