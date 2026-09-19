"""Regression of the ACEAMI P0/P1 findings; synthetic cases are not scientific truth."""
import base64
import copy
import hashlib
import io
import json
import pytest
from PIL import Image, ImageDraw


def composite_png():
    im=Image.new('RGB',(660,240),'white');d=ImageDraw.Draw(im)
    for x in (30,250,470):
        d.rectangle((x,25,x+160,195),outline='black',width=2)
        d.line([(x+2,170),(x+80,120),(x+158,170)],fill=(0,110,180),width=3)
        d.line([(x+2,70),(x+80,35),(x+158,70)],fill=(255,120,0),width=3)
    b=io.BytesIO();im.save(b,format='PNG');return b.getvalue()


def raster_pdf():
    import pymupdf as f
    with f.open() as d:
        p=d.new_page(width=700,height=400)
        p.insert_text((20,20),'Electronic bands, phonons and mobility: three distinct panels')
        p.insert_image(f.Rect(20,70,680,310),stream=composite_png())
        return d.tobytes()


def test_isolated_pdf_chunks_bind_one_parent_result(monkeypatch):
    from mcp_server import server
    monkeypatch.setenv('BAND_MCP_UPLOAD_ISOLATION','1')
    monkeypatch.delenv('BAND_MCP_WORKER',raising=False)
    monkeypatch.setattr(server,'_PDF_INLINE_CHARS',1)
    monkeypatch.setattr(server,'_PDF_CHUNK_CHARS',4096)
    args={'payload_base64':base64.b64encode(raster_pdf()).decode(),'kind':'pdf'}
    first=server.extract_band_from_image(**args)
    assert first['status']=='paged_result'
    chunks=[first['payload_chunk']];offset=first['next_offset']
    while offset is not None:
        part=server.extract_band_from_image(**args,pdf_result_offset=offset,pdf_result_sha256=first['payload_sha256'])
        assert part['status']=='paged_result',part
        assert part['payload_sha256']==first['payload_sha256']
        chunks.append(part['payload_chunk']);offset=part['next_offset']
    assert hashlib.sha256(''.join(chunks).encode('ascii')).hexdigest()==first['payload_sha256']


def test_raster_pdf_routes_to_host_selection_instead_of_silent_empty():
    from src.vision.multi_format_parser import digitize_pdf_band_panels
    out=digitize_pdf_band_panels(raster_pdf())
    assert out['status']=='requires_raster_review'
    assert out['needs_ocr_review'] is True
    assert out['carrier_kind']=='raster'
    assert out['raster_regions'] and out['next_action']
    assert out['band_data'] is None


def test_composite_returns_separate_candidates_not_global_frame():
    from mcp_server.server import extract_band_from_image
    out=extract_band_from_image(base64.b64encode(composite_png()).decode())
    assert out['status']=='requires_panel_selection'
    assert out['panel_bbox'] is None
    assert len(out['panel_candidates'])==3
    assert all(c['panel_kind']=='unknown' for c in out['panel_candidates'])


def test_selected_raster_is_source_bound_and_exports_all_visible_ink():
    from mcp_server.server import extract_band_from_image
    raw=composite_png();args={'payload_base64':base64.b64encode(raw).decode(),
        'panel_selection':{'source_sha256':hashlib.sha256(raw).hexdigest(),'page_number':1,
        'coordinate_unit':'pixel','panel_bbox_xyxy':[30,25,190,195],'panel_kind':'electronic_band'}}
    out=extract_band_from_image(**args)
    assert out['panel_bbox']==[30,25,190,195]
    assert out['visible_ink']['point_count']>0 and out['visible_ink']['svg']
    assert out['visible_ink']['band_identity_resolved'] is False
    assert out['source_sha256']==hashlib.sha256(raw).hexdigest()
    bad=copy.deepcopy(args);bad['panel_selection']['source_sha256']='0'*64
    assert extract_band_from_image(**bad)['status']=='refused'
    bad=copy.deepcopy(args);bad['panel_selection']['panel_kind']='phonon'
    assert extract_band_from_image(**bad)['status']=='non_electronic_panel'


def v2():
    return {'schema_version':2,'source':{'document_id':'synthetic:v2','page_number':1,'panel_label':'a'},
      'energy_unit':'eV','energy_reference':{'kind':'vbm','value_eV':0.,'observed':True,'evidence':'VBM = 0 label'},
      'k_unit':'relative','k_distance':[0,.5,1,1.2,1.5,2], 'segment_ids':[0,0,0,1,1,1],
      'bands':[{'band_id':'v1','role':'valence','energies_eV':[-1,0,-1,-1,0,-1]},
               {'band_id':'v2','role':'valence','energies_eV':[-2,-1,-2,-2,-1,-2]},
               {'band_id':'c1','role':'conduction','energies_eV':[4,3,4,4,3,4]}],
      'calibration_evidence':{'energy_tick_count':3,'k_anchor_count':4},'ambiguities':[]}


def test_vbm_reference_does_not_fabricate_fermi_and_preserves_multiband():
    from mcp_server.server import analyze_visual_observations
    o=v2();before=copy.deepcopy(o);r=analyze_visual_observations(o)
    assert r['status']=='ai_observation_unverified',r
    assert r['line_mode_gap_eV']==3 and r['fermi_eV'] is None
    assert r['full_numerical_band_reconstruction'] is False
    assert r['observation_summary']['band_count']==3
    assert r['exports']['csv_text'] and r['exports']['svg']
    assert r['effective_mass_electron_m0'] is None and r['human_audited'] is False
    assert o==before


def test_missing_samples_export_without_interpolation_or_certified_gap():
    from mcp_server.server import analyze_visual_observations
    o=v2();o['bands'][2]['energies_eV'][2]=None
    r=analyze_visual_observations(o)
    assert r['status']=='partial_observations'
    assert r['line_mode_gap_eV'] is None
    assert r['sampled_edge_separation_eV']==3
    assert r['coverage']['missing_samples']==1
    assert ',missing' in r['exports']['csv_text']


@pytest.mark.parametrize('change',[
    lambda o:o['energy_reference'].update(observed=False),
    lambda o:o['bands'][0]['energies_eV'].__setitem__(0,True),
    lambda o:o.update(segment_ids=[0,0,1,1,0,0]),
    lambda o:o['energy_reference'].update(kind='fermi'),
    lambda o:o['bands'][1].update(band_id='v1'),
])
def test_v2_bad_or_unobserved_inputs_never_produce_gap(change):
    from mcp_server.server import analyze_visual_observations
    o=v2();change(o);r=analyze_visual_observations(o)
    assert r['status'] in {'refused','needs_more_evidence'}
    assert r.get('line_mode_gap_eV') is None


def test_status_exposes_loaded_build_and_contract_version():
    from mcp_server.server import get_service_status
    s=get_service_status()
    assert len(s['build_sha256'])==64 and s['started_utc']
    assert s['observation_schema_versions']==[1,2]
    from mcp_server.version import __version__
    assert s['delivery_version']==__version__


def test_parent_cache_expiry_wrong_sha_interleaving_eviction():
    from mcp_server.pdf_result_cache import PDFResultCache,ResultCacheError
    now=[0.];calls=[]
    c=PDFResultCache(ttl=10,max_entries=2,max_bytes=100,clock=lambda:now[0])
    def factory(k):
        calls.append(k);return '{"key":"'+k+'"}'
    a=c.get('a',lambda:factory('a'));sa=hashlib.sha256(a.encode()).hexdigest()
    c.get('b',lambda:factory('b'))
    assert c.get('a',lambda:factory('wrong'),continuation=True,expected_sha=sa)==a
    assert calls==['a','b']
    with pytest.raises(ResultCacheError,match='VERSION_CHANGED'):
        c.get('a',lambda:factory('wrong'),continuation=True,expected_sha='0'*64)
    c.get('c',lambda:factory('c'))
    with pytest.raises(ResultCacheError,match='EXPIRED'):c.get('b',lambda:factory('wrong'),continuation=True)
    now[0]=11
    with pytest.raises(ResultCacheError,match='EXPIRED'):c.get('a',lambda:factory('wrong'),continuation=True,expected_sha=sa)
    assert calls==['a','b','c']


def test_parent_cache_parallel_first_requests_produce_once():
    from mcp_server.pdf_result_cache import PDFResultCache
    from concurrent.futures import ThreadPoolExecutor
    c=PDFResultCache();calls=[]
    def factory():calls.append(1);return '{}'
    with ThreadPoolExecutor(max_workers=4) as p:result=list(p.map(lambda _:c.get('same',factory),range(8)))
    assert result==['{}']*8 and calls==[1]


def test_parent_cache_oversize_and_factory_failure_are_not_cached():
    from mcp_server.pdf_result_cache import PDFResultCache,ResultCacheError
    c=PDFResultCache(max_bytes=5)
    with pytest.raises(ResultCacheError,match='TOO_LARGE'):c.get('x',lambda:'x'*6)
    with pytest.raises(ResultCacheError,match='EXPIRED'):c.get('x',lambda:'{}',continuation=True)
    assert c.get('x',lambda:'{}')=='{}'


def test_source_bound_pdf_roi_with_other_kind_never_analyzes_phonons():
    from mcp_server.server import extract_band_from_image
    raw=raster_pdf();selection={'source_sha256':hashlib.sha256(raw).hexdigest(),'page_number':1,
        'coordinate_unit':'pdf_point','panel_bbox_xyxy':[40,90,220,280],'panel_kind':'phonon'}
    r=extract_band_from_image(base64.b64encode(raw).decode(),kind='pdf',panel_selection=selection)
    assert r['status']=='non_electronic_panel' and r['band_data'] is None
    selection['panel_kind']='electronic_band'
    r=extract_band_from_image(base64.b64encode(raw).decode(),kind='pdf',panel_selection=selection)
    assert r['status']=='requires_calibration' and r['crop_png_base64']
    assert r['crop_pixel_to_pdf_point']['scale']==.4


def test_axis_ocr_flags_keep_raw_text_and_never_auto_correct():
    from src.vision.axis_ocr_review import review_axis_tokens
    r=review_axis_tokens([{'text':'O','ocr_score':.9,'box':[]},{'text':'XIW','ocr_score':.99,'box':[]}])
    assert r['corrected_text'] is None and all(not x['auto_applied'] for x in r['flagged_regions'])
    assert r['flagged_regions'][0]['raw_text']=='O'
    assert r['flagged_regions'][1]['proposed_tokens']==['X|W']


def test_v2_unknown_reference_and_ambiguities_have_actionable_review():
    from mcp_server.server import analyze_visual_observations,request_missing_evidence,prepare_human_audit_candidate
    o=v2();o['energy_reference']['kind']='arbitrary';o['ambiguities']=['crossing identity unknown']
    assert request_missing_evidence(o)['status']=='needs_more_evidence'
    r=analyze_visual_observations(o)
    assert r['status']=='partial_observations' and r['line_mode_gap_eV'] is None
    assert r['sampled_edge_separation_eV'] is None and r['exports']['svg']
    assert prepare_human_audit_candidate(o)['human_audited'] is False


@pytest.mark.parametrize('field',['kind','k_unit','role'])
def test_v2_unhashable_enums_fail_closed(field):
    from mcp_server.server import analyze_visual_observations
    o=v2()
    if field=='kind':o['energy_reference']['kind']=[]
    elif field=='role':o['bands'][0]['role']=[]
    else:o['k_unit']=[]
    assert analyze_visual_observations(o)['status']=='refused'


def test_v2_segment_labels_and_source_bound_ocr_corrections():
    from mcp_server.server import analyze_visual_observations
    o=v2();o['source']['source_sha256']='a'*64
    o['path_segments']=[{'segment_id':0,'start_label':'Γ','end_label':'X'},
                        {'segment_id':1,'start_label':'W','end_label':'L'}]
    o['ocr_corrections']=[{'raw_text':'O','corrected_text':'0','bbox':[2,3,8,9],
        'evidence':'AI compared original axis crop; no automatic application',
        'reviewer_kind':'ai_client','coordinate_unit':'pixel','source_sha256':'a'*64}]
    r=analyze_visual_observations(o)
    assert r['status']=='ai_observation_unverified'
    assert 'Γ–X' in r['exports']['svg'] and 'W–L' in r['exports']['svg']
    assert r['ocr_corrections']==o['ocr_corrections'] and not r['human_audited']
    o['ocr_corrections'][0]['source_sha256']='b'*64
    assert analyze_visual_observations(o)['status']=='refused'


def test_v2_large_finite_energies_and_singletons_export_finitely():
    from mcp_server.server import analyze_visual_observations
    o=v2()
    for b in o['bands']:b['energies_eV']=[1e308,None,None,None,None,None]
    r=analyze_visual_observations(o)
    assert r['status']=='partial_observations'
    svg=r['exports']['svg'].lower()
    assert 'nan' not in svg and 'inf' not in svg and '<circle' in svg


def test_roi_huge_integer_is_a_controlled_refusal():
    from mcp_server.server import extract_band_from_image
    raw=composite_png()
    r=extract_band_from_image(base64.b64encode(raw).decode(),panel_selection={
        'source_sha256':hashlib.sha256(raw).hexdigest(),'page_number':1,'coordinate_unit':'pixel',
        'panel_bbox_xyxy':[0,0,10**1000,200],'panel_kind':'electronic_band'})
    assert r['status']=='refused'
