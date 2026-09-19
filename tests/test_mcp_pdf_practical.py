"""Practical PDF contracts; synthetic fixtures are not scientific evidence."""
import inspect
import pymupdf
import pytest
from src.vision.multi_format_parser import extract_document_text


def sample_document():
    with pymupdf.open() as doc:
        for i in range(3):
            page=doc.new_page();page.insert_text((50,50),f'PAGE {i+1}')
        return doc.tobytes()


def test_document_returns_selected_page_with_cursor():
    assert 'page_start' in inspect.signature(extract_document_text).parameters
    out=extract_document_text(sample_document(),kind='pdf',max_pages=1,page_start=2)
    assert [p['page'] for p in out['pages']]==[2]
    assert out['pages'][0]['text'].strip()=='PAGE 2'
    assert out['total_pages']==3 and out['next_page']==3


@pytest.mark.parametrize('start',[0,-1,True,False,1.0,'1',4])
def test_document_refuses_invalid_page_start(start):
    with pytest.raises(ValueError):
        extract_document_text(sample_document(),kind='pdf',max_pages=1,page_start=start)


def test_raster_does_not_accept_a_second_page():
    import io
    from PIL import Image
    b=io.BytesIO();Image.new('RGB',(128,128),'white').save(b,format='PNG')
    with pytest.raises(ValueError):
        extract_document_text(b.getvalue(),kind='image',page_start=2)


def test_mcp_publishes_and_dispatches_selected_page():
    import asyncio,base64
    from mcp_server import server
    assert 'page_start' in inspect.signature(server.extract_document).parameters
    tools=asyncio.run(server.mcp.list_tools())
    schema=next(t.inputSchema for t in tools if t.name=='extract_document')
    assert schema['properties']['page_start']['type']=='integer'
    result=server.extract_document(base64.b64encode(sample_document()).decode(),kind='pdf',page_start=2,max_pages=1)
    assert result['status']=='ok' and [p['page'] for p in result['pages']]==[2]


def test_mcp_rejects_boolean_page_start_before_sdk_coercion():
    import asyncio,base64
    from mcp_server import server
    from mcp.server.fastmcp.exceptions import ToolError
    args={'payload_base64':base64.b64encode(sample_document()).decode(),'kind':'pdf','page_start':True}
    with pytest.raises(ToolError):
        asyncio.run(server.mcp.call_tool('extract_document',args))


def test_pdf_inspection_returns_selected_page_geometry_and_preview():
    import base64
    from src.vision import multi_format_parser as module
    assert hasattr(module,'inspect_pdf_band_page')
    out=module.inspect_pdf_band_page(sample_document(),page_number=2)
    assert out['status']=='geometry_only' and out['page_number']==2
    assert out['band_data'] is None and out['confidence'] is None
    assert any(w['text']=='PAGE' for w in out['text_words'])
    assert base64.b64decode(out['preview_png_base64']).startswith(b'\x89PNG')
    assert out['drawing_count']==0


def test_existing_band_tool_exposes_pdf_inspection_without_fake_energies():
    import base64
    from mcp_server import server
    assert 'kind' in inspect.signature(server.extract_band_from_image).parameters
    out=server.extract_band_from_image(base64.b64encode(sample_document()).decode(),kind='pdf',page_number=2)
    assert out['status']=='geometry_only' and out['page_number']==2
    assert out['band_data'] is None and out['calibration_verified'] is False


def test_registered_pdf_adapter_returns_digitizer_not_only_preview():
    import base64
    from mcp_server import server
    from tests.test_mcp_pdf_digitization import calibrated_pdf
    out=server.extract_band_from_image(base64.b64encode(calibrated_pdf(omit='ticks')).decode(),kind='pdf')
    assert 'panels' in out, 'PDF adapter must dispatch the real digitizer'
    assert len(out['panels'])==1 and out['panels'][0]['traces']
    assert out['panels'][0]['energy_mapping'] is None and out['band_data'] is None
    assert out['csv_text'] and out['clean_svg'] and out['preview_png_base64']
    assert out['confidence'] is None and out['ood_flag'] is None


def test_selected_source_pdf_keeps_vectors_not_a_reconstruction():
    import base64
    from src.vision.multi_format_parser import inspect_pdf_band_page
    with pymupdf.open(stream=sample_document(),filetype='pdf') as doc:
        doc[1].draw_line((50,80),(150,130),color=(1,0,0))
        doc[1].insert_link({'kind':pymupdf.LINK_URI,'from':pymupdf.Rect(50,40,150,55),'uri':'https://example.invalid/'})
        out=inspect_pdf_band_page(doc.tobytes(),page_number=2)
    assert 'source_page_pdf_base64' in out
    with pymupdf.open(stream=base64.b64decode(out['source_page_pdf_base64']),filetype='pdf') as exported:
        assert len(exported)==1 and exported[0].get_text().strip()=='PAGE 2'
        assert len(exported[0].get_drawings())==1 and not exported[0].get_images()
        assert not exported[0].get_links()
    assert out['source_page_pdf_role']=='original_graphics_copy_not_reconstruction'


def test_fragment_svg_frames_measured_panels_without_page_whitespace():
    import xml.etree.ElementTree as ET
    from src.vision.multi_format_parser import digitize_pdf_band_panels
    from tests.test_mcp_pdf_digitization import calibrated_pdf
    out=digitize_pdf_band_panels(calibrated_pdf())
    # XML is generated by our bounded exporter from this synthetic fixture, not supplied markup.
    root=ET.fromstring(out['clean_svg'])
    x,y,w,h=map(float,root.attrib['viewBox'].split())
    assert 0<x<80 and 0<y<80 and w<400 and h<400
    for panel in out['panels']:
        for trace in panel['traces']:
            assert all(x<=px<=x+w and y<=py<=y+h for px,py in trace['points_pdf'])


def test_service_status_names_pdf_fragments_without_certifying_physics():
    from mcp_server import server
    out=server.get_service_status()
    assert out['capabilities'].get('pdf_vector_digitization')=='unverified_visible_fragments'
    assert out['capabilities']['calibrated_uncertainty']=='blocked_no_calibration'
    assert out['confidence'] is None and out['ood_flag'] is None


def test_large_pdf_protocol_payload_roundtrips_bounded_verified_chunks():
    import hashlib,json
    from mcp_server import server
    assert hasattr(server,'_pdf_chunk_response'), 'Missing bounded response transport'
    original={'status':'geometry_only','source_sha256':'a'*64,'text':'曲线\\"'*150000}
    serialized=json.dumps(original,ensure_ascii=True,separators=(',',':'),allow_nan=False)
    offset=0;digest=None;parts=[]
    while offset is not None:
        chunk=server._pdf_chunk_response(serialized,offset,digest)
        assert chunk['status']=='paged_result' and chunk['chunk_offset']==(sum(map(len,parts)))
        assert len(json.dumps(chunk,ensure_ascii=True,indent=2))<1_100_000
        digest=digest or chunk['payload_sha256'];assert digest==chunk['payload_sha256']
        parts.append(chunk['payload_chunk']);offset=chunk['next_offset']
    joined=''.join(parts)
    assert hashlib.sha256(joined.encode('ascii')).hexdigest()==digest
    assert json.loads(joined)==original


@pytest.mark.parametrize('offset,sha_value',[
    (-1,None),(True,None),(False,None),(1.0,None),('0',None),(1,None),
    (500000,None),(500000,'b'*64),(0,'bad'),(0,False),(1000000,'MATCH')])
def test_pdf_chunks_refuse_invalid_or_unbound_cursor(offset,sha_value):
    import hashlib,json
    from mcp_server import server
    blob=json.dumps({'status':'geometry_only','text':'x'*800000},separators=(',',':'))
    expected=hashlib.sha256(blob.encode()).hexdigest() if sha_value=='MATCH' else sha_value
    with pytest.raises(ValueError):
        server._pdf_chunk_response(blob,offset,expected)


def test_small_pdf_protocol_payload_stays_inline():
    import json
    from mcp_server import server
    obj={'status':'geometry_only','text':'文字','confidence':None,'ood_flag':None}
    assert server._pdf_chunk_response(json.dumps(obj),0,None)==obj


def test_pdf_tool_continuations_bind_one_stable_uploaded_result(monkeypatch):
    import base64,hashlib,json
    from mcp_server import server
    from tests.test_mcp_pdf_digitization import calibrated_pdf
    assert 'pdf_result_offset' in inspect.signature(server.extract_band_from_image).parameters
    monkeypatch.setattr(server,'_PDF_INLINE_CHARS',1)
    monkeypatch.setattr(server,'_PDF_CHUNK_CHARS',4096)
    args={'payload_base64':base64.b64encode(calibrated_pdf()).decode(),'kind':'pdf','page_number':1}
    first=server.extract_band_from_image(**args)
    assert first['status']=='paged_result'
    again=server.extract_band_from_image(**args)
    assert again==first, 'OCR timing and PDF IDs must not mutate an active result'
    parts=[];current=first
    while True:
        parts.append(current['payload_chunk'])
        if current['next_offset'] is None:break
        current=server.extract_band_from_image(**args,pdf_result_offset=current['next_offset'],
                                              pdf_result_sha256=first['payload_sha256'])
        assert current['payload_sha256']==first['payload_sha256']
    joined=''.join(parts)
    assert hashlib.sha256(joined.encode('ascii')).hexdigest()==first['payload_sha256']
    restored=json.loads(joined)
    assert restored['source_sha256']==hashlib.sha256(base64.b64decode(args['payload_base64'])).hexdigest()
    assert len(restored['panels'])==1  # The established result contract uses a list, not panel_count.


def test_pdf_serialization_budget_prevents_oversized_cache_entry(monkeypatch):
    import base64,json
    from mcp_server import server
    assert hasattr(server,'_PDF_RESULT_JSON_CHARS'), 'No assembled result size budget'
    monkeypatch.setattr(server,'_PDF_RESULT_JSON_CHARS',100)
    server._cached_pdf_payload.cache_clear()
    out=server.extract_band_from_image(base64.b64encode(sample_document()).decode(),kind='pdf')
    assert out['status']=='refused' and server._cached_pdf_payload.cache_info().currsize==0
    with pytest.raises(ValueError,match='budget'):
        server._pdf_chunk_response(json.dumps({'text':'x'*101}),0,None)


@pytest.mark.parametrize('cursor',[{'pdf_result_offset':-1},{'pdf_result_offset':500000},
                                   {'pdf_result_sha256':'bad'}])
def test_bad_pdf_cursor_is_rejected_before_upload_decode(monkeypatch,cursor):
    from mcp_server import server
    def forbidden_decode(_):
        raise AssertionError('invalid cursor reached upload decode')
    monkeypatch.setattr(server,'_decode_upload',forbidden_decode)
    out=server.extract_band_from_image('not_decoded',kind='pdf',**cursor)
    assert out['status']=='refused'


def dense_text_document():
    # Same legal dense text commands as the independent PDF-DOC-001 generator.
    with pymupdf.open() as doc:
        op=b'q\nBT\n/helv 5 Tf\n1 0 0 1 40 750 Tm\n('+b'A'*100+b') Tj\nET\nQ\n'
        for _ in range(5):
            page=doc.new_page();page.insert_text((40,50),'setup',fontname='helv',fontsize=5)
            doc.update_stream(page.get_contents()[0],op*2200)
        return doc.tobytes(garbage=4,deflate=True)


def test_dense_document_response_is_bounded_or_actionably_refused():
    import base64,json,os
    from pathlib import Path
    from mcp_server import server
    frozen=os.environ.get('LIMITED_FIX_FIXTURES')
    payload=(Path(frozen)/'doc/fixtures/dense_text_5pages.pdf').read_bytes() if frozen else dense_text_document()
    encoded=base64.b64encode(payload).decode()
    out=server.extract_document(encoded,kind='pdf',max_pages=5,page_start=1)
    assert len(json.dumps(out,ensure_ascii=True,indent=2)) <= 750000, 'PDF-DOC-001: native reply can be truncated'
    if out['status']=='refused':
        assert out['error_code']=='DOCUMENT_RESPONSE_TOO_LARGE'
        assert out['retry']=={'max_pages':1,'page_start':1}
        assert out['partial_result_returned'] is False
    else:
        assert out['status']=='ok' and len(out['pages'])==5
    pages=[]
    for start in range(1,6):
        small=server.extract_document(encoded,kind='pdf',max_pages=1,page_start=start)
        assert small['status']=='ok' and [p['page'] for p in small['pages']]==[start]
        assert len(json.dumps(small,ensure_ascii=True,indent=2)) <= 750000
        assert small['next_page']==(start+1 if start<5 else None)
        pages.extend(small['pages'])
    assert len(pages)==5 and all(len(p['text'])>200000 for p in pages)
