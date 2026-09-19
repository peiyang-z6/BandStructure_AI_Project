"""Real stdio P0/P1 regression on a supplied paper; never rewrites old receipts."""
from pathlib import Path
import argparse
import asyncio
import base64
import hashlib
import json
import sys
import time
import pymupdf
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT=Path(__file__).resolve().parents[1]


async def run(paper,out):
    out.mkdir(parents=True,exist_ok=False)
    raw=paper.read_bytes();sha=hashlib.sha256(raw).hexdigest();encoded=base64.b64encode(raw).decode()
    if sha!='94b8896d0d54da69caaf6f4b408010ca93612a49b5066c3ce11c671562223fe3':
        raise ValueError('This real-paper regression uses the inspected ACEAMI snapshot only.')
    report={'source_sha256':sha,'cases':{},'pages':[],'scientific_acceptance':False}
    def save(name,obj):(out/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    with pymupdf.open(stream=raw,filetype='pdf') as d:
        png=pymupdf.Pixmap(d,289).tobytes('png')
    image=base64.b64encode(png).decode();image_sha=hashlib.sha256(png).hexdigest()
    params=StdioServerParameters(command=sys.executable,args=['-B',str(ROOT/'mcp_server/server.py')],
      env={'PYTHONDONTWRITEBYTECODE':'1','BAND_MCP_UPLOAD_ISOLATION':'1','CUDA_VISIBLE_DEVICES':'-1',
           'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    async with stdio_client(params,errlog=(out/'stderr.txt').open('w',encoding='utf-8')) as streams:
        async with ClientSession(*streams) as session:
            report['initialize']=(await session.initialize()).model_dump(mode='json',by_alias=True)
            report['tool_schema']=(await session.list_tools()).model_dump(mode='json',by_alias=True)
            async def call(case,tool,args):
                t=time.monotonic();wire=(await asyncio.wait_for(session.call_tool(tool,args),timeout=120)).model_dump(mode='json',by_alias=True)
                save(case+'.json',wire);r=wire['structuredContent']
                report['cases'][case]={'tool':tool,'status':r.get('status'),'isError':wire.get('isError'),
                                      'seconds':round(time.monotonic()-t,3)}
                save('report.json',report)
                print(case,r.get('status'),flush=True)
                assert not wire['isError'],wire
                return r
            status=await call('status','get_service_status',{})
            assert status['delivery_version'].startswith('0.6.')
            report['build_sha256']=status['build_sha256']
            await call('plan','plan_band_image_analysis',{})
            start=1
            while start:
                r=await call(f'pages_{start}','extract_document',{'payload_base64':encoded,'kind':'pdf','max_pages':5,'page_start':start})
                report['pages'].extend(p['page'] for p in r['pages']);start=r['next_page']
            assert report['pages']==list(range(1,8))
            args={'payload_base64':encoded,'kind':'pdf','page_number':2}
            first=await call('pdf_first','extract_band_from_image',args)
            assert first['status']=='paged_result'
            chunks=[first['payload_chunk']];offset=first['next_offset']
            while offset is not None:
                r=await call(f'pdf_{offset}','extract_band_from_image',{**args,'pdf_result_offset':offset,'pdf_result_sha256':first['payload_sha256']})
                assert r['status']=='paged_result' and r['payload_sha256']==first['payload_sha256']
                chunks.append(r['payload_chunk']);offset=r['next_offset']
            combined=''.join(chunks)
            assert hashlib.sha256(combined.encode('ascii')).hexdigest()==first['payload_sha256']
            decoded=json.loads(combined);save('pdf_complete.json',decoded)
            assert decoded['status']=='requires_raster_review' and decoded['needs_ocr_review']
            assert decoded['carrier_kind']=='raster' and decoded['raster_regions']
            report['isolated_pdf_chunks']=len(chunks)
            wrong=await call('pdf_wrong_sha','extract_band_from_image',{**args,'pdf_result_offset':500000,'pdf_result_sha256':'0'*64})
            assert wrong['error_code']=='PDF_RESULT_VERSION_CHANGED'
            multi=await call('composite','extract_band_from_image',{'payload_base64':image})
            assert multi['status']=='requires_panel_selection' and multi['panel_bbox'] is None
            report['raster_candidates']=multi['panel_candidates']
            assert len(multi['panel_candidates'])==3
            for label,kind,bbox in [('a','electronic_band',[88,37,454,403]),
                                   ('b','phonon',[605,37,973,403]),('c','transport',[1105,37,1472,403])]:
                selection={'source_sha256':image_sha,'page_number':1,'coordinate_unit':'pixel',
                    'panel_bbox_xyxy':bbox,'panel_kind':kind}
                r=await call('panel_'+label,'extract_band_from_image',{'payload_base64':image,'panel_selection':selection})
                if label=='a':
                    assert r['visible_ink']['export_complete']
                    (out/'mcp_visible_ink.svg').write_text(r['visible_ink']['svg'],encoding='utf-8')
                    (out/'mcp_visible_ink_runs.csv').write_text(r['visible_ink']['csv_text'],encoding='utf-8')
                    report['visible_ink_point_count']=r['visible_ink']['point_count']
                else:assert r['status']=='non_electronic_panel' and r['band_data'] is None
            selection={'source_sha256':sha,'page_number':2,'coordinate_unit':'pdf_point',
                       'panel_bbox_xyxy':[123.477,60.435,238.437,179.944],'panel_kind':'electronic_band'}
            r=await call('pdf_roi','extract_band_from_image',{**args,'panel_selection':selection})
            assert r['status']=='requires_calibration' and r['crop_png_base64']
            (out/'mcp_pdf_roi.png').write_bytes(base64.b64decode(r['crop_png_base64']))
            ocr=await call('ocr','extract_document',{'payload_base64':image,'kind':'image','max_pages':1})
            assert ocr['pages'][0]['axis_token_review']['corrected_text'] is None
            old=ROOT/'artifacts/reports/aceami_20260913_chain_test'
            data=json.loads((old/'edge_proxy_numerical.json').read_text(encoding='utf-8'))
            obs={'schema_version':2,'source':{'document_id':'sha256:'+sha,'source_sha256':sha,'page_number':2,'panel_label':'Fig2(a)','material_label':'ZnGa2O4'},
                'energy_unit':'eV','energy_reference':{'kind':'arbitrary','value_eV':0.,'observed':True,
                    'evidence':'Visible zero-energy tick; not asserted to be the physical Fermi level.'},
                'k_unit':'relative','k_distance':data['k_distance'],'segment_ids':data['segment_ids'],
                'bands':[{'band_id':'valence_envelope_proxy','role':'valence','energies_eV':data['energies_eV'][0]},
                         {'band_id':'conduction_visible_proxy','role':'conduction','energies_eV':data['energies_eV'][1]}],
                'calibration_evidence':{'energy_tick_count':6,'k_anchor_count':5},
                'ambiguities':['Raster envelopes are not unique physical band identities; overlaid and cropped branches remain unresolved.'],
                'human_confirmed':False}
            obs['bands'][1]['energies_eV'][10]=None
            save('paper_v2_observations.json',obs)
            r=await call('v2_paper','analyze_visual_observations',{'observations':obs})
            assert r['status']=='partial_observations' and r['fermi_eV'] is None and r['line_mode_gap_eV'] is None
            assert r['coverage']['missing_samples']==1 and r['exports']['csv_text']
            (out/'mcp_v2_samples.csv').write_text(r['exports']['csv_text'],encoding='utf-8')
            (out/'mcp_v2_samples.svg').write_text(r['exports']['svg'],encoding='utf-8')
            await call('v2_questions','request_missing_evidence',{'observations':obs})
            r=await call('v2_pending','prepare_human_audit_candidate',{'observations':obs})
            assert not r['human_audited'] and not r['eligible_for_scientific_acceptance']
            await call('knowledge','get_material_knowledge',{'material_system':'oxide'})
            # Clearly synthetic engine control, not an observed paper spectrum.
            control={'energies_eV':[[-2,-1,-2],[2,1,2]],'k_distance':[0,.5,1],
                     'segment_ids':[0,0,0],'fermi_eV':0,'k_unit':'relative'}
            await call('synthetic_numeric_control','analyze_band_structure',{'input_type':'numerical_data','data':control})
            await call('synthetic_physics_control','validate_physics',{'band_data':control})
            end=await call('status_end','get_service_status',{})
            assert end['build_sha256']==report['build_sha256'] and end['started_utc']==status['started_utc']
            report.update(status='engineering_repair_checks_passed',tool_count=len({x['tool'] for x in report['cases'].values()}),
                          call_count=len(report['cases']),full_physical_band_reconstruction=False,
                          corpus_membership='known_eval_figure_289_not_a_blind_test')
            save('report.json',report)
    # A new parent must explicitly expire the old result, never rerun and mix it.
    async with stdio_client(params,errlog=(out/'restart_stderr.txt').open('w',encoding='utf-8')) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            new_status=await call('restarted_status','get_service_status',{})
            assert new_status['build_sha256']==report['build_sha256']
            assert new_status['started_utc']!=status['started_utc']
            expired=await call('restarted_continuation','extract_band_from_image',{
                **args,'pdf_result_offset':500000,'pdf_result_sha256':first['payload_sha256']})
            assert expired['error_code']=='PDF_RESULT_EXPIRED'
            report.update(call_count=len(report['cases']),restart_continuation='PDF_RESULT_EXPIRED')
            save('report.json',report)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    save('manifest.json',{'files_sha256':hashes})
    print(json.dumps({k:v for k,v in report.items() if k not in {'cases','initialize','tool_schema'}},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--paper',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();asyncio.run(run(a.paper,a.output_dir))
