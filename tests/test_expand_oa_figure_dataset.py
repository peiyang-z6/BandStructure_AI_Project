import io
import json
from PIL import Image
from scripts import expand_oa_figure_dataset as expansion
from src.data.europe_pmc_figure_harvester import build_candidate_record, write_snapshot


def candidate(pmcid, colour):
    out=io.BytesIO();Image.new('RGB',(320,240),colour).save(out,format='PNG');payload=out.getvalue()
    article={'pmcid':pmcid,'doi':'10.1/test','title':'Test','pubYear':'2025',
             'isOpenAccess':'Y','isRetracted':'N','license':'cc by'}
    figure={'figure_id':'F1','figure_label':'Figure 1','caption':'Electronic band structure','asset_href':'f1.png'}
    return build_candidate_record(article,figure,payload,'a'*64,'b'*64),payload


def test_extension_preserves_parent_bytes_order_and_deduplicates_images(tmp_path,monkeypatch):
    old=candidate('PMC1','white');dup=candidate('PMC2','white');new=candidate('PMC3','blue')
    parent=tmp_path/'old';output=tmp_path/'new'
    write_snapshot(parent,[old],1)
    parent_bytes={p.relative_to(parent).as_posix():p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    monkeypatch.setattr(expansion,'_search_page',lambda *a:{'resultList':{'result':[{'pmcid':'PMC2'},{'pmcid':'PMC3'}]},'nextCursorMark':'end','hitCount':2})
    monkeypatch.setattr(expansion,'_article_candidates',lambda article,*args:[dup if article['pmcid']=='PMC2' else new])
    assert expansion.collect(parent,output,2,workers=1)
    m=json.loads((output/'manifest.json').read_text())
    assert m['record_count']==2 and m['added_count']==1
    assert m['records'][0]['candidate_id']==old[0]['candidate_id']
    assert len({r['image_sha256'] for r in m['records']})==2
    for rel,payload in parent_bytes.items():
        assert (parent/rel).read_bytes()==payload
        if rel!='manifest.json':assert (output/rel).read_bytes()==payload


def test_prepare_inherits_visual_labels_and_times_without_reannotation(tmp_path,monkeypatch):
    from test_figure_candidate_review import fixtures
    from mcp_server.figure_review_queue import load_figure_review_queue
    from mcp_server.visual_batch_review import save_visual_batch,finish_visual_review
    dataset,pre=fixtures(tmp_path)
    q=load_figure_review_queue(dataset,pre)
    parent_review=tmp_path/'old_review'
    save_visual_batch(q,0,[{'i':1,'target':'no','boxes':[],'note':'Test fixture white image without a band plot.'}],parent_review)
    finish_visual_review(q,parent_review)
    import shutil
    expanded=tmp_path/'expanded';shutil.copytree(dataset.parent,expanded)
    new_pre=tmp_path/'new_pre';new_review=tmp_path/'new_review'
    monkeypatch.setattr('src.vision.figure_candidate_preannotator._preannotate',lambda _: (_ for _ in ()).throw(AssertionError('must reuse parent')))
    expansion.prepare(dataset.parent,pre.parent,parent_review,expanded,new_pre,new_review)
    old_packet=expansion.read(next(parent_review.glob('batch_*.json')))
    new_packet=expansion.read(next(new_review.glob('batch_*.json')))
    assert old_packet['records']==new_packet['records']
    assert old_packet['reviewed_at']==new_packet['reviewed_at']
    assert new_packet['inherited_from']['sha256']==expansion.digest(next(parent_review.glob('batch_*.json')))
