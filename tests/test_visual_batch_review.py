import json
import pytest
from mcp_server.visual_batch_review import make_visual_batch, save_visual_batch, finish_visual_review

def queue():
    return {'record_count':2,'dataset_manifest_sha256':'a'*64,'preannotation_manifest_sha256':'b'*64,
        'records':[{'candidate_id':f'epmc-band-{i:024x}','group_id':'PMC1',
                    'image':{'sha256':str(i)*64,'width':1000,'height':500},
                    'preannotation_sha256':'c'*64} for i in (1,2)]}

def observations():
    return [{'i':1,'target':'yes','boxes':[[10,20,40,90],[60,20,90,90]],'note':'Two separate E(k) panels are visible.'},
            {'i':2,'target':'no','boxes':[],'note':'Crystal structure and a band alignment schematic.'}]

def test_multi_panel_pixels_coverage_and_immutable_save(tmp_path):
    q=queue()
    save_visual_batch(q,0,observations(),tmp_path)
    with pytest.raises(FileExistsError): save_visual_batch(q,0,observations(),tmp_path)
    result=finish_visual_review(q,tmp_path)
    assert result['target_counts']=={'yes':1,'no':1,'uncertain':0}
    assert result['panel_count']==2
    assert result['records'][0]['panel_boxes_pixels_xyxy'][0]==[100,100,400,450]
    assert result['human_audited_count']==0

@pytest.mark.parametrize('box',[[0,0,101,100],[40,10,20,30],[False,10,30,50],[0,0,float('nan'),100]])
def test_bad_coordinates_refused(box):
    items=observations(); items[0]['boxes']=[box]
    with pytest.raises(ValueError): make_visual_batch(queue(),0,items)

def test_large_composite_figure_keeps_all_panels_with_a_bound():
    items=observations()
    items[0]['boxes']=[[x,y,x+8,y+15] for y in (0,20,40,60,80) for x in range(0,90,10)]
    assert len(make_visual_batch(queue(),0,items)['records'][0]['panel_boxes_pixels_xyxy'])==45
    items[0]['boxes']=[[0,0,10,10]]*129
    with pytest.raises(ValueError):make_visual_batch(queue(),0,items)

def test_missing_or_wrong_figure_refused():
    items=observations();items[0]['i']=2
    with pytest.raises(ValueError): make_visual_batch(queue(),0,items)
    with pytest.raises(ValueError): make_visual_batch(queue(),0,items[:1])

def test_forged_saved_approval_and_missing_coverage_refused(tmp_path):
    q=queue()
    path=save_visual_batch(q,0,observations()[:1],tmp_path,1)
    with pytest.raises(ValueError,match='Incomplete'):finish_visual_review(q,tmp_path)
    obj=json.loads(path.read_text());obj['records'][0]['human_audited']=True
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='modified'):finish_visual_review(q,tmp_path)

def test_read_only_validation_refuses_changed_summary(tmp_path):
    q=queue();save_visual_batch(q,0,observations(),tmp_path);finish_visual_review(q,tmp_path)
    assert finish_visual_review(q,tmp_path,write_manifest=False)['record_count']==2
    path=tmp_path/'visual_review_manifest.json'
    saved=json.loads(path.read_text());saved['target_counts']['yes']=2
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='Summary'):finish_visual_review(q,tmp_path,write_manifest=False)

def test_export_preserves_source_binding_and_pixel_box(tmp_path):
    from test_figure_candidate_review import fixtures
    from mcp_server.figure_review_queue import load_figure_review_queue
    from scripts.export_oa_visual_review import export_review
    dataset,pre=fixtures(tmp_path)
    q=load_figure_review_queue(dataset,pre)
    review_dir=tmp_path/'visual'; output=tmp_path/'export'
    save_visual_batch(q,0,[{'i':1,'target':'yes','boxes':[[10,20,90,80]],
                           'note':'One isolated electronic dispersion panel.'}],review_dir)
    finish_visual_review(q,review_dir)
    report=export_review(dataset,pre,review_dir,output)
    assert report['record_count']==1 and report['formal_approved_count']==0
    coco=json.loads((output/'panels_coco_ai.json').read_text())
    assert coco['annotations'][0]['bbox']==[32,48,256,144]
    assert coco['annotations'][0]['human_audited'] is False
    with pytest.raises(FileExistsError):export_review(dataset,pre,review_dir,output)
