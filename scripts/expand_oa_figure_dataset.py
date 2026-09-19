"""Expand the existing OA figure corpus without rewriting the parent snapshot."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.data.europe_pmc_figure_harvester import (
    _article_candidates, _search_page, _write_candidate, _manifest, _atomic_json, RateLimiter)


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write_new(path, value):
    with Path(path).open('x',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2,ensure_ascii=False,allow_nan=False)


def seed(parent, output):
    manifest=read(parent/'manifest.json')
    if manifest['status']!='complete_ai_assisted_figure_candidate_snapshot':
        raise ValueError('Parent snapshot is not complete.')
    # Verify all parent images before creating the extension.
    for entry in manifest['records']:
        if digest(parent/entry['image_path'])!=entry['image_sha256']:
            raise ValueError('Parent image SHA mismatch.')
        record=read(parent/entry['record_path'])
        if record['candidate_id']!=entry['candidate_id'] or record['image']['sha256']!=entry['image_sha256']:
            raise ValueError('Parent record binding mismatch.')
    output.mkdir(parents=True,exist_ok=False)
    (output/'images').mkdir();(output/'records').mkdir();(output/'harvest_pages').mkdir()
    copied={}
    for entry in manifest['records']:
        for relative in (entry['image_path'],entry['record_path']):
            shutil.copyfile(parent/relative,output/relative)
            copied[relative]=digest(parent/relative)
            if digest(output/relative)!=copied[relative]:raise ValueError('Seed copy SHA mismatch.')
    lineage={'parent_manifest_path':str((parent/'manifest.json').resolve()),
             'parent_manifest_sha256':digest(parent/'manifest.json'),
             'parent_count':len(manifest['records']),'copied_files_sha256':copied}
    write_new(output/'parent_lineage.json',lineage)
    # Restart the last parent page: it may have fetched but not retained excess figures.
    # Full restart plus group/image dedup also handles changing search relevance order.
    write_new(output/'expansion_state.json',{'cursor':'*','pages':0,'articles_scanned':0,
        'query':manifest['source_query'],'target_count':1000,
        'started_utc':datetime.now(timezone.utc).isoformat()})
    return manifest


def collect(parent, output, target=1000, workers=6, page_size=40, timeout=25, query=None):
    parent=parent.resolve();output=output.resolve()
    parent_manifest=read(parent/'manifest.json')
    parent_groups={r['group_id'] for r in parent_manifest['records']}
    if not output.exists():seed(parent,output)
    elif (output/'manifest.json').exists():raise FileExistsError('Extension already frozen.')
    lineage=read(output/'parent_lineage.json')
    if lineage['parent_manifest_sha256']!=digest(parent/'manifest.json'):
        raise ValueError('Changed parent snapshot.')
    state=read(output/'expansion_state.json')
    state.setdefault('source_queries',[state['query']])
    if query and query!=state['query']:
        state['source_queries'].append(query)
        state['query']=query
        state['cursor']='*'
        _atomic_json(output/'expansion_state.json',state)
    records={}
    for path in sorted((output/'records').glob('*.json')):
        r=read(path)
        if digest(output/r['image']['path'])!=r['image']['sha256']:raise ValueError('Existing image SHA mismatch.')
        records[r['candidate_id']]=r
    groups=Counter(r['group_id'] for r in records.values())
    completed_groups_at_start=set(groups)
    hashes={r['image']['sha256'] for r in records.values()}
    identities={(r['source']['pmcid'],r['source']['figure_id']) for r in records.values()}
    limiter=RateLimiter(.25)
    while len(records)<target:
        page=_search_page(state['query'],state['cursor'],page_size,limiter,timeout)
        articles=page['resultList']['result']
        if not articles:break
        selected=[a for a in articles if a.get('pmcid') not in completed_groups_at_start
                  and groups.get(a.get('pmcid'),0)<2]
        outcomes=[]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(_article_candidates,a,limiter,timeout,2) for a in selected]
            for article,future in zip(selected,futures):
                try: candidates=future.result()
                except Exception as exc:
                    outcomes.append({'pmcid':article.get('pmcid'),'status':'failed','error_type':type(exc).__name__})
                    continue
                accepted=0
                for r,payload in candidates:
                    identity=(r['source']['pmcid'],r['source']['figure_id'])
                    if (len(records)>=target or groups[r['group_id']]>=2 or r['candidate_id'] in records
                            or r['image']['sha256'] in hashes or identity in identities):continue
                    persisted=_write_candidate(output,r,payload)
                    records[r['candidate_id']]=persisted;groups[r['group_id']]+=1
                    hashes.add(r['image']['sha256']);identities.add(identity);accepted+=1
                outcomes.append({'pmcid':article.get('pmcid'),'candidates':len(candidates),'accepted':accepted})
        state['pages']+=1;state['articles_scanned']+=len(articles)
        page_log=output/'harvest_pages'/f'page_{state["pages"]:04d}.json'
        write_new(page_log,{'query':state['query'],'requested_cursor':state['cursor'],
                            'next_cursor':page.get('nextCursorMark'),'articles':articles,'outcomes':outcomes})
        next_cursor=page.get('nextCursorMark')
        exhausted=not next_cursor or next_cursor==state['cursor']
        state.update(cursor=next_cursor or state['cursor'],record_count=len(records),
                     document_group_count=len(groups),hit_count=page.get('hitCount'),
                     updated_utc=datetime.now(timezone.utc).isoformat(),target_count=target)
        _atomic_json(output/'expansion_state.json',state)
        print(json.dumps({'records':len(records),'groups':len(groups),'scanned':state['articles_scanned'],
                          'hits':page.get('hitCount')}),flush=True)
        if exhausted:break
    if len(records)<target:
        print(json.dumps({'status':'incomplete','records':len(records),'target':target}),flush=True)
        return False
    manifest=_manifest(records.values(),target,state['query'])
    parent_ids=[r['candidate_id'] for r in parent_manifest['records']]
    ordered=parent_ids+sorted(set(records)-set(parent_ids))
    entries={r['candidate_id']:r for r in manifest['records']}
    manifest['records']=[entries[cid] for cid in ordered]
    manifest['parent_manifest_sha256']=lineage['parent_manifest_sha256']
    manifest['source_queries']=state['source_queries']
    manifest['inherited_count']=len(parent_ids)
    manifest['added_count']=target-len(parent_ids)
    manifest['completed_utc']=datetime.now(timezone.utc).isoformat()
    for relative,sha in lineage['copied_files_sha256'].items():
        if digest(output/relative)!=sha or digest(parent/relative)!=sha:raise ValueError('Parent changed during expansion.')
    write_new(output/'manifest.json',manifest)
    print(json.dumps({'status':'frozen','records':target,'new':target-len(parent_ids),
                      'groups':len(groups),'manifest_sha256':digest(output/'manifest.json')}),flush=True)
    return True


def prepare(parent_dataset, parent_pre, parent_review, dataset, pre_dir, review_dir):
    from src.vision.figure_candidate_preannotator import _load_snapshot, _preannotate
    from mcp_server.figure_review_queue import load_figure_review_queue
    from mcp_server.visual_batch_review import finish_visual_review, make_visual_batch
    old_queue=load_figure_review_queue(parent_dataset/'manifest.json',parent_pre/'manifest.json')
    old_review=finish_visual_review(old_queue,parent_review,write_manifest=False)
    manifest,loaded=_load_snapshot(dataset/'manifest.json')
    pre_dir.mkdir(parents=True,exist_ok=False);(pre_dir/'records').mkdir()
    old_ids={r['candidate_id'] for r in old_queue['records']}
    annotations={}
    for cid in old_ids:
        path=parent_pre/'records'/f'{cid}.json'
        shutil.copyfile(path,pre_dir/'records'/path.name)
        annotations[cid]=read(path)
    added=[item for item in loaded if item[0]['candidate_id'] not in old_ids]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for ann in pool.map(_preannotate,added):
            write_new(pre_dir/'records'/f'{ann["candidate_id"]}.json',ann)
            annotations[ann['candidate_id']]=ann
    ordered=[annotations[r['candidate_id']] for r in manifest['records']]
    pm={'schema_version':1,'status':'complete_ai_assisted_panel_preannotations',
        'source_manifest_path':str((dataset/'manifest.json').resolve()),
        'source_manifest_sha256':digest(dataset/'manifest.json'),
        'parent_preannotation_manifest_sha256':digest(parent_pre/'manifest.json'),
        'inherited_count':len(old_ids),'record_count':len(ordered),
        'status_counts':dict(Counter(a['status'] for a in ordered)),
        'human_audited_count':0,'formal_approved_count':0,'eligible_for_scientific_acceptance':False,
        'records':[{'candidate_id':a['candidate_id'],'group_id':a['group_id'],
                    'image_sha256':a['image_sha256'],'status':a['status'],
                    'record_path':f'records/{a["candidate_id"]}.json'} for a in ordered]}
    write_new(pre_dir/'manifest.json',pm)
    new_queue=load_figure_review_queue(dataset/'manifest.json',pre_dir/'manifest.json')
    review_dir.mkdir(parents=True,exist_ok=False)
    imports=[]
    for source_path in sorted(parent_review.glob('batch_*.json')):
        old=read(source_path)
        submissions=[{'i':r['index'],'target':r['target'],'boxes':r['panel_boxes_pct_xyxy'],
                      'note':r['observation']} for r in old['records']]
        packet=make_visual_batch(new_queue,old['start_index'],submissions,old['record_count'])
        if packet['records']!=old['records']:raise ValueError('Inherited labels do not match parent.')
        packet['reviewed_at']=old['reviewed_at']
        packet['inherited_from']={'path':str(source_path.resolve()),'sha256':digest(source_path)}
        write_new(review_dir/source_path.name,packet)
        imports.append(packet['inherited_from'])
    write_new(review_dir/'inheritance_manifest.json',{
        'parent_visual_manifest_sha256':digest(parent_review/'visual_review_manifest.json'),
        'inherited_labels_count':old_review['record_count'],'records_unchanged':True,
        'new_visual_labels_required':len(ordered)-old_review['record_count'],'source_batches':imports})
    print(json.dumps({'status':'ready_for_new_visual_labels','total':len(ordered),
                      'inherited':len(old_ids),'new':len(added),'cv_status_counts':pm['status_counts']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['collect','prepare'])
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--target',type=int,default=1000)
    p.add_argument('--query')
    p.add_argument('--parent-pre',type=Path);p.add_argument('--parent-review',type=Path)
    p.add_argument('--pre-output',type=Path);p.add_argument('--review-output',type=Path)
    a=p.parse_args()
    if a.mode=='collect':
        if not collect(a.parent,a.output,a.target,query=a.query):raise SystemExit(2)
    else:prepare(a.parent,a.parent_pre,a.parent_review,a.output,a.pre_output,a.review_output)
