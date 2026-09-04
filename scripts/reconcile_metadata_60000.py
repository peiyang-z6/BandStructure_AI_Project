"""Reconcile main-60k metadata sidecar with the post-merge HDF5.

Constitution §4: canonical metadata's material ID set must exactly match the
canonical HDF5. After merge_proxy_to_60000.py copied 4,524 proxy-only groups,
the metadata JSON is short by the same set. Copy those records (all present in
the proxy metadata) into the main metadata file; overwrite conflict-free
(batch_download's merge_metadata_json already keeps non-empty-field precedence).
"""
import json
import shutil

MAIN_META = "data/raw/aflow/snapshots/aflow_60000_20260831/aflow_metadata.json"
PROXY_META = "data/raw/aflow/snapshots/aflow_60000_proxy_20260901/aflow_metadata.json"
MERGE_MANIFEST = "data/raw/aflow/snapshots/aflow_60000_20260831/merge_to_60000_manifest.json"

main_meta = json.load(open(MAIN_META, encoding="utf-8"))
proxy_meta = json.load(open(PROXY_META, encoding="utf-8"))
manifest = json.load(open(MERGE_MANIFEST, encoding="utf-8"))

main_ids = {m.get("material_id") for m in main_meta}
needed = set(manifest["selected_ids"]) - main_ids
assert len(needed) == len(manifest["selected_ids"]), "selected ids already present?"

shutil.copy(MAIN_META, MAIN_META + ".pre_reconcile.bak")
proxy_by_id = {m.get("material_id"): m for m in proxy_meta}
added = 0
for mid in manifest["selected_ids"]:
    rec = proxy_by_id.get(mid)
    if rec is None:
        raise RuntimeError(f"selected id missing from proxy metadata: {mid}")
    main_meta.append(rec)
    added += 1

json.dump(main_meta, open(MAIN_META, "w", encoding="utf-8"), ensure_ascii=False)
print(f"metadata: {len(main_meta) - added} + {added} = {len(main_meta)}")
print(
    "ids now:",
    len({m.get("material_id") for m in main_meta}),
    "(expect 60000)",
)
