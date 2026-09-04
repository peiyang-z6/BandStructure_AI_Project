"""Merge proxy-channel unique groups into the main 60k snapshot up to 60,000 total.

Deterministic, provenance-preserving: selects proxy-only material IDs by
seeded random sample (seed 42), copies the FULL HDF5 groups byte-for-byte via
h5py group copy (datasets + attrs), then verifies counts and integrity.
"""
import json
import random
import shutil
from datetime import datetime, timezone

import h5py

MAIN = "data/raw/aflow/snapshots/aflow_60000_20260831/aflow_bands.h5"
PROXY = "data/raw/aflow/snapshots/aflow_60000_proxy_20260901/aflow_bands.h5"
TARGET = 60000
SEED = 42

with h5py.File(MAIN, "r") as h:
    main_ids = {k for k in h if not k.startswith("__tmp__")}
with h5py.File(PROXY, "r") as h:
    proxy_ids = {k for k in h if not k.startswith("__tmp__")}

proxy_only = sorted(proxy_ids - main_ids)
need = TARGET - len(main_ids)
assert 0 <= need <= len(proxy_only), f"need {need} vs proxy-only {len(proxy_only)}"

rng = random.Random(SEED)
selected = sorted(rng.sample(proxy_only, need))
dropped = sorted(set(proxy_only) - set(selected))

# Backup main before mutation.
shutil.copy(MAIN, MAIN + ".pre_merge_to_60k.bak")

with h5py.File(MAIN, "a") as h_main:
    with h5py.File(PROXY, "r") as h_proxy:
        for mid in selected:
            h_proxy.copy(h_proxy[mid], h_main, name=mid)

with h5py.File(MAIN, "r") as h:
    final_ids = [k for k in h if not k.startswith("__tmp__")]
    tmp = [k for k in h if k.startswith("__tmp__")]
    broken = []
    for k in final_ids:
        g = h[k]
        for req in ("energies", "k_distances", "kpoints"):
            if req not in g:
                broken.append(k)
                break

assert len(final_ids) == TARGET, f"final count {len(final_ids)} != {TARGET}"
assert not tmp, f"tmp groups present: {len(tmp)}"
assert not broken, f"broken groups: {len(broken[:10])}"

manifest = {
    "operation": "merge_proxy_channel_to_60000",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "main_before": len(main_ids),
    "proxy_total": len(proxy_ids),
    "proxy_only": len(proxy_only),
    "merged": need,
    "dropped": len(dropped),
    "final_total": len(final_ids),
    "seed": SEED,
    "selected_ids": selected,
    "dropped_ids": dropped,
}
with open(
    "data/raw/aflow/snapshots/aflow_60000_20260831/merge_to_60000_manifest.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(manifest, f, ensure_ascii=False, indent=1)

print(
    f"main {len(main_ids)} + merged {need} = {len(final_ids)} "
    f"(dropped {len(dropped)} proxy-only), tmp={len(tmp)}, broken={len(broken)}"
)
print("manifest:", "merge_to_60000_manifest.json")
