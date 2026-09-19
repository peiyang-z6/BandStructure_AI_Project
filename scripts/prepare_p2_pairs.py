"""Split-local, immutable P2 pair preparation and lightweight artifact contracts.

New outputs only. Legacy caches are inspection-only (--legacy-diagnostic), never
training inputs. Provenance paths are descriptive: consumers do not follow them
into peer/outer data. Completion is published only after payload/source checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
import tempfile
import zipfile
from contextlib import contextmanager

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.crystal_graph import MAX_NEIGHBORS, build_crystal_graph
from src.utils import selection_manifest as upstream

SCHEMA = "p2-runtime-v1"
FEATURE_ORDER = ["VBM_E", "VBM_curv", "VBM_k_dist", "CBM_E", "CBM_curv", "CBM_k_dist"]
FEATURE_SEMANTICS = {"order": FEATURE_ORDER, "energy_reference": "E-E_F",
                     "edges": "label_free_occupied_empty", "curvature": "segment_local_proxy",
                     "distance": "extremum_k_distance", "segments": "per_row_segment_ids"}


class SnapshotRef(dict):
    """Serializable content identity plus process-local physical snapshot pin."""


class SnapshotMetadata(dict):
    def __init__(self, value, sources):
        super().__init__(value)
        self.snapshots = sources


def _stamp(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _capture(path, target=None):
    path = Path(path)
    h = hashlib.sha256()
    resolved = path.resolve()
    with path.open("rb") as f:
        before = _stamp(os.fstat(f.fileno()))
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
            if target is not None:
                target.write(block)
        if before != _stamp(os.fstat(f.fileno())):
            raise ValueError("source changed while reading snapshot")
    if path.resolve() != resolved or before != _stamp(path.stat()):
        raise ValueError("source changed while reading snapshot")
    ref = SnapshotRef(sha256=h.hexdigest(), bytes=before[2])
    ref.stamp = (str(resolved), before)
    return ref


def file_ref(path):
    return _capture(path)


def check_ref(actual, expected):
    if actual != expected or (hasattr(expected, "stamp") and actual.stamp != expected.stamp):
        raise ValueError("source changed / snapshot SHA identity mismatch")


def verify_sources(sources):
    for path, expected in sources.items():
        check_ref(file_ref(path), expected)


@contextmanager
def snapshot(path, expected=None):
    """Hash and consume one private copy, bounded RAM even for large NPZs."""
    with tempfile.TemporaryDirectory(prefix="p2-read-") as tmp:
        copy = Path(tmp) / Path(path).name
        with copy.open("xb") as target:
            ref = _capture(path, target)
        if expected is not None:
            check_ref(ref, expected)
        yield copy, ref
        verify_sources({Path(path):ref})


def read_json_snapshot(path, expected=None):
    with snapshot(path, expected) as (copy, ref):
        value = json.loads(copy.read_text(encoding="utf-8"))
    return value, ref


def source_refs(*metadata):
    result = {}
    for value in metadata:
        result.update(getattr(value, "snapshots", {}))
    return result


def code_refs(*paths):
    return {Path(p).resolve().relative_to(ROOT).as_posix(): file_ref(p) for p in paths}


CODE_STAGES = {
    "raw_tensor": set(upstream.P2_RAW_CODE),
    "ssl_anchor": set(upstream.P2_SSL_CODE),
    "pairs": {"scripts/prepare_p2_pairs.py", "src/data/crystal_graph.py"},
    "equivariant_graphs": {"scripts/prepare_p2_pairs.py", "scripts/prepare_p2_equivariant_graphs.py",
                           "src/data/crystal_graph.py", "src/data/structure_descriptors.py"},
    "band_embeddings": {"scripts/extract_band_embeddings.py", "scripts/prepare_p2_pairs.py",
                        "src/data/band_structure_dataset.py", "src/models/band_structure_encoder.py"},
    "tensorflow": {"scripts/train_p2_contrastive.py", "scripts/extract_band_embeddings.py",
                   "scripts/prepare_p2_pairs.py", "src/data/band_structure_dataset.py",
                   "src/models/crystal_graph_encoder.py", "src/models/band_structure_encoder.py"},
    "torch": {"scripts/train_p2_equivariant.py", "scripts/prepare_p2_pairs.py",
              "src/models/equivariant_structure_encoder.py"},
}


def check_code(refs, stage=None, *, read=True):
    allowed = set().union(*CODE_STAGES.values())
    if not isinstance(refs, dict) or not refs:
        raise ValueError("missing code bindings")
    if stage is not None and (stage not in CODE_STAGES or set(refs) != CODE_STAGES[stage]):
        raise ValueError("incomplete or unexpected stage code bindings")
    # Validate EVERY path before opening ANY file, including nested link targets.
    paths = []
    for rel in refs:
        if not isinstance(rel, str) or rel not in allowed:
            raise ValueError("code path outside explicit allowlist")
        path = ROOT / rel
        if any(p.is_symlink() for p in (path, *path.parents) if p != ROOT.parent):
            raise ValueError("symlink code path is forbidden")
        if path.resolve() != ROOT.resolve() / rel:
            raise ValueError("resolved code path outside allowlist")
        expected = refs[rel]
        validate_ref(expected)
        if "path" in expected and expected["path"] != str(path):
            raise ValueError("code reference path mismatch")
        paths.append((rel, path))
    if not read:
        return
    for rel, path in paths:
        expected = refs[rel]
        if "path" in expected and expected["path"] != str(path):
            raise ValueError("code reference path mismatch")
        if file_ref(path) != {k:expected[k] for k in ("bytes","sha256")}:
            raise ValueError(f"frozen code SHA mismatch: {rel}")


def fresh_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def check_stage_code(value, stage):
    """A known structural role cannot opt out by deleting its discriminator."""
    field = "framework" if stage in ("tensorflow", "torch") else "kind"
    if not isinstance(value, dict) or value.get(field) != stage:
        raise ValueError(f"expected explicit {stage} {field} for known stage")
    check_code(value.get("code"), stage, read=False)


def check_code_tree(value, *, read=True):
    """Verify embedded upstream code without following provenance data paths."""
    # Child roles come from the consuming schema, never from child self-report.
    roles = {
        "pairs": [(("raw_contract",), "raw_tensor")],
        "equivariant_graphs": [(("parent",), "pairs")],
        "band_embeddings": [(("parent",), "pairs"), (("anchor", "upstream"), "ssl_anchor")],
        "tensorflow": [(("data", "pairs"), "pairs"), (("anchor", "upstream"), "ssl_anchor")],
        "torch": [(("data", "graphs"), "equivariant_graphs"),
                  (("data", "embeddings"), "band_embeddings"), (("anchor", "upstream"), "ssl_anchor")],
    }
    bindings = []
    def collect(node):
        if isinstance(node,dict):
            stage = node.get("kind",node.get("framework"))
            if "code" in node or stage in CODE_STAGES:
                if stage not in CODE_STAGES:
                    raise ValueError("unknown code binding stage")
                bindings.append((node.get("code"), stage))
            for keys, expected_stage in roles.get(stage, ()):
                child = node
                for key in keys:
                    child = child.get(key) if isinstance(child, dict) else None
                check_stage_code(child, expected_stage)
            for child in node.values():
                collect(child)
        elif isinstance(node,list):
            for child in node:
                collect(child)
    collect(value)
    for refs, stage in bindings:
        check_code(refs, stage, read=False)
    if not read:
        return
    for refs, stage in bindings:
        check_code(refs, stage)


def write_json(path, content, *, sources=None):
    """Atomic replace is used only inside an exclusively created new run dir."""
    path = Path(path)
    pending = path.with_name(path.name + ".pending")
    with pending.open("x", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    verify_sources(sources or {})
    os.replace(pending, path)


def completion_path(path):
    return Path(str(path) + ".completion.json")


def save_completed_npz(path, arrays, contract, sources=None):
    path = Path(path)
    with path.open("xb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    # Source mutation/interruption leaves diagnostic bytes, never authorization.
    verify_sources(sources or {})
    check_code_tree(contract)
    identity = {k:arrays[k].tolist() for k in ("material_ids", "groups", "valid", "valid_idx", "valid_material_ids")}
    receipt = {**contract, "schema": SCHEMA, "status": "complete", "artifact": file_ref(path), "identity":identity}
    write_json(completion_path(path), receipt, sources=sources)
    return path


def read_completed_npz(path, kind, split, test_scope=False):
    receipt = completion_path(path)
    if not receipt.is_file():
        raise ValueError("missing completion contract; legacy diagnostic only")
    c, receipt_ref = read_json_snapshot(receipt)
    if (c.get("schema") != SCHEMA or c.get("status") != "complete"
            or c.get("kind") != kind or c.get("split") != split):
        raise ValueError("incompatible split/stage completion contract")
    if (type(c.get("test_scope")) is not bool or not isinstance(c.get("code"), dict) or not c["code"]
            or not isinstance(c.get("sources"), dict) or not c["sources"]):
        raise ValueError("incomplete code/source/scope completion contract; legacy diagnostic only")
    if c["test_scope"] and not test_scope:
        raise ValueError("test-scope artifact cannot authorize a formal run")
    check_code_tree(c, read=False)
    if file_ref(path) != c.get("artifact"):
        raise ValueError("payload SHA mismatch")
    check_code_tree(c)
    identity = {"material_ids", "valid", "valid_idx", "valid_material_ids", "groups"}
    allowed = {
        "pairs": identity | {"band_input", "segment_ids", "atom_features", "neighbor_list", "neighbor_dist", "n_atoms"},
        "band_embeddings": identity | {"band_embeddings"},
        "equivariant_graphs": identity | {"atom_features", "n_atoms", "descriptors", "edge_offsets", "pair_valid",
            "schema_version", "max_atoms", "max_neighbors", "cutoff_angstrom", "builder_sha256", "crystal_graph_sha256",
            "descriptors_sha256", "edge_src", "edge_dst", "edge_vec", "edge_len", "edge_image",
            "neighbor_count", "truncated_neighbors", "distance_12_13", "boundary_tied"},
    }
    with snapshot(path, c["artifact"]) as (copy, artifact_ref):
        with np.load(copy, allow_pickle=False) as f:
            if kind not in allowed or not set(f.files).issubset(allowed[kind]):
                raise ValueError("unexpected arrays in single-split schema; peer arrays are never decoded")
            arrays = {k: f[k] for k in f.files}
    c["artifact"] = artifact_ref
    sources = {Path(path):artifact_ref, receipt:receipt_ref}
    verify_sources(sources)
    return arrays, SnapshotMetadata(c, sources)


def validate_identity(ids, valid, valid_idx, valid_ids, groups):
    n = len(ids)
    if (ids.ndim != 1 or ids.dtype.kind != "U" or not n
            or any(not s or s != s.strip() for s in ids) or len(np.unique(ids)) != n):
        raise ValueError("material_ids must be unique nonempty literal strings")
    if valid.dtype != np.bool_ or valid.shape != (n,):
        raise ValueError("valid must be an aligned boolean vector")
    if (valid_idx.dtype.kind not in "iu" or not np.array_equal(valid_idx, np.flatnonzero(valid))
            or not np.array_equal(valid_ids, ids[valid])):
        raise ValueError("valid indices/material IDs order mismatch")
    if groups.shape != (n,) or groups.dtype.kind not in "iu" or np.any(groups <= 0):
        raise ValueError("groups must be aligned known integer space groups")


def validate_pairs(a, c):
    required = {"material_ids", "valid", "valid_idx", "valid_material_ids", "groups", "band_input",
                "segment_ids", "atom_features", "neighbor_list", "neighbor_dist", "n_atoms"}
    if not required.issubset(a):
        raise ValueError("missing pair arrays")
    if c.get("input_space") != "raw_canonical_6d" or c.get("feature_semantics") != FEATURE_SEMANTICS:
        raise ValueError("raw MBM feature contract required; never normalize a normalized cache twice")
    validate_identity(*(a[k] for k in ("material_ids", "valid", "valid_idx", "valid_material_ids", "groups")))
    n = len(a["material_ids"])
    for k in required - {"valid_idx", "valid_material_ids"}:
        if a[k].ndim == 0 or a[k].shape[0] != n:
            raise ValueError(f"raw sample axis mismatch: {k}")
        if a[k].dtype.kind in "fiu" and not np.isfinite(a[k]).all():
            raise ValueError(f"nonfinite pair array: {k}")
    band, atom = a["band_input"], a["atom_features"]
    if band.ndim != 3 or band.shape[-1] != 6 or not band.shape[1]:
        raise ValueError("canonical band shape required")
    if a["segment_ids"].shape != band.shape[:2] or a["segment_ids"].dtype.kind not in "iu":
        raise ValueError("segment schema mismatch")
    if atom.ndim != 3 or atom.shape[-1] != 110:
        raise ValueError("atom feature schema mismatch")
    neighbors, distances, na = a["neighbor_list"], a["neighbor_dist"], a["n_atoms"]
    if (neighbors.shape != (n, atom.shape[1], MAX_NEIGHBORS) or distances.shape != neighbors.shape
            or neighbors.dtype.kind not in "iu" or na.shape != (n,) or na.dtype.kind not in "iu"
            or np.any(na < 0) or np.any(na > atom.shape[1]) or np.any(distances < 0)
            or np.any(neighbors < -1) or np.any(neighbors >= na[:, None, None])):
        raise ValueError("graph capacity/neighbor contract mismatch")
    if not a["valid"].any() or np.any(na[a["valid"]] <= 0):
        raise ValueError("no usable graph rows")


def load_pairs(path, split, test_scope=False):
    a, c = read_completed_npz(path, "pairs", split, test_scope)
    validate_pairs(a, c)
    if test_scope and len(a["material_ids"]) > 32:
        raise ValueError("test scope is limited to 32 raw samples")
    return a, c


def load_sidecar_structures(sidecar_path, material_ids=None):
    from pymatgen.core import Lattice, Structure
    records, _ = read_json_snapshot(sidecar_path)
    allowed = set(map(str, material_ids)) if material_ids is not None else None
    out = {}
    for r in records:
        mid = r.get("material_id")
        if allowed is not None and mid not in allowed:
            continue
        if mid in out:
            raise ValueError(f"duplicate sidecar material_id: {mid}")
        sp = r.get("species_per_atom") or r.get("species")
        fc, lat = r.get("fractional_coordinates"), r.get("lattice")
        if not mid or sp is None or fc is None or lat is None or len(sp) != len(fc):
            continue
        try:
            Structure(Lattice(lat), sp, fc)
        except (TypeError, ValueError):
            continue
        out[mid] = r
    return out


def build_graphs_for_ids(structures, material_ids, max_atoms):
    n = len(material_ids)
    result = {"atom_features": np.zeros((n, max_atoms, 110), np.float32),
              "neighbor_list": np.full((n, max_atoms, MAX_NEIGHBORS), -1, np.int32),
              "neighbor_dist": np.zeros((n, max_atoms, MAX_NEIGHBORS), np.float32),
              "n_atoms": np.zeros(n, np.int32), "valid": np.zeros(n, bool)}
    for i, mid in enumerate(material_ids):
        rec = structures.get(str(mid))
        if rec is None:
            continue
        g = build_crystal_graph(rec["lattice"], rec.get("species_per_atom") or rec["species"],
                                rec["fractional_coordinates"], max_atoms=max_atoms)
        na = g["n_atoms"]
        for key in ("atom_features", "neighbor_list", "neighbor_dist"):
            result[key][i, :na] = g[key][:na]
        result["n_atoms"][i] = na
        result["valid"][i] = True
    return result


def prepare_split(tensor_npz, sidecar, out_dir, split, max_atoms=50, test_scope=False, *, tensor_contract=None):
    if split not in ("train", "test"):
        raise ValueError("one explicit split is required")
    if tensor_contract is None:
        raise ValueError("explicit upstream raw tensor contract is required; shape is not provenance")
    declared, contract_ref = read_json_snapshot(tensor_contract)
    check_stage_code(declared, "raw_tensor")
    check_code_tree(declared, read=False)
    sources = {Path(tensor_npz):file_ref(tensor_npz), Path(tensor_contract):contract_ref,
               Path(sidecar):file_ref(sidecar)}
    raw_contract = upstream.validate_raw_tensor_contract(tensor_contract, tensor_path=tensor_npz,
                                                         split=split, allow_test_scope=test_scope)
    verify_sources(sources)
    out = fresh_directory(out_dir)
    # NPZ lookup is lazy; never touch peer keys or a global split manifest.
    with snapshot(tensor_npz, sources[Path(tensor_npz)]) as (copy, _):
        with np.load(copy, allow_pickle=False) as f:
            ids = f[f"material_ids_{split}"]
            raw = f[f"X_{split}"]
            groups = f[f"groups_{split}"]
            segments = f[f"segment_ids_{split}"]
    if ids.tolist() != raw_contract["material_ids"] or groups.tolist() != raw_contract["groups"]:
        raise ValueError("raw contract row identity mismatch")
    if test_scope and len(ids) > 32:
        raise ValueError("test scope is limited to 32 raw samples; no prefix filtering")
    if raw.shape != (len(ids), 2, segments.shape[1], 3) or not np.isfinite(raw).all():
        raise ValueError("raw canonical tensor shape/finiteness mismatch")
    structures = load_sidecar_structures(sidecar, ids)
    graphs = build_graphs_for_ids(structures, ids, max_atoms)
    if not graphs["valid"].any():
        raise ValueError("zero valid graphs cannot complete a stage")
    arrays = {**graphs, "band_input": raw.transpose(0, 2, 1, 3).reshape(len(ids), raw.shape[2], 6),
              "material_ids": ids, "groups": groups, "segment_ids": segments,
              "valid_idx": np.flatnonzero(graphs["valid"]),
              "valid_material_ids": ids[graphs["valid"]]}
    contract = {"kind": "pairs", "split": split, "test_scope": bool(test_scope),
                "input_space": "raw_canonical_6d", "feature_semantics": FEATURE_SEMANTICS,
                "raw_contract":raw_contract, "feature_schema":raw_contract["feature_schema"],
                "sources": {"tensor": sources[Path(tensor_npz)], "sidecar": sources[Path(sidecar)],
                            "tensor_contract":sources[Path(tensor_contract)]},
                "code": code_refs(Path(__file__), ROOT/"src/data/crystal_graph.py"),
                "max_atoms": max_atoms}
    validate_pairs(arrays, contract)
    return save_completed_npz(out/f"p2_{split}.npz", arrays, contract, sources)


def prepare_equivariant_split(pairs, sidecar, out_dir, split, max_atoms=50, test_scope=False):
    """Credentialed adapter around the existing periodic graph builder, ONE split.

    Kept here so all P2 preparation shares the same upstream completion protocol;
    no changes to the geometry recipe or its independently owned implementation.
    """
    from scripts import prepare_p2_equivariant_graphs as builder
    out = fresh_directory(out_dir)
    a, parent = load_pairs(pairs, split, test_scope)
    sources = {**source_refs(parent), Path(sidecar):file_ref(sidecar)}
    structures = load_sidecar_structures(sidecar, a["material_ids"])
    n = len(a["material_ids"])
    result = {k:a[k] for k in ("material_ids", "valid", "valid_idx", "valid_material_ids", "groups")}
    result.update({"atom_features":np.zeros((n,max_atoms,110),np.float32),
                   "n_atoms":np.zeros(n,np.int64), "descriptors":np.zeros((n,22),np.float32),
                   "edge_offsets":np.zeros((n,2),np.int64), "pair_valid":a["valid"].copy(),
                   "schema_version":np.asarray(builder.GRAPH_SCHEMA_VERSION),
                   "max_atoms":np.asarray(max_atoms), "max_neighbors":np.asarray(12),
                   "cutoff_angstrom":np.asarray(8.0)})
    ledger = {"neighbor_count":np.zeros((n,max_atoms),np.int64),
              "truncated_neighbors":np.zeros((n,max_atoms),np.int64),
              "distance_12_13":np.zeros((n,max_atoms,2),np.float32),
              "boundary_tied":np.zeros((n,max_atoms),bool)}
    result.update(ledger)
    code = code_refs(Path(__file__), Path(builder.__file__), ROOT/"src/data/crystal_graph.py",
                     ROOT/"src/data/structure_descriptors.py")
    for key, path in (("builder_sha256",Path(builder.__file__)),
                      ("crystal_graph_sha256",ROOT/"src/data/crystal_graph.py"),
                      ("descriptors_sha256",ROOT/"src/data/structure_descriptors.py")):
        result[key] = np.asarray(file_ref(path)["sha256"])
    edge_keys = ("edge_src", "edge_dst", "edge_vec", "edge_len", "edge_image")
    edges = {k:[] for k in edge_keys}
    offset = 0
    for i, mid in enumerate(a["material_ids"]):
        result["edge_offsets"][i] = (offset,offset)
        if not a["valid"][i]:
            continue
        g = builder.build_one_graph(structures.get(str(mid), {}), max_atoms)
        if g is None:
            raise ValueError("equivariant graph quarantine differs from pair valid mask; explicit re-pairing required")
        for k in ("atom_features", "n_atoms"):
            result[k][i] = g[k]
        result["descriptors"][i] = g["desc"]
        for k in ledger:
            result[k][i] = g[k]
        for k in edge_keys:
            edges[k].append(g[k])
        stop = offset + len(g["edge_src"])
        result["edge_offsets"][i] = (offset,stop)
        offset = stop
    result.update({k:np.concatenate(v) for k,v in edges.items()})
    c = {"kind":"equivariant_graphs", "split":split, "test_scope":bool(test_scope),
         "code":code, "parent":parent, "graph_schema":builder.GRAPH_SCHEMA_VERSION,
         "feature_semantics":FEATURE_SEMANTICS,
         "sources":{"pairs":sources[Path(pairs)], "sidecar":sources[Path(sidecar)],
                    "pairs_completion":sources[completion_path(pairs)]}}
    validate_graphs(result, c, a["material_ids"], a["valid"])
    return save_completed_npz(out/f"graphs_{split}.npz", result, c, sources)


def validate_graphs(g, c, ids, valid):
    from scripts import prepare_p2_equivariant_graphs as builder
    builder.validate_graph_cache(g, expected_material_ids=ids, expected_valid=valid)
    if c.get("graph_schema") != builder.GRAPH_SCHEMA_VERSION or not np.array_equal(g["pair_valid"], valid):
        raise ValueError("graph schema/pair_valid contract mismatch")
    for key, rel in (("builder_sha256", "scripts/prepare_p2_equivariant_graphs.py"),
                     ("crystal_graph_sha256", "src/data/crystal_graph.py"),
                     ("descriptors_sha256", "src/data/structure_descriptors.py")):
        if g[key].item() != c["code"][rel]["sha256"]:
            raise ValueError("graph builder SHA differs from verified code")
    n, capacity = g["atom_features"].shape[:2]
    for key, shape, kind in (("neighbor_count",(n,capacity),"iu"),
                              ("truncated_neighbors",(n,capacity),"iu"),
                              ("distance_12_13",(n,capacity,2),"f"),
                              ("boundary_tied",(n,capacity),"b")):
        a = g.get(key)
        if a is None or a.shape != shape or a.dtype.kind not in kind or not np.isfinite(a).all() or np.any(a < 0):
            raise ValueError("missing/corrupt neighbor diagnostic: " + key)
    count, truncated, distances = g["neighbor_count"], g["truncated_neighbors"], g["distance_12_13"]
    present = np.arange(capacity)[None,:] < g["n_atoms"][:,None]
    if (np.any(count[~present]) or not np.array_equal(truncated, np.maximum(count-12,0))
            or np.any(distances > 8.00001)
            or not np.array_equal(distances[:,:,0] > 0, count >= 12)
            or not np.array_equal(distances[:,:,1] > 0, count >= 13)
            or np.any(distances[:,:,1][count >= 13] < distances[:,:,0][count >= 13])
            or not np.array_equal(g["boundary_tied"], (count > 12) & np.isclose(distances[:,:,0],distances[:,:,1],rtol=1e-7,atol=1e-8))):
        raise ValueError("inconsistent neighbor truncation/tie diagnostics")
    for i, (lo,hi) in enumerate(g["edge_offsets"]):
        if not np.array_equal(np.bincount(g["edge_src"][lo:hi], minlength=capacity), np.minimum(count[i],12)):
            raise ValueError("neighbor count disagrees with selected edges")


def inspect_anchor(encoder_path, norm_path, expected=None, *, anchor_contract=None, test_scope=False):
    """Check actual producer/norm/model metadata before any model deserialization."""
    if any(p is None for p in (anchor_contract, encoder_path, norm_path)):
        raise ValueError("explicit upstream SSL anchor contract required; unknown history is blocked")
    declared, contract_ref = read_json_snapshot(anchor_contract)
    check_stage_code(declared, "ssl_anchor")
    check_code_tree(declared, read=False)
    dependencies = anchor_dependencies(declared, encoder_path)
    raw_receipt_path = Path(declared["source_tensor_contract_ref"]["path"])
    raw_declared, raw_receipt_ref = read_json_snapshot(raw_receipt_path)
    check_stage_code(raw_declared, "raw_tensor")
    check_code_tree(raw_declared, read=False)
    sources = {Path(p):file_ref(p) for p in (encoder_path, norm_path)}
    sources[Path(anchor_contract)] = contract_ref
    sources[raw_receipt_path] = raw_receipt_ref
    for path, expected_ref in dependencies.items():
        actual = file_ref(path)
        check_ref(actual, {k:expected_ref[k] for k in ("bytes","sha256")})
        if path in sources:
            check_ref(actual, sources[path])
        sources[path] = actual
    try:
        validated = upstream.validate_ssl_anchor_contract(anchor_contract, encoder_path=encoder_path,
                                                          norm_path=norm_path, allow_test_scope=test_scope)
    except ValueError as exc:
        raise ValueError("upstream anchor validation failed: " + str(exc)) from exc
    verify_sources(sources)
    stats, norm_ref = read_json_snapshot(norm_path, sources[Path(norm_path)])
    if expected is not None:
        check_ref(norm_ref, expected["norm"])
    # Validate actual numbers before model I/O. No fit, clamping or default stats.
    mean, std = np.asarray(stats["mean"]), np.asarray(stats["std"])
    if mean.shape != (6,) or std.shape != (6,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("invalid frozen normalization statistics")
    with snapshot(encoder_path, sources[Path(encoder_path)]) as (copy, encoder_ref):
        if expected is not None:
            check_ref(encoder_ref, expected["encoder"])
        with zipfile.ZipFile(copy) as archive:
            config = json.loads(archive.read("config.json"))
        if config.get("class_name") != "SSLEncoder":
            raise ValueError("unexpected anchor model class")
    identity = {"encoder": encoder_ref, "norm": norm_ref,
                "norm_stats": stats, "feature_mode": "SSLEncoder.return_features.pooled",
                "feature_semantics": FEATURE_SEMANTICS, "upstream":validated,
                "feature_schema":validated["feature_schema"], "config":config["config"]}
    validate_anchor_metadata(identity, test_scope)
    if expected is not None and identity != expected:
        raise ValueError("reopened anchor contract differs from frozen run")
    return stats, SnapshotMetadata(identity, sources)


def anchor_dependencies(c, encoder_path):
    """Only fixed local producer evidence and the canonical train source.

    Preflight all dependency paths before reading any evidence. Never follow a
    raw HDF5 path or an arbitrary history/checkpoint filename from provenance.
    """
    try:
        directory = Path(encoder_path).resolve().parent
        epochs = c["completed_epochs"]
        if type(epochs) is not int or epochs < 1:
            raise ValueError("invalid anchor completed epochs")
        refs = {directory/"ssl_history.json":c["history_ref"],
                directory/"ssl_run_config.json":c["config_ref"],
                directory/"ssl_mbm_best.keras":c["states"]["best"],
                directory/f"ssl_mbm_final_epoch{epochs}.keras":c["states"]["last"]}
        for state in ("best","last"):
            expected_names = {f"ckpt-{state}.index", f"ckpt-{state}.data-00000-of-00001"}
            items = c["checkpoints"][state]
            if not isinstance(items,list) or len(items) != 2 or {Path(r["path"]).name for r in items} != expected_names:
                raise ValueError("invalid anchor checkpoint evidence set")
            refs.update({directory/Path(r["path"]).name:r for r in items})
        tensor = Path(c["source_tensor_ref"]["path"])
        receipt = tensor.with_name("band_tensors_ood_split.train.tensor_contract.json")
        if not tensor.is_absolute() or tensor.name != "band_tensors_ood_split.npz":
            raise ValueError("invalid anchor train source path")
        refs[tensor], refs[receipt] = c["source_tensor_ref"], c["source_tensor_contract_ref"]
        for path, expected in refs.items():
            validate_ref(expected)
            if expected.get("path") != str(path) or path.resolve() != path:
                raise ValueError("anchor dependency path escape/mismatch")
        return refs
    except (KeyError,TypeError) as exc:
        raise ValueError("incomplete anchor dependency contract") from exc


def inner_partition(ids, groups, valid_idx, validation_size=.2, seed=42, *, anchor=None, raw_contract=None):
    if not isinstance(anchor,dict) or not isinstance(raw_contract,dict):
        raise ValueError("upstream anchor allocation and raw contract required; no random re-partition")
    u = anchor["upstream"]
    if (u["source_tensor_ref"] != raw_contract["tensor_ref"]
            or u["source_tensor_contract_ref"] != raw_contract["contract_ref"]
            or ids.tolist() != u["material_ids"] or groups.tolist() != u["groups"]):
        raise ValueError("upstream fit/selection source or row identity conflict")
    fit = valid_idx[np.isin(ids[valid_idx], u["fit_material_ids"])]
    val = valid_idx[np.isin(ids[valid_idx], u["selection_material_ids"])]
    if min(len(fit), len(val)) < 2:
        raise ValueError("InfoNCE fit and validation each require at least two valid pairs")
    if sorted(np.concatenate((fit,val)).tolist()) != valid_idx.tolist() or set(groups[fit]) & set(groups[val]):
        raise ValueError("upstream allocation cannot cover paired valid rows without overlap")
    return fit, val, {"material_ids":ids.tolist(), "groups":groups.tolist(),
                      "fit_indices":fit.tolist(), "validation_indices":val.tolist(),
                      "fit_ids":ids[fit].tolist(), "validation_ids":ids[val].tolist(),
                      "allocation_source":"upstream_ssl", "upstream_run_id":u["run_id"],
                      "upstream_contract_ref":u["contract_ref"],
                      "excluded_ids":ids[~np.isin(np.arange(len(ids)), valid_idx)].tolist(),
                      "group_overlap":0}


def validate_anchor_metadata(anchor, test_scope):
    u = anchor.get("upstream")
    check_stage_code(u, "ssl_anchor")
    needed = {"scope","feature_schema","encoder_ref","norm_ref","contract_ref","material_ids","groups",
              "fit_material_ids","fit_groups","selection_material_ids","selection_groups",
              "source_tensor_ref","source_tensor_contract_ref","run_id","best_epoch","history"}
    if not isinstance(u,dict) or not needed <= u.keys() or u["scope"] not in ("test","formal") or (u["scope"] == "test" and not test_scope):
        raise ValueError("incomplete upstream anchor history/scope")
    for key in ("encoder","norm"):
        if anchor[key] != {k:u[key+"_ref"][k] for k in ("bytes","sha256")}:
            raise ValueError("anchor byte identity mismatch")
    if anchor.get("feature_schema") != u["feature_schema"]:
        raise ValueError("anchor feature schema mismatch")
    rows = dict(zip(u["material_ids"],u["groups"]))
    fit, val = u["fit_material_ids"], u["selection_material_ids"]
    if (not fit or not val or len(rows) != len(u["material_ids"]) or set(fit)&set(val)
            or set(fit+val) != set(rows) or [rows.get(i) for i in fit] != u["fit_groups"]
            or [rows.get(i) for i in val] != u["selection_groups"] or set(u["fit_groups"])&set(u["selection_groups"])
            or not isinstance(u["history"],dict) or not u["history"].get("epochs")
            or type(u["best_epoch"]) is not int or u["best_epoch"] < 1):
        raise ValueError("invalid upstream anchor allocation/history")


def batches(indices, size):
    if size < 2:
        raise ValueError("contrastive batch size must be >= 2")
    chunks = [indices[i:i+size] for i in range(0, len(indices), size)]
    if len(chunks) > 1 and len(chunks[-1]) == 1:
        chunks[-2:] = [np.concatenate((chunks[-2], chunks[-1]))]
    return chunks


def freeze_selection(out, manifest, sources=None):
    out = Path(out)
    manifest = {**manifest, "schema":SCHEMA, "status":"frozen"}
    validate_selection_metadata(manifest, out, manifest.get("framework"))
    write_json(out/"selection.json", manifest)
    write_json(out/"stage_completion.json", {"schema":SCHEMA, "status":"complete",
               "kind":"selection", "selection":file_ref(out/"selection.json"),
               "upstream":manifest["data"]}, sources=sources)
    return verify_selection(out, manifest["framework"])


def validate_ref(ref):
    if (not isinstance(ref, dict) or type(ref.get("bytes")) is not int or ref["bytes"] <= 0
            or not isinstance(ref.get("sha256"), str) or len(ref["sha256"]) != 64
            or any(ch not in "0123456789abcdef" for ch in ref["sha256"])):
        raise ValueError("invalid artifact bytes/SHA reference")


def validate_receipt_metadata(c, kind, split, test_scope):
    if (not isinstance(c, dict) or c.get("schema") != SCHEMA or c.get("status") != "complete"
            or c.get("kind") != kind or c.get("split") != split or type(c.get("test_scope")) is not bool
            or (c["test_scope"] and not test_scope)):
        raise ValueError("invalid upstream completion/scope contract")
    validate_ref(c.get("artifact"))
    check_code_tree(c, read=False)
    check_code(c.get("code"), kind)
    if not isinstance(c.get("sources"), dict) or not c["sources"]:
        raise ValueError("missing source contract")
    for ref in c["sources"].values():
        validate_ref(ref)
    identity = c.get("identity", {})
    keys = ("material_ids", "valid", "valid_idx", "valid_material_ids", "groups")
    if not all(k in identity for k in keys):
        raise ValueError("missing upstream row identity")
    validate_identity(*(np.asarray(identity[k]) for k in keys))
    if c.get("feature_semantics") != FEATURE_SEMANTICS:
        raise ValueError("upstream feature semantics mismatch")
    if kind == "pairs":
        if c.get("input_space") != "raw_canonical_6d":
            raise ValueError("raw pair contract required")
        raw = c.get("raw_contract", {})
        schema = c.get("feature_schema", {})
        if (raw.get("kind") != "raw_tensor" or raw.get("split") != split
                or raw.get("scope") != ("test" if c["test_scope"] else "formal")
                or raw.get("input_space") != "raw_canonical_6d"
                or raw.get("raw_semantics") != upstream.P2_RAW_SEMANTICS
                or type(schema.get("seq_len")) is not int
                or schema != upstream.p2_feature_schema(schema["seq_len"])
                or raw.get("feature_schema") != schema
                or raw.get("material_ids") != identity["material_ids"] or raw.get("groups") != identity["groups"]):
            raise ValueError("frozen raw source space/schema/identity mismatch")
        for key, source in (("tensor_ref","tensor"),("contract_ref","tensor_contract")):
            validate_ref(raw.get(key))
            if {k:raw[key][k] for k in ("bytes","sha256")} != c["sources"].get(source):
                raise ValueError("frozen raw source reference mismatch")
    else:
        if kind == "equivariant_graphs":
            from scripts.prepare_p2_equivariant_graphs import GRAPH_SCHEMA_VERSION
            if c.get("graph_schema") != GRAPH_SCHEMA_VERSION:
                raise ValueError("frozen graph schema mismatch")
        elif kind == "band_embeddings":
            if c.get("input_space") != "frozen_mbm_pooled" or (split == "train" and c.get("frozen_run") is not None):
                raise ValueError("frozen embedding feature space/stage mismatch")
            validate_anchor_metadata(c.get("anchor",{}), test_scope)
        validate_receipt_metadata(c.get("parent"), "pairs", split, test_scope)
        if c["sources"].get("pairs") != c["parent"]["artifact"] or identity != c["parent"]["identity"]:
            raise ValueError("upstream pair identity mismatch")


def validate_selection_metadata(m, out, framework):
    """Validate the ENTIRE prior without data or model deserialization."""
    required = {"schema", "status", "framework", "test_scope", "run_id", "config", "anchor", "data",
                "code", "inner", "history", "history_ref", "best_epoch", "checkpoints", "runtime", "initial_state_sha256"}
    if not isinstance(m, dict) or not required <= m.keys():
        raise ValueError("incomplete frozen selection contract")
    if (framework not in ("tensorflow", "torch") or m["framework"] != framework
            or m["schema"] != SCHEMA or m["status"] != "frozen" or type(m["test_scope"]) is not bool
            or not isinstance(m["run_id"], str) or len(m["run_id"]) != 32):
        raise ValueError("invalid frozen selection identity/scope")
    check_code(m["code"], framework, read=False)
    check_code_tree(m)
    scope, cfg, runtime = m["test_scope"], m["config"], m["runtime"]
    if (not isinstance(cfg, dict) or not isinstance(runtime, dict)
            or runtime.get("scope") != ("cpu_test" if scope else "formal_gpu")
            or runtime.get("device") != (("/CPU:0" if scope else "/GPU:0") if framework == "tensorflow" else ("cpu" if scope else "cuda"))):
        raise ValueError("frozen runtime/scope mismatch")
    fixed = {"temperature":.07, "embedding_dim":128, "num_elements":109,
             "loss":"one_way_structure_to_band_InfoNCE", "selection":"complete_inner_gallery_InfoNCE"}
    fixed.update({"hidden_dim":128,"conv_layers":3} if framework == "tensorflow" else
                 {"multiplicity":32,"num_layers":3,"lmax":2,"descriptor_dim":22,"pooling_version":"irreps-vector-norm-v3"})
    if any(cfg.get(k) != v for k,v in fixed.items()):
        raise ValueError("frozen model/loss config mismatch")
    if (type(cfg.get("epochs")) is not int or cfg["epochs"] < 1 or (scope and cfg["epochs"] > 2)
            or type(cfg.get("batch_size")) is not int or cfg["batch_size"] < 2):
        raise ValueError("invalid epochs/batch size")
    if framework == "tensorflow" and (type(cfg.get("max_atoms")) is not int or not 1 <= cfg["max_atoms"] <= 50):
        raise ValueError("invalid model atom capacity")
    anchor = m["anchor"]
    if (not isinstance(anchor, dict) or anchor.get("feature_semantics") != FEATURE_SEMANTICS
            or anchor.get("feature_mode") != "SSLEncoder.return_features.pooled"):
        raise ValueError("incomplete anchor contract")
    for key in ("encoder", "norm"):
        validate_ref(anchor.get(key))
    validate_anchor_metadata(anchor, scope)
    data = m["data"]
    expected = {"pairs":"pairs"} if framework == "tensorflow" else {"graphs":"equivariant_graphs", "embeddings":"band_embeddings"}
    if not isinstance(data, dict) or set(data) != set(expected):
        raise ValueError("incomplete frozen upstream data contract")
    for key, kind in expected.items():
        validate_receipt_metadata(data[key], kind, "train", scope)
    if framework == "torch" and (data["embeddings"]["anchor"] != anchor
            or data["embeddings"]["parent"] != data["graphs"]["parent"]
            or data["embeddings"]["sources"]["pairs_completion"] != data["graphs"]["sources"]["pairs_completion"]):
        raise ValueError("frozen graph/embedding parent or anchor mismatch")
    identity = next(iter(data.values()))["identity"]
    if any(c["identity"] != identity for c in data.values()):
        raise ValueError("frozen data row identity mismatch")
    raw_contract = data["pairs"]["raw_contract"] if framework == "tensorflow" else data["graphs"]["parent"]["raw_contract"]
    _, _, expected_inner = inner_partition(np.asarray(identity["material_ids"]), np.asarray(identity["groups"]),
        np.asarray(identity["valid_idx"]), anchor=anchor, raw_contract=raw_contract)
    if m["inner"] != expected_inner:
        raise ValueError("frozen inner allocation differs from upstream selection")
    inner = m["inner"]
    if not isinstance(inner, dict) or any(inner.get(k) != identity[k] for k in ("material_ids", "groups")):
        raise ValueError("frozen inner pool differs from data")
    ids, groups = np.asarray(inner["material_ids"]), np.asarray(inner["groups"])
    if scope and len(ids) > 32:
        raise ValueError("test scope limited to 32 raw rows")
    fit, val = np.asarray(inner.get("fit_indices")), np.asarray(inner.get("validation_indices"))
    if (fit.ndim != 1 or val.ndim != 1 or fit.dtype.kind not in "iu" or val.dtype.kind not in "iu"
            or min(len(fit),len(val)) < 2 or sorted(np.concatenate((fit,val)).tolist()) != identity["valid_idx"]
            or set(groups[fit]) & set(groups[val]) or inner.get("group_overlap") != 0
            or inner.get("fit_ids") != ids[fit].tolist() or inner.get("validation_ids") != ids[val].tolist()):
        raise ValueError("invalid inner fit/selection coverage or overlap")
    history = m["history"]
    if not isinstance(history, list) or len(history) != cfg["epochs"]:
        raise ValueError("missing/empty training history")
    for epoch, row in enumerate(history, 1):
        if (not isinstance(row,dict) or row.get("epoch") != epoch or row.get("fit_n") != len(fit)
                or row.get("validation_n") != len(val)
                or any(type(row.get(k)) not in (float,int) or not np.isfinite(row[k]) or row[k] < 0
                       for k in ("loss","val_loss","seconds"))):
            raise ValueError("invalid complete-inner training history")
    if type(m["best_epoch"]) is not int or m["best_epoch"] != min(history,key=lambda r:r["val_loss"])["epoch"]:
        raise ValueError("best_epoch does not select the complete inner minimum")
    observed_history, history_ref = read_json_snapshot(out/"history.json", m["history_ref"])
    if observed_history != history:
        raise ValueError("history SHA/content mismatch")
    m["history_ref"] = history_ref
    checkpoints = m["checkpoints"]
    if not isinstance(checkpoints,dict) or set(checkpoints) != {"best","last","accepted"}:
        raise ValueError("missing checkpoint states")
    for key, item in checkpoints.items():
        filename = key + (".weights.h5" if framework == "tensorflow" else ".pt")
        if (not isinstance(item,dict) or item.get("file") != filename
                or not isinstance(item.get("state_sha256"),str) or len(item["state_sha256"]) != 64):
            raise ValueError("invalid checkpoint path/state contract")
        validate_ref(item.get("artifact"))
    if (checkpoints["best"]["state_sha256"] == m["initial_state_sha256"]
            or checkpoints["accepted"]["state_sha256"] != checkpoints["best"]["state_sha256"]):
        raise ValueError("untrained/random checkpoint or accepted state differs from best")


def verify_selection(out, framework=None):
    out = Path(out)
    try:
        completion, completion_ref = read_json_snapshot(out/"stage_completion.json")
        if (completion.get("schema") != SCHEMA or completion.get("status") != "complete"
                or completion.get("kind") != "selection"
                or completion.get("selection") != file_ref(out/"selection.json")):
            raise ValueError("frozen selection SHA/completion mismatch")
        m, selection_ref = read_json_snapshot(out/"selection.json", completion["selection"])
        if framework is None:
            framework = m.get("framework")
        if (m.get("schema") != SCHEMA or m.get("status") != "frozen"
                or m.get("framework") != framework or completion["upstream"] != m["data"]):
            raise ValueError("frozen selection identity mismatch")
        validate_selection_metadata(m, out, framework)
        sources = {out/"stage_completion.json":completion_ref, out/"selection.json":selection_ref,
                   out/"history.json":m["history_ref"]}
        for key in ("best", "last", "accepted"):
            item = m["checkpoints"][key]
            actual = file_ref(out/item["file"])
            if Path(item["file"]).name != item["file"] or actual != item["artifact"]:
                raise ValueError(f"frozen checkpoint SHA mismatch: {key}")
            item["artifact"] = actual
            sources[out/item["file"]] = actual
        if m["checkpoints"]["best"]["artifact"] != m["checkpoints"]["accepted"]["artifact"]:
            raise ValueError("accepted checkpoint is not restored best")
        verify_sources(sources)
        return SnapshotMetadata(m, sources)
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("missing/corrupt frozen selection; legacy diagnostic only") from exc


def reject_outer_overlap(ids, groups, manifest):
    # Compare ALL raw rows, including invalid/excluded rows, with the full pool.
    seen = manifest["inner"]
    if set(ids.tolist()) & set(seen["material_ids"]) or set(groups.tolist()) & set(seen["groups"]):
        raise ValueError("outer ID/group overlap with frozen fit-plus-selection pool")


def frozen_run_ref(run_dir, manifest):
    return {"kind":"frozen_run_reference", "run_id":manifest["run_id"], "framework":manifest["framework"],
            "selection":source_refs(manifest)[Path(run_dir)/"selection.json"]}


def retrieval_evidence(structure, band, ids, chunk_size=128):
    """Exact full-gallery ranks with bounded similarity memory and stable ties."""
    if (structure.ndim != 2 or structure.shape != band.shape or len(ids) != len(band)
            or not len(ids) or not np.isfinite(structure).all() or not np.isfinite(band).all()):
        raise ValueError("invalid retrieval embeddings/IDs")
    if type(chunk_size) is not int or chunk_size < 1 or structure.shape[1] < 1:
        raise ValueError("positive chunk size and embedding dimension required")
    def unit_rows(x):
        if x.dtype.kind not in "fiu":
            raise ValueError("real embeddings required")
        x = x.astype(np.float64)
        scale = np.max(np.abs(x), axis=1, keepdims=True)
        if np.any(scale == 0):
            raise ValueError("zero embedding has no cosine norm")
        # Scale before squaring; float64 alone still overflows at 1e300.
        with np.errstate(under="ignore"):
            scaled = x / scale
            norm = np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True))
            value = scaled / norm
        if not np.isfinite(norm).all() or not np.isfinite(value).all():
            raise ValueError("nonfinite normalized embeddings")
        return value
    s, b = unit_rows(structure), unit_rows(band)
    result = {}
    n = len(ids)
    for name, query, gallery in (("structure_to_band",s,b), ("band_to_structure",b,s)):
        ranks, positives = [], []
        for start in range(0, n, chunk_size):
            sim = query[start:start+chunk_size] @ gallery.T
            if not np.isfinite(sim).all():
                raise ValueError("nonfinite retrieval similarities")
            order = np.argsort(-sim, axis=1, kind="stable")
            truth = np.arange(start, start+len(sim))
            ranks.extend((np.argmax(order == truth[:,None], axis=1)+1).tolist())
            positives.extend(sim[np.arange(len(sim)), truth].tolist())
        r = np.asarray(ranks)
        mrr = float(np.mean(1/r))
        result[name] = {"query_n":n, "gallery_n":n, "query_ids":ids.tolist(), "gallery_ids":ids.tolist(),
                        "ranks":ranks, "positive_cosine":positives, "tie_break":"original gallery order",
                        "map_interpretation":"single positive per query: AP=reciprocal rank; mAP=MRR, not multi-relevant-material mAP",
                        "metrics":{**{f"recall@{k}":float(np.mean(r <= k)) for k in (1,5,10)},
                                   "mrr":mrr, "map":mrr, "median_rank":float(np.median(r)),
                                   "mean_rank":float(np.mean(r))}}
    return result


def publish_evaluation(out, report, s, b, ids, sources=None):
    path = Path(out)/"predictions.npz"
    with path.open("xb") as f:
        np.savez_compressed(f, material_ids=ids, structure_embeddings=s, band_embeddings=b,
                            structure_to_band_ranks=np.asarray(report["retrieval"]["structure_to_band"]["ranks"]),
                            band_to_structure_ranks=np.asarray(report["retrieval"]["band_to_structure"]["ranks"]))
    report["predictions"] = file_ref(path)
    write_json(Path(out)/"report.json", report)
    verify_sources(sources or {})
    write_json(Path(out)/"stage_completion.json", {"schema":SCHEMA, "status":"complete", "kind":"evaluation",
               "selection":report["selection"], "data":report["data"], "report":file_ref(Path(out)/"report.json"),
               "predictions":report["predictions"]}, sources=sources)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar")
    parser.add_argument("--tensor-npz")
    parser.add_argument("--tensor-contract", help="upstream split-local raw tensor contract")
    parser.add_argument("--equivariant", action="store_true", help="wrap the existing periodic graph builder for one credentialed split")
    parser.add_argument("--pairs", help="upstream single-split pair NPZ for --equivariant")
    parser.add_argument("--out-dir")
    parser.add_argument("--split", choices=("train", "test"))
    parser.add_argument("--max-atoms", type=int, default=50)
    parser.add_argument("--test-scope", action="store_true")
    parser.add_argument("--legacy-diagnostic", metavar="NPZ")
    args = parser.parse_args(argv)
    if args.legacy_diagnostic:
        with np.load(args.legacy_diagnostic, allow_pickle=False) as f:
            print(json.dumps({"status":"legacy_diagnostic_not_accepted", "keys": f.files,
                              "artifact":file_ref(args.legacy_diagnostic)}))
        return
    if args.equivariant:
        if not all((args.sidecar, args.pairs, args.out_dir, args.split)) or args.tensor_npz:
            parser.error("--equivariant requires --sidecar --pairs --out-dir --split (no --tensor-npz)")
        print(prepare_equivariant_split(args.pairs, args.sidecar, args.out_dir, args.split,
                                       args.max_atoms, args.test_scope))
        return
    if not all((args.sidecar, args.tensor_npz, args.tensor_contract, args.out_dir, args.split)):
        parser.error("--sidecar --tensor-npz --tensor-contract --out-dir --split are required")
    print(prepare_split(args.tensor_npz, args.sidecar, args.out_dir, args.split,
                        args.max_atoms, args.test_scope, tensor_contract=args.tensor_contract))


if __name__ == "__main__":
    main()
