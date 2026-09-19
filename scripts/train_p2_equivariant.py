"""P2 e3nn: one-way InfoNCE against credentialed frozen MBM embeddings.

Train-only reads one split, selects on the complete group-disjoint inner gallery,
then freezes best/last/accepted including BN buffers and pooling-version state.
Evaluation-only verifies that freeze before loading any outer rows. The existing
3-layer / multiplicity-32 / 128-output architecture and temperature .07 stay fixed.
Formal execution is GPU-only; bounded synthetic CPU work needs --test-scope.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.retrieval import bidirectional_metrics
from scripts import prepare_p2_pairs as contracts

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "equivariant_structure_encoder",
    os.path.join(os.path.dirname(__file__), "..", "src", "models", "equivariant_structure_encoder.py"),
)
_ese = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ese)
EquivariantStructureEncoder = _ese.EquivariantStructureEncoder


def info_nce(s_emb, b_emb, temperature):
    s_emb = F.normalize(s_emb, dim=-1)
    b_emb = F.normalize(b_emb, dim=-1)
    logits = s_emb @ b_emb.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


class GraphDataset:
    """Preloads one split's graph data into RAM (numpy) and builds merged
    mini-batches with vectorised edge-offset construction."""

    def __init__(self, d, device):
        self.device = device
        self.raw = d
        self.material_ids = d["material_ids"]
        self.raw_groups = d["groups"]
        self.N = int(d["atom_features"].shape[1])
        self.valid = d["valid"].astype(bool)
        self.idx = np.where(self.valid)[0]
        self.ids = self.material_ids[self.idx]

        self.atom_features = d["atom_features"][self.idx]  # (n, N, ne+1)
        self.n_atoms = d["n_atoms"][self.idx]
        self.descriptors = d["descriptors"][self.idx]

        # Edge data: per-graph slices into the flat edge arrays.
        self.edge_start = d["edge_offsets"][self.idx, 0].astype(np.int64)
        self.edge_end = d["edge_offsets"][self.idx, 1].astype(np.int64)
        self.edge_src = d["edge_src"].astype(np.int64)
        self.edge_dst = d["edge_dst"].astype(np.int64)
        self.edge_vec = d["edge_vec"].astype(np.float32)
        self.edge_len = d["edge_len"].astype(np.float32)

    def __len__(self):
        return len(self.idx)

    def collate(self, indices):
        """Merge graphs at `indices` (positions into self.idx) into one batched
        graph, with node/edge indices rebased to batch-local space."""
        B = len(indices)
        N = self.N
        atom = self.atom_features[indices].reshape(B * N, -1)

        lo = self.edge_start[indices]
        hi = self.edge_end[indices]
        seg_lens = hi - lo
        bases = np.repeat(np.arange(B, dtype=np.int64) * N, seg_lens)

        # Flatten per-graph edge slices into one array (single concatenate).
        flat_idx = np.concatenate(
            [np.arange(lo[j], hi[j], dtype=np.int64) for j in range(B)]
        ) if B else np.array([], dtype=np.int64)

        return {
            "atom_features": torch.from_numpy(atom).float().to(self.device),
            "n_atoms": torch.from_numpy(self.n_atoms[indices]).long().to(self.device),
            "graph_offsets": torch.from_numpy(np.arange(B, dtype=np.int64) * N).long().to(self.device),
            "edge_src": torch.from_numpy(self.edge_src[flat_idx] + bases).long().to(self.device),
            "edge_dst": torch.from_numpy(self.edge_dst[flat_idx] + bases).long().to(self.device),
            "edge_vec": torch.from_numpy(self.edge_vec[flat_idx]).float().to(self.device),
            "edge_len": torch.from_numpy(self.edge_len[flat_idx]).float().to(self.device),
            "descriptors": torch.from_numpy(self.descriptors[indices]).float().to(self.device),
        }


def load_split(graphs, band_emb, split, device, test_scope=False):
    g, gc = contracts.read_completed_npz(graphs, "equivariant_graphs", split, test_scope)
    b, bc = contracts.read_completed_npz(band_emb, "band_embeddings", split, test_scope)
    for a in (g, b):
        contracts.validate_identity(*(a[k] for k in ("material_ids", "valid", "valid_idx", "valid_material_ids", "groups")))
    for k in ("material_ids", "valid", "valid_idx", "valid_material_ids", "groups"):
        if not np.array_equal(g[k], b[k]):
            raise ValueError(f"graph/band identity or valid order mismatch: {k}")
    if gc["sources"]["pairs"] != bc["sources"]["pairs"] or gc["sources"]["pairs_completion"] != bc["sources"]["pairs_completion"]:
        raise ValueError("graphs/embeddings derive from different pair artifacts")
    if (bc.get("input_space") != "frozen_mbm_pooled" or bc.get("feature_semantics") != contracts.FEATURE_SEMANTICS
            or bc["anchor"].get("feature_mode") != "SSLEncoder.return_features.pooled"):
        raise ValueError("frozen MBM feature semantics mismatch")
    contracts.validate_anchor_metadata(bc["anchor"], test_scope)
    n, nv = len(g["material_ids"]), len(g["valid_idx"])
    if test_scope and n > 32:
        raise ValueError("test scope limited to 32 raw rows")
    if b["band_embeddings"].shape != (nv,128) or not np.isfinite(b["band_embeddings"]).all():
        raise ValueError("valid-only band embeddings shape/finiteness mismatch")
    contracts.validate_graphs(g, gc, b["material_ids"], b["valid"])
    return GraphDataset(g, device), b["band_embeddings"], {"graphs":gc, "embeddings":bc}


def torch_state_sha(model):
    import hashlib
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        if isinstance(value, torch.Tensor):
            a = value.detach().cpu().contiguous().numpy()
            h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
        else:
            h.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
    return h.hexdigest()


def encode_evaluation(model, ds, indices, batch_size, formal=False):
    if batch_size < 1 or not len(indices):
        raise ValueError("nonempty evaluation and positive batch size required")
    before = torch_state_sha(model)
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            g = ds.collate(indices[start:start+batch_size])
            value = model(g, g["descriptors"])
            if formal and value.device.type != "cuda":
                raise RuntimeError("GPU required at inference seam")
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite structure embeddings")
            result.append(value.cpu().numpy())
    if torch_state_sha(model) != before:
        raise ValueError("evaluation mutated frozen parameters or BN buffers")
    return np.concatenate(result)


def configure_runtime(test_scope=False, samples=None, epochs=1):
    if epochs < 1 or (samples is not None and samples < 1):
        raise ValueError("positive samples and epochs required")
    if test_scope and (epochs > 2 or (samples is not None and samples > 32)):
        raise ValueError("CPU test scope: at most 32 raw samples and 2 epochs")
    if not test_scope and not torch.cuda.is_available():
        raise RuntimeError("GPU required; no automatic CPU fallback")
    return torch.device("cpu" if test_scope else "cuda")


def require_gpu_tensor(tensor, formal):
    if formal and tensor.device.type != "cuda":
        raise RuntimeError("GPU required at forward/loss/gradient seam")
    if not torch.isfinite(tensor).all():
        raise ValueError("nonfinite forward/loss/gradient")


def full_validation_loss(s, b, batch_size, device, formal=False):
    with torch.no_grad():
        s = F.normalize(torch.as_tensor(s, device=device), dim=-1)
        b = F.normalize(torch.as_tensor(b, device=device), dim=-1)
        total = torch.zeros((), device=device)
        for start in range(0, len(s), batch_size):
            logits = s[start:start+batch_size] @ b.T / .07
            loss = F.cross_entropy(logits, torch.arange(start, min(start+batch_size,len(s)), device=device), reduction="sum")
            require_gpu_tensor(loss, formal)
            total += loss
        return float((total/len(s)).item())


def train_only(graphs, band_emb, output_dir, *, epochs=60, batch_size=64,
               validation_size=.2, seed=42, learning_rate=1e-3, test_scope=False,
               band_encoder=None, norm=None, anchor_contract=None):
    from pathlib import Path
    import platform
    import shutil
    import e3nn
    device = configure_runtime(test_scope, epochs=epochs)
    _, anchor = contracts.inspect_anchor(band_encoder,norm,anchor_contract=anchor_contract,test_scope=test_scope)
    out = contracts.fresh_directory(output_dir)
    ds, band, data = load_split(graphs, band_emb, "train", device, test_scope)
    if data["embeddings"]["anchor"] != anchor:
        raise ValueError("embedding anchor differs from verified upstream producer")
    configure_runtime(test_scope, len(ds.material_ids), epochs)
    fit_raw, val_raw, inner = contracts.inner_partition(ds.material_ids, ds.raw_groups, ds.idx, validation_size, seed,
        anchor=data["embeddings"]["anchor"], raw_contract=data["graphs"]["parent"]["raw_contract"])
    fit, val = np.searchsorted(ds.idx, fit_raw), np.searchsorted(ds.idx, val_raw)
    config = {"epochs":epochs, "batch_size":batch_size, "validation_size":validation_size, "seed":seed,
              "learning_rate":learning_rate, "temperature":.07, "multiplicity":32, "num_layers":3,
              "lmax":2, "embedding_dim":128, "num_elements":109, "descriptor_dim":22,
              "pooling_version":EquivariantStructureEncoder.POOLING_VERSION,
              "loss":"one_way_structure_to_band_InfoNCE", "selection":"complete_inner_gallery_InfoNCE"}
    root = Path(__file__).resolve().parents[1]
    code = contracts.code_refs(Path(__file__), root/"scripts/prepare_p2_pairs.py",
                               root/"src/models/equivariant_structure_encoder.py")
    torch.manual_seed(seed)
    model = EquivariantStructureEncoder(num_elements=109, multiplicity=32, num_layers=3, lmax=2, embedding_dim=128).to(device)
    initial_state = torch_state_sha(model)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    targets = torch.from_numpy(band).to(device)
    history, checkpoints = [], {}
    rng, best = np.random.default_rng(seed), float("inf")
    for epoch in range(1, epochs+1):
        started = time.monotonic()
        model.train()
        total = torch.zeros((), device=device)
        for idx in contracts.batches(rng.permutation(fit), batch_size):
            g = ds.collate(idx)
            opt.zero_grad(set_to_none=True)
            value = model(g, g["descriptors"])
            loss = info_nce(value, targets[idx], .07)
            require_gpu_tensor(value, not test_scope); require_gpu_tensor(loss, not test_scope)
            loss.backward()
            for param in model.parameters():
                if param.requires_grad:
                    if param.grad is None:
                        raise ValueError("missing structure gradient")
                    require_gpu_tensor(param.grad, not test_scope)
            opt.step()
            total += loss.detach() * len(idx)
        values = encode_evaluation(model, ds, val, batch_size, not test_scope)
        val_loss = full_validation_loss(values, band[val], batch_size, device, not test_scope)
        row = {"epoch":epoch, "loss":float((total/len(fit)).item()), "val_loss":val_loss,
               "fit_n":len(fit), "validation_n":len(val), "seconds":time.monotonic()-started}
        history.append(row)
        contracts.write_json(out/"history.json", history)
        print(row, flush=True)
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), out/"best.pt")
            checkpoints["best"] = {"file":"best.pt", "artifact":contracts.file_ref(out/"best.pt"),
                                   "state_sha256":torch_state_sha(model)}
    torch.save(model.state_dict(), out/"last.pt")
    checkpoints["last"] = {"file":"last.pt", "artifact":contracts.file_ref(out/"last.pt"), "state_sha256":torch_state_sha(model)}
    model.load_state_dict(torch.load(out/"best.pt", map_location=device, weights_only=True), strict=True)
    model.eval()
    if torch_state_sha(model) != checkpoints["best"]["state_sha256"]:
        raise ValueError("restored best including BN state differs")
    shutil.copyfile(out/"best.pt", out/"accepted.pt")
    checkpoints["accepted"] = {**checkpoints["best"], "file":"accepted.pt"}
    for path, item in ((graphs,data["graphs"]), (band_emb,data["embeddings"])):
        if contracts.file_ref(path) != item["artifact"]:
            raise ValueError("upstream data changed during training")
    manifest = {"framework":"torch", "test_scope":bool(test_scope), "config":config, "code":code,
                "run_id":contracts.uuid.uuid4().hex, "initial_state_sha256":initial_state,
                "history_ref":contracts.file_ref(out/"history.json"),
                "anchor":data["embeddings"]["anchor"], "data":data, "inner":inner, "history":history,
                "best_epoch":best_epoch, "checkpoints":checkpoints,
                "runtime":{"python":platform.python_version(), "torch":torch.__version__, "e3nn":e3nn.__version__,
                           "numpy":np.__version__, "device":str(device), "scope":"cpu_test" if test_scope else "formal_gpu"}}
    return contracts.freeze_selection(out, manifest, sources=contracts.source_refs(data["graphs"], data["embeddings"], anchor))


def load_frozen(run_dir, test_scope=False, *, band_encoder=None, norm=None, anchor_contract=None):
    from pathlib import Path
    m = contracts.verify_selection(run_dir, "torch")
    if m["test_scope"] and not test_scope:
        raise ValueError("test-scope checkpoint cannot authorize formal evaluation")
    device = configure_runtime(test_scope)
    _, anchor = contracts.inspect_anchor(band_encoder,norm,m["anchor"],anchor_contract=anchor_contract,test_scope=test_scope)
    m.snapshots.update(contracts.source_refs(anchor))
    model = EquivariantStructureEncoder(num_elements=109, multiplicity=32, num_layers=3, lmax=2, embedding_dim=128).to(device)
    item = m["checkpoints"]["accepted"]
    with contracts.snapshot(Path(run_dir)/item["file"], item["artifact"]) as (copy, _):
        model.load_state_dict(torch.load(copy, map_location=device, weights_only=True), strict=True)
    model.eval()
    model.requires_grad_(False)
    if torch_state_sha(model) != m["checkpoints"]["accepted"]["state_sha256"]:
        raise ValueError("reloaded full state/BN mismatch")
    return model, m


def evaluation_only(graphs, band_emb, run_dir, output_dir, *, batch_size=64, test_scope=False,
                    band_encoder=None, norm=None, anchor_contract=None):
    from pathlib import Path
    import platform
    import e3nn
    model, m = load_frozen(run_dir, test_scope, band_encoder=band_encoder, norm=norm, anchor_contract=anchor_contract)
    out = contracts.fresh_directory(output_dir)
    device = configure_runtime(test_scope)
    ds, band, data = load_split(graphs, band_emb, "test", device, test_scope)
    sources = contracts.source_refs(m, data["graphs"], data["embeddings"])
    if data["embeddings"].get("frozen_run") != contracts.frozen_run_ref(run_dir, m):
        raise ValueError("test embeddings were not extracted for this frozen selection/run")
    if data["embeddings"]["anchor"] != m["anchor"]:
        raise ValueError("frozen anchor identity mismatch")
    contracts.reject_outer_overlap(ds.material_ids, ds.raw_groups, m)
    before = torch_state_sha(model)
    values = encode_evaluation(model, ds, np.arange(len(ds)), batch_size, not test_scope)
    after = torch_state_sha(model)
    if before != after:
        raise ValueError("evaluation changed frozen model/BN state")
    contracts.verify_selection(run_dir, "torch")
    report = {"schema":contracts.SCHEMA, "kind":"evaluation", "test_scope":bool(test_scope),
              "selection":sources[Path(run_dir)/"selection.json"], "data":data,
              "state_before":before, "state_after":after, "anchor":m["anchor"], "config":m["config"],
              "runtime":{"python":platform.python_version(), "torch":torch.__version__, "e3nn":e3nn.__version__,
                         "device":str(device), "batch_size":batch_size},
              "retrieval":contracts.retrieval_evidence(values, band, ds.ids)}
    return contracts.publish_evaluation(out, report, values, band, ds.ids, sources=sources)


def main(argv=None):
    p = argparse.ArgumentParser(description="P2 one-split, GPU-only formal train/frozen evaluation")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train-only", action="store_true")
    mode.add_argument("--evaluation-only", action="store_true")
    p.add_argument("--graphs", required=True, help="single credentialed split NPZ, NOT a directory")
    p.add_argument("--band-emb", required=True, help="single credentialed embedding split NPZ")
    p.add_argument("--band-encoder", required=True)
    p.add_argument("--norm", required=True)
    p.add_argument("--anchor-contract", required=True)
    p.add_argument("--output-dir", "--out", dest="output_dir", required=True)
    p.add_argument("--run-dir")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--validation-size", type=float, default=.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-3)
    p.add_argument("--test-scope", action="store_true")
    a = p.parse_args(argv)
    if a.train_only:
        if a.run_dir:
            p.error("train-only forbids --run-dir; output must be new")
        train_only(a.graphs, a.band_emb, a.output_dir, epochs=a.epochs, batch_size=a.batch_size,
                   validation_size=a.validation_size, seed=a.seed, learning_rate=a.learning_rate,
                   test_scope=a.test_scope, band_encoder=a.band_encoder, norm=a.norm, anchor_contract=a.anchor_contract)
    else:
        if not a.run_dir:
            p.error("evaluation-only requires --run-dir")
        evaluation_only(a.graphs, a.band_emb, a.run_dir, a.output_dir, batch_size=a.batch_size,
                        test_scope=a.test_scope, band_encoder=a.band_encoder, norm=a.norm, anchor_contract=a.anchor_contract)


if __name__ == "__main__":
    main()
