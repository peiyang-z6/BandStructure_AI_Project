"""P2 v2: train equivariant structure encoder against frozen band embeddings.

Constitution 5.0 §8 P2. Runs in the PyTorch `nequip` env. Band embeddings
(from extract_band_embeddings.py) are the InfoNCE targets; equivariant graph
data + descriptors come from prepare_p2_equivariant_graphs.py (precomputed).

Batching: each mini-batch merges its graphs into ONE batched graph with
per-graph node offsets (torch_geometric-style), so message passing runs once
per batch instead of once per graph (the v1 single-graph loop was ~10 h/epoch).

Performance notes (v2, 2026-09-07):
- All graph data is preloaded into RAM ONCE (numpy), then sliced per batch;
  edge offsets for the merged graph are built with np.repeat/concatenate so
  the only Python loop is over batches, not over graphs.
- loss is accumulated as a detached tensor (no per-batch GPU->CPU sync).

Usage:
    python scripts/train_p2_equivariant.py \
        --graphs data/processed/aflow/ood_tensors_v7_60000_seed42/p2_pairs/equivariant_graphs \
        --band-emb data/processed/aflow/ood_tensors_v7_60000_seed42/p2_pairs/band_embeddings.npz \
        --out artifacts/models/aflow_noleak_v7_60k_seed42/p2_equivariant \
        [--epochs 60 --batch-size 64 --temperature 0.07 --lr 1e-3]
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

    def __init__(self, npz_path, device):
        d = np.load(npz_path)
        self.device = device
        self.N = int(d["atom_features"].shape[1])
        self.valid = d["valid"].astype(bool)
        self.idx = np.where(self.valid)[0]

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train P2 equivariant contrastive encoder")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--band-emb", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--multiplicity", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[P2v2] device {device}", flush=True)

    band_emb = np.load(args.band_emb)
    ds, band_t = {}, {}
    for split in ("train", "test"):
        ds[split] = GraphDataset(os.path.join(args.graphs, f"graphs_{split}.npz"), device)
        band_t[split] = torch.from_numpy(band_emb[f"band_emb_{split}"]).float().to(device)
        print(f"[P2v2] {split}: {len(ds[split])} valid, band dim {band_t[split].shape}", flush=True)

    encoder = EquivariantStructureEncoder(
        num_elements=109, multiplicity=args.multiplicity,
        num_layers=args.num_layers, lmax=2, embedding_dim=128,
    ).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    ds_tr = ds["train"]
    n = len(ds_tr)
    history = []
    for epoch in range(args.epochs):
        perm = torch.randperm(n).numpy()
        t0 = time.time()
        epoch_loss = torch.tensor(0.0, device=device)
        nb = 0
        for start in range(0, n, args.batch_size):
            bidx = perm[start : start + args.batch_size]
            g = ds_tr.collate(bidx)
            b_emb = band_t["train"][bidx]
            opt.zero_grad()
            s_emb = encoder(g, g["descriptors"])
            loss = info_nce(s_emb, b_emb, args.temperature)
            loss.backward()
            opt.step()
            epoch_loss = epoch_loss + loss.detach()
            nb += 1
        avg = float(epoch_loss.item()) / max(nb, 1)
        history.append({"epoch": epoch + 1, "loss": avg})
        print(f"[P2v2] epoch {epoch+1}/{args.epochs} loss {avg:.4f} ({time.time()-t0:.1f}s)", flush=True)

    @torch.no_grad()
    def encode(dsplit):
        out = []
        for start in range(0, len(dsplit), args.batch_size):
            bidx = np.arange(start, min(start + args.batch_size, len(dsplit)))
            g = dsplit.collate(bidx)
            out.append(encoder(g, g["descriptors"]).cpu().numpy())
        return np.concatenate(out, axis=0)

    os.makedirs(args.out, exist_ok=True)
    report = {}
    for split in ("train", "test"):
        s_emb = encode(ds[split])
        report[f"{split}_retrieval"] = bidirectional_metrics(s_emb, band_t[split].cpu().numpy())
    print(f"[P2v2] train retrieval: {json.dumps(report['train_retrieval'], indent=2)}", flush=True)
    print(f"[P2v2] test retrieval: {json.dumps(report['test_retrieval'], indent=2)}", flush=True)

    torch.save(encoder.state_dict(), os.path.join(args.out, "structure_encoder.pt"))
    report["history"] = history
    with open(os.path.join(args.out, "p2_equivariant_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"[P2v2] report -> {os.path.join(args.out, 'p2_equivariant_report.json')}", flush=True)


if __name__ == "__main__":
    main()
