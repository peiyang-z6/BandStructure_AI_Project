"""P2 v2: train equivariant structure encoder against frozen band embeddings.

Constitution 5.0 §8 P2. Runs in the PyTorch `nequip` env. Band embeddings
(from extract_band_embeddings.py) are the InfoNCE targets; equivariant graph
data + descriptors come from prepare_p2_equivariant_graphs.py (precomputed).

Cross-framework bridge: band embeddings are precomputed numpy; this script
only uses torch.

Training is single-graph forward with gradient accumulation: the encoder's
edges are intra-graph indices, so each graph is forwarded alone and gradients
accumulate over `batch_size` graphs before one optimizer step. This keeps the
InfoNCE batch (the contrastive negatives) at `batch_size` while avoiding
cross-graph edge-index bookkeeping.

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

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.retrieval import bidirectional_metrics

# Import the equivariant encoder WITHOUT triggering src.models.__init__ (TF).
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


def single_graph(graph, i):
    """Return a single-graph dict for graph index i (intra-graph edges)."""
    na = int(graph["n_atoms"][i])
    # edge offsets: consecutive per-graph edge blocks
    lo, hi = int(graph["edge_offsets"][i][0]), int(graph["edge_offsets"][i][1])
    return {
        "atom_features": graph["atom_features"][i:i + 1],
        "n_atoms": graph["n_atoms"][i:i + 1],
        "edge_src": graph["edge_src"][lo:hi],
        "edge_dst": graph["edge_dst"][lo:hi],
        "edge_vec": graph["edge_vec"][lo:hi],
        "edge_len": graph["edge_len"][lo:hi],
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
    graphs, band_t = {}, {}
    for split in ("train", "test"):
        gz = np.load(os.path.join(args.graphs, f"graphs_{split}.npz"))
        g = {k: torch.from_numpy(gz[k]).to(device) if k in
             ("atom_features", "n_atoms", "edge_src", "edge_dst", "edge_vec", "edge_len")
             else (torch.from_numpy(gz[k]).to(device) if k == "descriptors" else gz[k])
             for k in gz.keys()}
        graphs[split] = g
        band_t[split] = torch.from_numpy(band_emb[f"band_emb_{split}"]).float().to(device)
        print(f"[P2v2] {split}: {int(gz['valid'].sum())} valid, band dim {band_t[split].shape}", flush=True)

    encoder = EquivariantStructureEncoder(
        num_elements=109, multiplicity=args.multiplicity,
        num_layers=args.num_layers, lmax=2, embedding_dim=128,
    ).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    tr = graphs["train"]
    valid_idx = np.where(tr["valid"])[0]
    n = len(valid_idx)

    history = []
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        epoch_loss, nb = 0.0, 0
        # accumulate contrastive batch
        s_batch, b_batch = [], []
        opt.zero_grad()
        for t, pos in enumerate(perm.numpy()):
            i = valid_idx[pos]
            g = single_graph(tr, i)
            desc = tr["descriptors"][i:i + 1]
            s_emb = encoder(g, desc)
            s_batch.append(s_emb)
            b_batch.append(band_t["train"][i:i + 1])
            if len(s_batch) == args.batch_size or t == n - 1:
                s = torch.cat(s_batch, 0)
                b = torch.cat(b_batch, 0)
                loss = info_nce(s, b, args.temperature) / args.batch_size
                loss.backward()
                opt.step()
                opt.zero_grad()
                epoch_loss += float(loss) * args.batch_size
                nb += 1
                s_batch, b_batch = [], []
        avg = epoch_loss / max(n, 1)
        history.append({"epoch": epoch + 1, "loss": avg})
        if (epoch + 1) % 5 == 0:
            print(f"[P2v2] epoch {epoch+1}/{args.epochs} loss {avg:.4f}", flush=True)

    @torch.no_grad()
    def encode(graph):
        idx = np.where(graph["valid"])[0]
        out = []
        for i in idx:
            g = single_graph(graph, i)
            out.append(encoder(g, graph["descriptors"][i:i + 1]).cpu().numpy())
        return np.concatenate(out, axis=0)

    os.makedirs(args.out, exist_ok=True)
    report = {}
    for split in ("train", "test"):
        s_emb = encode(graphs[split])
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
