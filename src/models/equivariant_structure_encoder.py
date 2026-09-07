"""Equivariant crystal graph encoder (PyTorch + e3nn) for P2 v2.

Constitution 5.0 §8 P2: the v1 CGCNN encoder overfit (its one-hot + distance
representation could not distinguish structures across OOD space groups).
This replaces it with an O(3)-equivariant message-passing encoder built on
e3nn, plus the enhanced global descriptors (spacegroup / lattice / composition
statistics) that give structures distinguishing power.

The encoder runs in the PyTorch `nequip` env; the frozen TF band encoder's
embeddings are pre-computed and passed as numpy targets (cross-framework
bridge at the embedding level only).

Design (single-graph forward; the trainer batches by gradient accumulation):
- atom one-hot (num_elements+1) -> scalar embedding (multiplicity m);
- node features live in irreps `m x 0e + m x 1o` (scalar + one l=1 vector
  channel — a cheap but real O(3)-equivariant signal that captures directional
  structure the scalar-only CGCNN missed);
- N message-passing layers: for each edge, tensor-product the source node
  features with the spherical harmonics of the (minimum-image) edge vector,
  sum into the destination, then o3.Linear + BatchNorm + residual;
- mean-pool scalar features per graph, plus |mean vector| as one extra scalar;
- concat pooled features + global descriptors -> MLP -> 128-dim embedding.

`forward` takes {atom_features (1,N,ne+1), edge_src (E,), edge_dst (E,),
edge_vec (E,3), edge_len (E,), n_atoms (1,)} and returns (1, embedding_dim).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from e3nn import o3
from e3nn.nn import BatchNorm


def _node_irreps(m: int) -> o3.Irreps:
    return o3.Irreps(f"{m}x0e + {m}x1o")


class EquivariantConvLayer(nn.Module):
    def __init__(self, m: int, lmax: int = 2):
        super().__init__()
        self.m = m
        self.lmax = lmax
        self.node_irreps = _node_irreps(m)
        self.edge_sh_irreps = o3.Irreps.spherical_harmonics(lmax)
        # shared_weights=True gives each instruction learnable weights
        # (e3nn 0.6 raises without internal_weights on forward otherwise).
        self.tp = o3.FullyConnectedTensorProduct(
            self.node_irreps, self.edge_sh_irreps, self.node_irreps, shared_weights=True
        )
        self.linear = o3.Linear(self.node_irreps, self.node_irreps)
        self.norm = BatchNorm(self.node_irreps)

    def forward(self, node_feats, edge_src, edge_dst, edge_vec, edge_len):
        # edge_vec is already unit-ish; normalize defensively
        edge_vec_n = edge_vec / (edge_len.unsqueeze(-1) + 1e-8)
        edge_sh = o3.spherical_harmonics(
            self.edge_sh_irreps, edge_vec_n, normalize=True, normalization="component"
        )
        src_feats = node_feats[edge_src]
        messages = self.tp(src_feats, edge_sh)
        out = torch.zeros_like(node_feats)
        out.index_add_(0, edge_dst, messages)
        return self.norm(self.linear(out)) + node_feats


class EquivariantStructureEncoder(nn.Module):
    def __init__(
        self,
        num_elements: int = 109,
        multiplicity: int = 32,
        num_layers: int = 3,
        lmax: int = 2,
        embedding_dim: int = 128,
        descriptor_dim: int = 22,
    ):
        super().__init__()
        self.num_elements = num_elements
        self.multiplicity = multiplicity
        self.embedding_dim = embedding_dim
        self.descriptor_dim = descriptor_dim

        self.node_irreps = _node_irreps(multiplicity)
        self.atom_embedding = nn.Linear(num_elements + 1, multiplicity)

        self.conv_layers = nn.ModuleList(
            [EquivariantConvLayer(multiplicity, lmax) for _ in range(num_layers)]
        )

        head_in = multiplicity + 1 + descriptor_dim  # scalar pool + |vec| + desc
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, graph: Dict[str, torch.Tensor], descriptors: torch.Tensor):
        atom_features = graph["atom_features"]  # (1, N, ne+1) or (B, N, ne+1)
        edge_src = graph["edge_src"]
        edge_dst = graph["edge_dst"]
        edge_vec = graph["edge_vec"]
        edge_len = graph["edge_len"]
        n_atoms = graph["n_atoms"]

        B = atom_features.shape[0]
        N = atom_features.shape[1]
        flat_atoms = atom_features.reshape(-1, self.num_elements + 1)
        node_scalar = self.atom_embedding(flat_atoms)
        node_feats = torch.zeros(
            node_scalar.shape[0], self.node_irreps.dim,
            device=node_scalar.device, dtype=node_scalar.dtype,
        )
        node_feats[:, : self.multiplicity] = node_scalar

        for conv in self.conv_layers:
            node_feats = conv(node_feats, edge_src, edge_dst, edge_vec, edge_len)

        scalar_part = node_feats[:, : self.multiplicity]
        vector_part = node_feats[:, self.multiplicity: self.multiplicity * 2]
        atom_present = (flat_atoms.sum(dim=-1) > 0).float()

        scalar_pool, vec_norm_pool = [], []
        for b in range(B):
            lo, hi = b * N, b * N + int(n_atoms[b])
            mask = atom_present[lo:hi].unsqueeze(-1)
            denom = mask.sum() + 1e-8
            scalar_pool.append((scalar_part[lo:hi] * mask).sum(0) / denom)
            vec_mean = (vector_part[lo:hi] * mask).sum(0) / denom
            vec_norm_pool.append(vec_mean.norm().unsqueeze(0))
        scalar_pool = torch.stack(scalar_pool, 0)
        vec_norm_pool = torch.stack(vec_norm_pool, 0)

        pooled = torch.cat([scalar_pool, vec_norm_pool, descriptors], dim=-1)
        return self.head(pooled)
