"""Equivariant crystal graph encoder (PyTorch + e3nn) for P2 v2 — batched.

Constitution 5.0 §8 P2. O(3)-equivariant message passing on e3nn, plus global
descriptors (spacegroup / lattice / composition stats) fused into the pooled
features. Replaces the v1 CGCNN encoder that overfit (its one-hot + distance
representation could not distinguish structures across OOD space groups).

BATCHING (the v1 single-graph loop was ~10 h/epoch; this fixes it):
the caller merges a mini-batch of graphs into ONE graph with per-graph node
offsets. Node features are (sum_n_atoms_padded, ne+1); edge src/dst are
absolute indices into that merged node tensor. Message passing runs once over
the whole batch; pooling uses per-graph node ranges. This is the standard
GNN mini-batching scheme (torch_geometric-style) applied to e3nn.

`forward` takes:
  atom_features (Ntot, ne+1)  — merged node one-hots (padded rows zeroed)
  edge_src/edge_dst (E,)     — absolute merged indices
  edge_vec (E,3), edge_len (E,)
  n_atoms (B,)               — per-graph real atom counts
  graph_offsets (B,)         — start node index of each graph in the merged tensor
and returns (B, embedding_dim).
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
        self.tp = o3.FullyConnectedTensorProduct(
            self.node_irreps, self.edge_sh_irreps, self.node_irreps, shared_weights=True
        )
        self.linear = o3.Linear(self.node_irreps, self.node_irreps)
        self.norm = BatchNorm(self.node_irreps)

    def forward(self, node_feats, edge_src, edge_dst, edge_vec, edge_len):
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

        head_in = multiplicity + 1 + descriptor_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, graph: Dict[str, torch.Tensor], descriptors: torch.Tensor):
        atom_features = graph["atom_features"]  # (Ntot, ne+1)
        edge_src = graph["edge_src"]
        edge_dst = graph["edge_dst"]
        edge_vec = graph["edge_vec"]
        edge_len = graph["edge_len"]
        n_atoms = graph["n_atoms"]  # (B,)
        offsets = graph["graph_offsets"]  # (B,) start node index per graph

        node_scalar = self.atom_embedding(atom_features)  # (Ntot, m)
        node_feats = torch.zeros(
            node_scalar.shape[0], self.node_irreps.dim,
            device=node_scalar.device, dtype=node_scalar.dtype,
        )
        node_feats[:, : self.multiplicity] = node_scalar

        for conv in self.conv_layers:
            node_feats = conv(node_feats, edge_src, edge_dst, edge_vec, edge_len)

        scalar_part = node_feats[:, : self.multiplicity]
        vector_part = node_feats[:, self.multiplicity: self.multiplicity * 2]
        atom_present = (atom_features.sum(dim=-1) > 0).float()  # (Ntot,)

        B = n_atoms.shape[0]
        scalar_pool, vec_norm_pool = [], []
        for b in range(B):
            lo, hi = int(offsets[b]), int(offsets[b]) + int(n_atoms[b])
            mask = atom_present[lo:hi].unsqueeze(-1)
            denom = mask.sum() + 1e-8
            scalar_pool.append((scalar_part[lo:hi] * mask).sum(0) / denom)
            vec_mean = (vector_part[lo:hi] * mask).sum(0) / denom
            vec_norm_pool.append(vec_mean.norm().unsqueeze(0))
        scalar_pool = torch.stack(scalar_pool, 0)
        vec_norm_pool = torch.stack(vec_norm_pool, 0)

        pooled = torch.cat([scalar_pool, vec_norm_pool, descriptors], dim=-1)
        return self.head(pooled)
