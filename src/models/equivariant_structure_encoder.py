"""Equivariant crystal graph encoder (PyTorch + e3nn), versioned P2 pooling.

The default v3 pool includes every component of each 1o irrep. Its invariance
holds for a fixed selected graph; rebuilding a capped graph may change tied-shell
selection. Explicit legacy-component-slice-v2 mode preserves historical tensor
layouts and pooling for diagnostics only, NOT new invariant-model acceptance.
New pool semantics require new training; unversioned weights cannot silently
load into v3. This module does not establish scientific retrieval performance.

BATCHING:
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
    POOLING_VERSION = "irreps-vector-norm-v3"
    LEGACY_POOLING_VERSION = "legacy-component-slice-v2"

    def __init__(
        self,
        num_elements: int = 109,
        multiplicity: int = 32,
        num_layers: int = 3,
        lmax: int = 2,
        embedding_dim: int = 128,
        descriptor_dim: int = 22,
        pooling_version: str = POOLING_VERSION,
    ):
        super().__init__()
        if pooling_version not in (self.POOLING_VERSION, self.LEGACY_POOLING_VERSION):
            raise ValueError("unknown pooling_version")
        self.pooling_version = pooling_version
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

    def get_extra_state(self):
        return {"schema_version": "p2-e3nn-encoder-v3", "pooling_version": self.pooling_version}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("encoder pooling/version mismatch; retraining or explicit legacy mode required")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + "_extra_state"
        if key not in state_dict:
            if self.pooling_version != self.LEGACY_POOLING_VERSION:
                raise RuntimeError("unversioned legacy weights require pooling_version='legacy-component-slice-v2'; retrain for v3")
            state_dict[key] = self.get_extra_state()
        self.set_extra_state(state_dict[key])
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def _validate_graph(self, graph, descriptors):
        n_atoms = graph["n_atoms"]
        for key in ("n_atoms", "graph_offsets", "edge_src", "edge_dst"):
            if graph[key].ndim != 1 or graph[key].dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{key} must have 1D integer shape")
        batch = n_atoms.numel()
        edges = graph["edge_src"].numel()
        if batch == 0:
            raise ValueError("empty graph batch is not supported")
        if (graph["graph_offsets"].shape != (batch,)
                or descriptors.shape != (batch, self.descriptor_dim)
                or graph["atom_features"].ndim != 2
                or graph["atom_features"].shape[1] != self.num_elements + 1
                or graph["edge_dst"].shape != (edges,)
                or graph["edge_vec"].shape != (edges, 3)
                or graph["edge_len"].shape != (edges,)):
            raise ValueError("graph/descriptor tensor shape mismatch")
        if torch.any(n_atoms < 1) or torch.any(n_atoms > 50):
            raise ValueError("n_atoms must be in [1, 50]; no capacity truncation")
        for key in ("atom_features", "edge_vec", "edge_len"):
            if not torch.isfinite(graph[key]).all():
                raise ValueError(f"{key} must be finite")
        if not torch.isfinite(descriptors).all():
            raise ValueError("descriptors must be finite")
        present = graph["atom_features"].abs().sum(dim=-1) > 0
        if int(present.sum()) != int(n_atoms.sum()):
            raise ValueError("n_atoms does not match the real atom count")
        owners = torch.full_like(present, -1, dtype=torch.long)
        total = present.numel()
        for b, (offset, count) in enumerate(zip(graph["graph_offsets"], n_atoms)):
            lo, hi = int(offset), int(offset) + int(count)
            if lo < 0 or hi > total or torch.any(owners[lo:hi] >= 0):
                raise ValueError("graph_offsets contain out-of-range or overlapping graph ranges")
            owners[lo:hi] = b
        if not torch.equal(owners >= 0, present):
            raise ValueError("graph_offsets/n_atoms do not match real versus padding rows")
        src, dst = graph["edge_src"], graph["edge_dst"]
        if torch.any(src < 0) or torch.any(dst < 0) or torch.any(src >= total) or torch.any(dst >= total):
            raise ValueError("edge index is outside the merged graph")
        if torch.any(owners[src] < 0) or torch.any(owners[dst] < 0) or torch.any(owners[src] != owners[dst]):
            raise ValueError("edge touches padding or crosses graph IDs")
        if torch.any(torch.bincount(src.long(), minlength=total) > 12):
            raise ValueError("at most 12 selected neighbors per source atom are supported")
        length = graph["edge_len"]
        if (torch.any(length <= 0) or torch.any(length > 8.0 + 1e-5)
                or not torch.allclose(length, torch.linalg.vector_norm(graph["edge_vec"], dim=-1),
                                      rtol=1e-5, atol=1e-6)):
            raise ValueError("edge_len must match edge_vec norm and lie in (0, 8 Angstrom]")

    def forward(self, graph: Dict[str, torch.Tensor], descriptors: torch.Tensor):
        self._validate_graph(graph, descriptors)
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

        # Each 1o multiplicity owns THREE components. Extract whole irreps,
        # rather than treating multiplicity as the flattened vector dimension.
        blocks = list(zip(self.node_irreps, self.node_irreps.slices()))
        scalar_part = torch.cat([node_feats[:, sl] for (_, ir), sl in blocks if ir.l == 0], dim=-1)
        vector_part = torch.cat([
            node_feats[:, sl].reshape(node_feats.shape[0], mul, ir.dim)
            for (mul, ir), sl in blocks if ir.l == 1
        ], dim=1)
        atom_present = (atom_features.sum(dim=-1) > 0).float()  # (Ntot,)

        B = n_atoms.shape[0]
        scalar_pool, vec_norm_pool = [], []
        for b in range(B):
            lo, hi = int(offsets[b]), int(offsets[b]) + int(n_atoms[b])
            mask = atom_present[lo:hi].unsqueeze(-1)
            denom = mask.sum() + 1e-8
            scalar_pool.append((scalar_part[lo:hi] * mask).sum(0) / denom)
            if self.pooling_version == self.LEGACY_POOLING_VERSION:
                # Explicit diagnostic compatibility only: this historical slice
                # is NOT rotation invariant and must never be labelled v3.
                old_vectors = node_feats[lo:hi, self.multiplicity:2 * self.multiplicity]
                vec_mean = (old_vectors * mask).sum(0) / denom
                vec_norm_pool.append(vec_mean.norm().unsqueeze(0))
            else:
                vec_mean = (vector_part[lo:hi] * mask.unsqueeze(-1)).sum(0) / denom
                per_vector_norm = torch.linalg.vector_norm(vec_mean, dim=-1)
                # Frobenius norm across channels preserves the old head shape.
                vec_norm_pool.append(torch.linalg.vector_norm(per_vector_norm).unsqueeze(0))
        scalar_pool = torch.stack(scalar_pool, 0)
        vec_norm_pool = torch.stack(vec_norm_pool, 0)

        pooled = torch.cat([scalar_pool, vec_norm_pool, descriptors], dim=-1)
        return self.head(pooled)
