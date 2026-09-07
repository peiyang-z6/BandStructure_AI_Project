"""Minimal e3nn 0.6 API smoke test for the equivariant encoder (P2 v2).

Runs in the nequip env. Verifies every e3nn call the encoder depends on,
plus a forward pass through the actual EquivariantStructureEncoder.
"""
import numpy as np
import torch

from e3nn import o3

print("e3nn version:", o3.__file__)


def test_irreps_and_linear():
    ir = o3.Irreps("8x0e + 8x1o")
    lin = o3.Linear(ir, ir)
    x = lin(torch.randn(4, ir.dim))
    assert x.shape == (4, ir.dim)
    print("  o3.Linear OK", x.shape)


def test_spherical_harmonics():
    ir_sh = o3.Irreps.spherical_harmonics(2)
    vec = torch.randn(10, 3)
    vec = vec / vec.norm(dim=-1, keepdim=True)
    sh = o3.spherical_harmonics(ir_sh, vec, normalize=True, normalization="component")
    print("  spherical_harmonics OK", sh.shape)


def test_tensor_product():
    node_ir = o3.Irreps("8x0e + 8x1o")
    sh_ir = o3.Irreps.spherical_harmonics(2)
    tp = o3.FullyConnectedTensorProduct(node_ir, sh_ir, node_ir, shared_weights=True)
    x = torch.randn(6, node_ir.dim)
    y = torch.randn(6, sh_ir.dim)
    out = tp(x, y)
    print("  FullyConnectedTensorProduct OK", out.shape)


def test_batchnorm():
    from e3nn.nn import BatchNorm
    ir = o3.Irreps("8x0e + 8x1o")
    bn = BatchNorm(ir)
    x = bn(torch.randn(4, ir.dim))
    print("  e3nn BatchNorm OK", x.shape)


def test_encoder_forward():
    import sys
    sys.path.insert(0, "/home/zhao/BandStructure_AI_60k_20260903")
    # Import the module directly (not via src.models package __init__ which
    # imports tensorflow, absent in the nequip env).
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "equivariant_structure_encoder",
        "/home/zhao/BandStructure_AI_60k_20260903/src/models/equivariant_structure_encoder.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    EquivariantStructureEncoder = mod.EquivariantStructureEncoder

    enc = EquivariantStructureEncoder(num_elements=109, multiplicity=8, num_layers=2, lmax=2, embedding_dim=128)
    B, N = 1, 8
    atom_features = torch.zeros(B, N, 110)
    for i in range(4):  # 4 real atoms
        atom_features[0, i, (i) % 109 + 1] = 1.0
    # intra-graph edges: each real atom -> its 2 neighbours
    edge_src, edge_dst, edge_vec, edge_len = [], [], [], []
    for i in range(4):
        for k in range(2):
            j = (i + k + 1) % 4
            edge_src.append(i)
            edge_dst.append(j)
            v = torch.randn(3)
            edge_vec.append(v)
            edge_len.append(v.norm())
    edge_src = torch.tensor(edge_src, dtype=torch.long)
    edge_dst = torch.tensor(edge_dst, dtype=torch.long)
    edge_vec = torch.stack(edge_vec).float()
    edge_len = torch.stack(edge_len).float()
    graph = {"atom_features": atom_features, "edge_src": edge_src, "edge_dst": edge_dst,
             "edge_vec": edge_vec, "edge_len": edge_len, "n_atoms": torch.tensor([4])}
    descriptors = torch.randn(B, 22)
    out = enc(graph, descriptors)
    print("  EquivariantStructureEncoder forward OK", out.shape)
    assert out.shape == (B, 128)


if __name__ == "__main__":
    test_irreps_and_linear()
    test_spherical_harmonics()
    test_tensor_product()
    test_batchnorm()
    test_encoder_forward()
    print("ALL E3NN SMOKE TESTS PASSED")
