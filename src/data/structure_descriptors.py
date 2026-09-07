"""Enhanced physical descriptors for crystal structures (P2 v2).

Constitution 5.0 §8 P2: the pure one-hot CGCNN encoder overfit because its
structure representation lacked distinguishing power across space groups.
These descriptors (spacegroup, lattice parameters, composition statistics)
augment the equivariant encoder's global feature so different structures
produce different embeddings.

All functions are pure numpy (no torch/TF/pymatgen) so they are unit-testable
on the Windows side and reusable in both the PyTorch (nequip) and TF paths.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

# Pauling electronegativities for the canonical element set (indexed by atomic
# number; 0 = unknown). Values are the standard Pauling scale.
ELECTRONEGATIVITY = {
    "H": 2.20, "He": 0.0, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55,
    "N": 3.04, "O": 3.44, "F": 3.98, "Ne": 0.0, "Na": 0.93, "Mg": 1.31,
    "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16, "Ar": 0.0,
    "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66,
    "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65,
    "Ga": 1.81, "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96, "Kr": 3.00,
    "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33, "Nb": 1.6, "Mo": 2.16,
    "Tc": 1.9, "Ru": 2.2, "Rh": 2.28, "Pd": 2.20, "Ag": 1.93, "Cd": 1.69,
    "In": 1.78, "Sn": 1.96, "Sb": 2.05, "Te": 2.1, "I": 2.66, "Xe": 2.6,
    "Cs": 0.79, "Ba": 0.89, "La": 1.10, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14,
    "Pm": 1.13, "Sm": 1.17, "Eu": 1.2, "Gd": 1.20, "Tb": 1.1, "Dy": 1.22,
    "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": 1.1, "Lu": 1.27, "Hf": 1.3,
    "Ta": 1.5, "W": 2.36, "Re": 1.9, "Os": 2.2, "Ir": 2.20, "Pt": 2.28,
    "Au": 2.54, "Hg": 2.00, "Tl": 1.62, "Pb": 2.33, "Bi": 2.02, "Po": 2.0,
    "At": 2.2, "Rn": 0.0, "Fr": 0.7, "Ra": 0.9, "Ac": 1.1, "Th": 1.3,
    "Pa": 1.5, "U": 1.38, "Np": 1.36, "Pu": 1.28, "Am": 1.13, "Cm": 1.28,
    "Bk": 1.3, "Cf": 1.3, "Es": 1.3, "Fm": 1.3, "Md": 1.3, "No": 1.3,
    "Lr": 1.3, "Rf": 0.0, "Db": 0.0, "Sg": 0.0, "Bh": 0.0, "Hs": 0.0,
    "Mt": 0.0,
}


def lattice_params(lattice: Sequence[Sequence[float]]) -> np.ndarray:
    """Lattice 3x3 row-vector matrix -> [a, b, c, alpha, beta, gamma, volume].

    a,b,c are vector norms; alpha=angle(b,c), beta=angle(a,c), gamma=angle(a,b)
    in degrees; volume = |a . (b x c)|.
    """
    L = np.asarray(lattice, dtype=np.float64)
    a, b, c = L[0], L[1], L[2]
    la, lb, lc = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
    alpha = np.degrees(np.arccos(np.clip(np.dot(b, c) / (lb * lc), -1, 1)))
    beta = np.degrees(np.arccos(np.clip(np.dot(a, c) / (la * lc), -1, 1)))
    gamma = np.degrees(np.arccos(np.clip(np.dot(a, b) / (la * lb), -1, 1)))
    vol = abs(np.dot(a, np.cross(b, c)))
    return np.array([la, lb, lc, alpha, beta, gamma, vol], dtype=np.float32)


def composition_stats(species: Sequence[str]) -> np.ndarray:
    """Per-atom species -> composition descriptors (electronegativity stats).

    Returns [mean_EN, std_EN, min_EN, max_EN, n_unique_elements,
    mean_atomic_number, std_atomic_number]. Uses the dedup-free per-atom list.
    """
    if not species:
        return np.zeros(7, dtype=np.float32)
    ens = [ELECTRONEGATIVITY.get(s, 0.0) for s in species]
    # atomic numbers via a compact map (H=1 ... via symbol position)
    import src.data.crystal_graph as cg  # reuse canonical element index
    zs = [cg.ELEMENT_INDEX.get(s, 0) for s in species]
    ens = np.asarray(ens, dtype=np.float32)
    zs = np.asarray(zs, dtype=np.float32)
    return np.array([
        ens.mean(), ens.std(), ens.min(), ens.max(),
        len(set(species)), zs.mean(), zs.std(),
    ], dtype=np.float32)


def global_descriptors(
    lattice: Sequence[Sequence[float]],
    species: Sequence[str],
    spacegroup: int,
) -> np.ndarray:
    """Concatenated global descriptors for one structure.

    Returns a fixed-length vector: lattice (7) + composition (7) +
    one-hot spacegroup bucket. Spacegroup is bucketed into 8 log2 bins
    (1-2, 3-4, 5-8, ... 129-230) to avoid a 230-dim sparse one-hot.
    """
    lp = lattice_params(lattice)
    comp = composition_stats(species)
    # spacegroup bucket: log2 index
    sg = int(spacegroup) if spacegroup else 0
    bucket = 0 if sg < 1 else min(7, int(np.floor(np.log2(sg))))
    sg_onehot = np.zeros(8, dtype=np.float32)
    sg_onehot[bucket] = 1.0
    return np.concatenate([lp, comp, sg_onehot]).astype(np.float32)
