"""Crystal graph construction for the P2 cross-modal structure encoder.

Constitution 5.0 §8 P2: reuse a mature GNN. This builds CGCNN-style crystal
graphs (Xie & Grossman 2018) from the P1 structure sidecar:

- node features: one-hot over a canonical element set (atomic number sorted),
  padded to a fixed max_atoms per graph;
- edges: for each atom, the 12 nearest neighbours within an 8 A cutoff
  (CGCNN default), with the inter-atomic distance as the scalar edge feature;
- distance RBF expansion into fixed bins for the encoder.

Graphs are built with pymatgen (Structure.get_all_neighbors), then serialized
to a fixed-size numpy array per structure for batched TF feeding.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# Canonical element set: atomic number -> element symbol. Sorted by atomic
# number; index 0 reserved for the padding atom.
ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
]
ELEMENT_INDEX = {sym: i + 1 for i, sym in enumerate(ELEMENTS)}  # 0 = padding


# CGCNN defaults
CUTOFF = 8.0
MAX_NEIGHBORS = 12
MAX_ATOMS = 50
RBF_BINS = 40
RBF_DMAX = 8.0


def element_onehot(species: Sequence[str], num_elements: Optional[int] = None) -> np.ndarray:
    """One-hot encode a per-atom species list (n_atoms, n_elements+1)."""
    n = num_elements if num_elements is not None else len(ELEMENTS)
    n_atoms = len(species)
    onehot = np.zeros((n_atoms, n + 1), dtype=np.float32)
    for i, sym in enumerate(species):
        idx = ELEMENT_INDEX.get(sym)
        if idx is not None and idx <= n:
            onehot[i, idx] = 1.0
    return onehot


def rbf_expand(distances: np.ndarray, bins: int = RBF_BINS, dmax: float = RBF_DMAX) -> np.ndarray:
    """Gaussian RBF expansion of scalar distances -> (..., bins).

    mu_i = (i / bins) * dmax, sigma = dmax / bins (CGCNN-style, scaled).
    """
    distances = np.asarray(distances, dtype=np.float32)
    centers = np.linspace(0.0, dmax, bins, endpoint=False, dtype=np.float32)
    width = dmax / bins
    # (..., bins)
    diff = distances[..., None] - centers[None, :]
    return np.exp(-(diff ** 2) / (2.0 * width ** 2)).astype(np.float32)


def validate_max_atoms(max_atoms: Optional[int]) -> int:
    """Reject silent truncation or capacity expansion beyond the P2 contract."""
    if max_atoms is None:
        return MAX_ATOMS
    if (isinstance(max_atoms, (bool, np.bool_))
            or not isinstance(max_atoms, (int, np.integer))
            or not 1 <= max_atoms <= MAX_ATOMS):
        raise ValueError("max_atoms must be an integer in [1, 50]")
    return int(max_atoms)


def validate_geometry(lattice, fractional_coordinates):
    """Validate before entering pymatgen's native periodic neighbour search."""
    lat = np.asarray(lattice, dtype=np.float64)
    frac = np.asarray(fractional_coordinates, dtype=np.float64)
    if not np.isfinite(lat).all() or not np.isfinite(frac).all():
        raise ValueError("lattice and fractional coordinates must be finite")
    if lat.shape != (3, 3) or np.linalg.matrix_rank(lat) != 3:
        raise ValueError("lattice must be a nonsingular 3x3 matrix")
    if frac.ndim != 2 or frac.shape[1] != 3 or not len(frac):
        raise ValueError("fractional coordinates must be a nonempty (n_atoms, 3) array")
    with np.errstate(over="ignore", invalid="ignore"):
        cartesian = frac @ lat
    if not np.isfinite(cartesian).all():
        raise ValueError("Cartesian coordinates must remain finite")
    return lat, frac


def validate_species(species, n_atoms):
    """Unknown elements cannot silently become real atoms with padding features."""
    if (isinstance(species, str) or len(species) != n_atoms
            or any(not isinstance(s, str) or s not in ELEMENT_INDEX for s in species)):
        raise ValueError("species must contain one supported element per atom")


def build_crystal_graph(
    lattice: Sequence[Sequence[float]],
    species: Sequence[str],
    fractional_coordinates: Sequence[Sequence[float]],
    max_atoms: Optional[int] = None,
    cutoff: float = CUTOFF,
    max_neighbors: int = MAX_NEIGHBORS,
) -> Dict[str, np.ndarray]:
    """Build a CGCNN-style graph for one structure.

    Returns dict with:
      atom_features  (max_atoms, n_elements+1) one-hot (padded)
      neighbor_list  (max_atoms, max_neighbors) neighbor atom indices (self pad)
      neighbor_dist  (max_atoms, max_neighbors) RAW distances (RBF done in-model)
      n_atoms        int

    Raw distances are stored (not RBF-expanded) to keep the cached pair data
    memory-bounded: RBF expansion happens inside the TF encoder.
    """
    from pymatgen.core import Lattice, Structure

    capacity = validate_max_atoms(max_atoms)
    lat, frac = validate_geometry(lattice, fractional_coordinates)
    validate_species(species, len(frac))
    struct = Structure(Lattice(lat), list(species), frac)
    n_atoms = len(struct)
    if n_atoms > capacity:
        raise ValueError("atom count exceeds max_atoms capacity (at most 50)")
    all_nbrs = struct.get_all_neighbors(cutoff, include_index=True, numerical_tol=0.01)

    n = max_atoms if max_atoms is not None else n_atoms
    atom_features = element_onehot(species)
    neighbor_list = np.full((n, max_neighbors), -1, dtype=np.int32)
    neighbor_dist = np.zeros((n, max_neighbors), dtype=np.float32)

    for i in range(min(n_atoms, n)):
        site_nbrs = sorted(all_nbrs[i], key=lambda x: x.nn_distance)[:max_neighbors]
        for j, nbr in enumerate(site_nbrs):
            if nbr.index < n:
                neighbor_list[i, j] = nbr.index
                neighbor_dist[i, j] = float(nbr.nn_distance)

    # Pad atom_features to (n, ...)
    if n > n_atoms:
        pad = np.zeros((n - n_atoms, atom_features.shape[1]), dtype=np.float32)
        atom_features = np.concatenate([atom_features, pad], axis=0)

    return {
        "atom_features": atom_features,
        "neighbor_list": neighbor_list,
        "neighbor_dist": neighbor_dist,
        "n_atoms": n_atoms,
    }


def build_crystal_graph_batch(
    structures: Sequence[Dict[str, object]],
    max_atoms: Optional[int] = None,
    cutoff: float = CUTOFF,
    max_neighbors: int = MAX_NEIGHBORS,
) -> Dict[str, np.ndarray]:
    """Build graphs for a batch of structure dicts (lattice/species/coords)."""
    graphs = [
        build_crystal_graph(
            s["lattice"], s["species"], s["fractional_coordinates"],
            max_atoms=max_atoms, cutoff=cutoff, max_neighbors=max_neighbors,
        )
        for s in structures
    ]
    if max_atoms is None:
        max_atoms = max(g["n_atoms"] for g in graphs)
    n = max_atoms
    b = len(graphs)
    atom_features = np.zeros((b, n, len(ELEMENTS) + 1), dtype=np.float32)
    neighbor_list = np.full((b, n, max_neighbors), -1, dtype=np.int32)
    neighbor_dist = np.zeros((b, n, max_neighbors), dtype=np.float32)
    for gi, g in enumerate(graphs):
        na = min(g["n_atoms"], n)
        atom_features[gi, :na] = g["atom_features"][:na]
        neighbor_list[gi, :na] = g["neighbor_list"][:na]
        neighbor_dist[gi, :na] = g["neighbor_dist"][:na]
    return {
        "atom_features": atom_features,
        "neighbor_list": neighbor_list,
        "neighbor_dist": neighbor_dist,
        "n_atoms": np.asarray([g["n_atoms"] for g in graphs], dtype=np.int32),
    }
