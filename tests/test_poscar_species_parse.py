"""Regression tests for POSCAR per-atom species parsing (P2 pre-flight)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from fix_sidecar_species import parse_poscar_species

POSCAR_VASP5 = """CoSc_sv/T0001.AB2C
   1.24482600000000
     0.0000000000000000    2.5973505930014307    2.5973505930014307
     2.5973505930014307    0.0000000000000000    2.5973505930014307
     2.5973505930014307    2.5973505930014307    0.0000000000000000
   Co   Sc   Zn
     1     2     1
Direct
 -0.0000000000000000  0.0000000000000000 -0.0000000000000000
  0.2500000000000000  0.2500000000000000  0.2500000000000000
"""


def test_parse_poscar_vasp5_with_symbols_and_counts():
    per_atom = parse_poscar_species(POSCAR_VASP5)
    assert per_atom == ["Co", "Sc", "Sc", "Zn"]


def test_parse_poscar_vasp4_counts_only():
    # VASP4 style: no species-symbol line; counts only. Parser must refuse.
    text = """CoSc
   1.0
     4.0 0.0 0.0
     0.0 4.0 0.0
     0.0 0.0 4.0
     1     2
Direct
 0 0 0
 0.25 0.25 0.25
"""
    # Without symbols, counts-only cannot be expanded -> None (safe refusal).
    assert parse_poscar_species(text) is None


def test_parse_too_short():
    assert parse_poscar_species("one\ntwo\n") is None


def test_parse_potassium_title_not_misdetected_as_mode():
    # Regression: a title line starting with "K_" must not be mistaken for a
    # "K" (k-points) coordinate-mode line.
    text = """K_svSeSn/T0001.ABC2 - (T0001.ABC2) - AlC
   1.22474500000000
     0.0    3.176   3.176
     3.176  0.0     3.176
     3.176  3.176   0.0
   K    Se   Sn
     1     1     2
Direct
 0 0 0
 0.5 0.5 0.5
 0.75 0.75 0.75
 0.25 0.25 0.25
"""
    assert parse_poscar_species(text) == ["K", "Se", "Sn", "Sn"]


def test_parse_vasp4_counts_only_with_known_species():
    text = """CoSc
   1.0
     4.0 0.0 0.0
     0.0 4.0 0.0
     0.0 0.0 4.0
     1     2     1
Direct
 0 0 0
 0.25 0.25 0.25
 0.5 0.5 0.5
 0.75 0.75 0.75
"""
    per_atom = parse_poscar_species(text, known_species=["Co", "Sc", "Zn"])
    assert per_atom == ["Co", "Sc", "Sc", "Zn"]


def test_parse_vasp4_counts_only_without_known_species_refuses():
    text = """CoSc
   1.0
     4.0 0.0 0.0
     0.0 4.0 0.0
     0.0 0.0 4.0
     1     2
Direct
 0 0 0
 0.25 0.25 0.25
"""
    assert parse_poscar_species(text) is None
