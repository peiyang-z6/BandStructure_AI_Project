"""Regression tests for P1 KPOINTS.bands 3D k-point parsing."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from build_structure_sidecar import parse_kpoints_bands

SAMPLE = """MCL (monoclinic) G-Y-H-C-E-M1-A-X-G-Z-D-M Z-A D-Y X-H1 
20   ! 20 grids  
Line-mode 
reciprocal 
   0.0000   0.0000   0.0000   ! \\Gamma 
   0.0000   0.0000   0.5000   ! Y 
  
   0.0000   0.0000   0.5000   ! Y 
   0.0000   0.4309   0.5749   ! H 
"""


def test_parse_kpoints_bands_segments():
    out = parse_kpoints_bands(SAMPLE)
    assert "error" not in out
    assert out["nkpts"] == 20
    assert len(out["segments"]) == 2
    seg0 = out["segments"][0]
    assert seg0["label_from"] == "\\Gamma"
    assert seg0["label_to"] == "Y"
    assert seg0["k_from"] == [0.0, 0.0, 0.0]
    assert seg0["k_to"] == [0.0, 0.0, 0.5]
    seg1 = out["segments"][1]
    assert seg1["k_to"] == [0.0, 0.4309, 0.5749]


def test_parse_kpoints_bands_too_short():
    assert "error" in parse_kpoints_bands("only two lines\n")


def test_parse_kpoints_bands_no_label():
    text = "FCC G-X \n10 ! 10 grids \nLine-mode \nreciprocal \n 0 0 0 \n 0.5 0 0 \n"
    out = parse_kpoints_bands(text)
    assert "error" not in out
    assert out["segments"][0]["label_from"] is None
    assert out["segments"][0]["k_to"] == [0.5, 0.0, 0.0]
