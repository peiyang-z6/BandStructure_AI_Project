"""
Material Classifier — element, crystal structure, and material type identification.
===============================================================================

Combines Materials Project metadata + PhysicsBrain predictions to produce a
comprehensive material classification from a band structure image.

Data sources (in priority order):
  1. Materials Project metadata cache (data_cache/mp_metadata.json) — 36,550 entries
     → formula_pretty, spacegroup_number, band_gap, is_direct, num_sites
  2. Local OOD manifest (data_cache/ood_tensors/ood_split_manifest.json) — 200 entries
     → material_id, spacegroup_number, band_gap
  3. PhysicsBrain inference (gap, direct/indirect, effective mass)
     → physics-based crystal type classification

Output:
  - elements: parsed chemical formula with element list
  - crystal_system: triclinic/monoclinic/orthorhombic/tetragonal/trigonal/hexagonal/cubic
  - bravais_lattice: P/I/F/C/R + centering description
  - crystal_type: metallic/semiconducting/insulating + bonding character hints
  - confidence: how reliable the classification is
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Spacegroup → Crystal System mapping ─────────────────────────────────────
# Source: International Tables for Crystallography, Vol. A

_SPACEGROUP_TO_SYSTEM: Dict[int, Tuple[str, str, str]] = {
    # (crystal_system, bravais_lattice, point_group_short)
    # Triclinic (1-2)
    1:  ("Triclinic",      "Primitive (aP)",    "1"),
    2:  ("Triclinic",      "Primitive (aP)",    "-1"),

    # Monoclinic (3-15)
    3:  ("Monoclinic",     "Primitive (mP)",    "2"),
    4:  ("Monoclinic",     "Primitive (mP)",    "2₁"),
    5:  ("Monoclinic",     "Base-centered (mC)","2"),
    6:  ("Monoclinic",     "Primitive (mP)",    "m"),
    7:  ("Monoclinic",     "Base-centered (mC)","m"),
    8:  ("Monoclinic",     "Base-centered (mC)","2/m"),
    9:  ("Monoclinic",     "Base-centered (mC)","2/m"),
    10: ("Monoclinic",     "Primitive (mP)",    "2/m"),
    11: ("Monoclinic",     "Primitive (mP)",    "2/m"),
    12: ("Monoclinic",     "Base-centered (mC)","2/m"),
    13: ("Monoclinic",     "Primitive (mP)",    "2/m"),
    14: ("Monoclinic",     "Primitive (mP)",    "2/m"),
    15: ("Monoclinic",     "Base-centered (mC)","2/m"),

    # Orthorhombic (16-74)
    16: ("Orthorhombic",   "Primitive (oP)",    "222"),
    17: ("Orthorhombic",   "Primitive (oP)",    "222"),
    18: ("Orthorhombic",   "Primitive (oP)",    "2₁2₁2"),
    19: ("Orthorhombic",   "Primitive (oP)",    "2₁2₁2₁"),
    20: ("Orthorhombic",   "Base-centered (oC)","222"),
    21: ("Orthorhombic",   "Base-centered (oC)","222"),
    22: ("Orthorhombic",   "Face-centered (oF)","222"),
    23: ("Orthorhombic",   "Body-centered (oI)","222"),
    24: ("Orthorhombic",   "Body-centered (oI)","2₁2₁2₁"),
    25: ("Orthorhombic",   "Primitive (oP)",    "mm2"),
    26: ("Orthorhombic",   "Primitive (oP)",    "mm2"),
    27: ("Orthorhombic",   "Primitive (oP)",    "mm2"),
    28: ("Orthorhombic",   "Primitive (oP)",    "mm2"),
    29: ("Orthorhombic",   "Primitive (oP)",    "mm2"),
    30: ("Orthorhombic",   "Primitive (oP)",    "mm2"),
    31: ("Orthorhombic",   "Primitive (oP)",    "mm2"),
    32: ("Orthorhombic",   "Primitive (oP)",    "mm2"),
    33: ("Orthorhombic",   "Primitive (oP)",    "mm2"),
    34: ("Orthorhombic",   "Primitive (oP)",    "mm2"),
    35: ("Orthorhombic",   "Base-centered (oC)","mm2"),
    36: ("Orthorhombic",   "Base-centered (oC)","mm2"),
    37: ("Orthorhombic",   "Base-centered (oC)","mm2"),
    38: ("Orthorhombic",   "Base-centered (oC)","mm2"),
    39: ("Orthorhombic",   "Base-centered (oC)","mm2"),
    40: ("Orthorhombic",   "Base-centered (oC)","mm2"),
    41: ("Orthorhombic",   "Base-centered (oC)","mm2"),
    42: ("Orthorhombic",   "Face-centered (oF)","mm2"),
    43: ("Orthorhombic",   "Face-centered (oF)","mm2"),
    44: ("Orthorhombic",   "Body-centered (oI)","mm2"),
    45: ("Orthorhombic",   "Body-centered (oI)","mm2"),
    46: ("Orthorhombic",   "Body-centered (oI)","mm2"),
    47: ("Orthorhombic",   "Primitive (oP)",    "mmm"),
    48: ("Orthorhombic",   "Primitive (oP)",    "mmm"),
    49: ("Orthorhombic",   "Primitive (oP)",    "mmm"),
    50: ("Orthorhombic",   "Primitive (oP)",    "mmm"),
    51: ("Orthorhombic",   "Primitive (oP)",    "mmm"),
    52: ("Orthorhombic",   "Primitive (oP)",    "mmm"),
    53: ("Orthorhombic",   "Primitive (oP)",    "mmm"),
    54: ("Orthorhombic",   "Primitive (oP)",    "mmm"),
    55: ("Orthorhombic",   "Primitive (oP)",    "mmm"),
    56: ("Orthorhombic",   "Primitive (oP)",    "mmm"),
    57: ("Orthorhombic",   "Primitive (oP)",    "mmm"),
    58: ("Orthorhombic",   "Primitive (oP)",    "mmm"),
    59: ("Orthorhombic",   "Primitive (oP)",    "mmm"),
    60: ("Orthorhombic",   "Primitive (oP)",    "mmm"),
    61: ("Orthorhombic",   "Primitive (oP)",    "mmm"),
    62: ("Orthorhombic",   "Primitive (oP)",    "mmm"),
    63: ("Orthorhombic",   "Base-centered (oC)","mmm"),
    64: ("Orthorhombic",   "Base-centered (oC)","mmm"),
    65: ("Orthorhombic",   "Base-centered (oC)","mmm"),
    66: ("Orthorhombic",   "Base-centered (oC)","mmm"),
    67: ("Orthorhombic",   "Base-centered (oC)","mmm"),
    68: ("Orthorhombic",   "Base-centered (oC)","mmm"),
    69: ("Orthorhombic",   "Face-centered (oF)","mmm"),
    70: ("Orthorhombic",   "Face-centered (oF)","mmm"),
    71: ("Orthorhombic",   "Body-centered (oI)","mmm"),
    72: ("Orthorhombic",   "Body-centered (oI)","mmm"),
    73: ("Orthorhombic",   "Body-centered (oI)","mmm"),
    74: ("Orthorhombic",   "Body-centered (oI)","mmm"),

    # Tetragonal (75-142)
    # 75-80: P, 79-80: I, 81-82: I, 83-88: P, 89-98: P, 99-110: I, 111-122: P, 123-142: I
}

# Use a more compact representation for the remaining groups:
# Tetragonal (75-142)
for sg in range(75, 143):
    if sg <= 80:    lat = "Primitive (tP)"
    elif sg <= 88:  lat = "Body-centered (tI)"
    elif sg <= 98:  lat = "Primitive (tP)"
    elif sg <= 110: lat = "Body-centered (tI)"
    elif sg <= 122: lat = "Primitive (tP)"
    else:           lat = "Body-centered (tI)"
    if sg <= 82:    pg = "4" if sg in (75,76) else ("-4" if sg in (81,82) else "4/m")
    elif sg <= 88:  pg = "4/m"
    elif sg <= 98:  pg = "422"
    elif sg <= 110: pg = "4mm"
    elif sg <= 122: pg = "-42m"
    else:           pg = "4/mmm"
    _SPACEGROUP_TO_SYSTEM[sg] = ("Tetragonal", lat, pg)

# Trigonal/Rhombohedral (143-167)
for sg in range(143, 168):
    lat = "Rhombohedral (hR)" if sg <= 148 or sg in (155,160,161,166,167) else "Hexagonal (hP)"
    pg = "3" if sg <= 146 else ("-3" if sg <= 148 else ("32" if sg<=155 else ("3m" if sg<=161 else "-3m")))
    _SPACEGROUP_TO_SYSTEM[sg] = ("Trigonal", lat, pg)

# Hexagonal (168-194)
for sg in range(168, 195):
    lat = "Primitive (hP)"
    pg = "6" if sg<=173 else ("-6" if sg<=174 else ("6/m" if sg<=176 else ("622" if sg<=182 else ("6mm" if sg<=186 else ("-62m" if sg<=190 else "6/mmm")))))
    _SPACEGROUP_TO_SYSTEM[sg] = ("Hexagonal", lat, pg)

# Cubic (195-230)
for sg in range(195, 231):
    if sg <= 199: lat = "Primitive (cP)"
    elif sg <= 206: lat = "Face-centered (cF)"
    else: lat = "Body-centered (cI)"
    pg = "23" if sg<=199 else ("m-3" if sg<=206 else ("432" if sg<=211 else ("-43m" if sg<=220 else "m-3m")))
    _SPACEGROUP_TO_SYSTEM[sg] = ("Cubic", lat, pg)


# ── Element parsing ─────────────────────────────────────────────────────────

_ELEMENT_NAMES = {
    "H":"Hydrogen","He":"Helium","Li":"Lithium","Be":"Beryllium","B":"Boron",
    "C":"Carbon","N":"Nitrogen","O":"Oxygen","F":"Fluorine","Ne":"Neon",
    "Na":"Sodium","Mg":"Magnesium","Al":"Aluminum","Si":"Silicon","P":"Phosphorus",
    "S":"Sulfur","Cl":"Chlorine","Ar":"Argon","K":"Potassium","Ca":"Calcium",
    "Sc":"Scandium","Ti":"Titanium","V":"Vanadium","Cr":"Chromium","Mn":"Manganese",
    "Fe":"Iron","Co":"Cobalt","Ni":"Nickel","Cu":"Copper","Zn":"Zinc",
    "Ga":"Gallium","Ge":"Germanium","As":"Arsenic","Se":"Selenium","Br":"Bromine",
    "Kr":"Krypton","Rb":"Rubidium","Sr":"Strontium","Y":"Yttrium","Zr":"Zirconium",
    "Nb":"Niobium","Mo":"Molybdenum","Tc":"Technetium","Ru":"Ruthenium","Rh":"Rhodium",
    "Pd":"Palladium","Ag":"Silver","Cd":"Cadmium","In":"Indium","Sn":"Tin",
    "Sb":"Antimony","Te":"Tellurium","I":"Iodine","Xe":"Xenon","Cs":"Cesium",
    "Ba":"Barium","La":"Lanthanum","Ce":"Cerium","Pr":"Praseodymium","Nd":"Neodymium",
    "Pm":"Promethium","Sm":"Samarium","Eu":"Europium","Gd":"Gadolinium",
    "Tb":"Terbium","Dy":"Dysprosium","Ho":"Holmium","Er":"Erbium","Tm":"Thulium",
    "Yb":"Ytterbium","Lu":"Lutetium","Hf":"Hafnium","Ta":"Tantalum","W":"Tungsten",
    "Re":"Rhenium","Os":"Osmium","Ir":"Iridium","Pt":"Platinum","Au":"Gold",
    "Hg":"Mercury","Tl":"Thallium","Pb":"Lead","Bi":"Bismuth","Po":"Polonium",
    "At":"Astatine","Rn":"Radon","Fr":"Francium","Ra":"Radium","Ac":"Actinium",
    "Th":"Thorium","Pa":"Protactinium","U":"Uranium","Np":"Neptunium","Pu":"Plutonium",
}

_ELEMENT_PATTERN = re.compile(r'([A-Z][a-z]?)(\d*)')

def parse_formula(formula: str) -> List[Dict[str, Any]]:
    """Parse a chemical formula like 'CsPbI3' or 'Ac2S3' into element list."""
    matches = _ELEMENT_PATTERN.findall(formula)
    elements = []
    for symbol, count_str in matches:
        count = int(count_str) if count_str else 1
        name = _ELEMENT_NAMES.get(symbol, symbol)
        elements.append({"symbol": symbol, "name": name, "count": count})
    return elements


# ── Material type inference from physics ────────────────────────────────────

def infer_crystal_type(gap_ev: float, is_direct: bool,
                       vbm_mass: Optional[float] = None,
                       cbm_mass: Optional[float] = None,
                       spacegroup: Optional[int] = None) -> Tuple[str, str, List[str]]:
    """Infer crystal type (bonding character) from physics properties.

    Returns: (primary_type, detailed_type, evidence_list)
    """
    evidence = []
    primary = "Unknown"
    detail = ""

    # Gap-based classification
    if gap_ev <= 0.05:
        primary = "Metal"
        detail = "Metallic (zero or near-zero band gap)"
        evidence.append(f"Band gap = {gap_ev:.3f} eV (metallic)")
        # Check for semi-metal hints
        if is_direct and gap_ev > 0:
            detail = "Semi-metal (very small direct gap)"
    elif gap_ev <= 0.5:
        primary = "Narrow-gap Semiconductor"
        detail = "Narrow-gap semiconductor"
        evidence.append(f"Narrow band gap = {gap_ev:.3f} eV")
    elif gap_ev <= 3.0:
        primary = "Semiconductor"
        detail = "Semiconductor"
        evidence.append(f"Band gap = {gap_ev:.3f} eV (semiconducting range)")
    else:
        primary = "Insulator"
        detail = "Wide-bandgap insulator"
        evidence.append(f"Wide band gap = {gap_ev:.3f} eV (>3 eV)")

    # Gap type
    evidence.append(f"Gap type: {'direct' if is_direct else 'indirect'}")

    # Bonding character from effective mass
    mass_hints = []
    if vbm_mass is not None and abs(vbm_mass) < 10:
        if abs(vbm_mass) < 0.5:
            mass_hints.append("Light VB holes (suggests covalent/delocalized bonding)")
        elif abs(vbm_mass) > 3:
            mass_hints.append("Heavy VB holes (suggests ionic/localized character)")
    if cbm_mass is not None and abs(cbm_mass) < 10:
        if abs(cbm_mass) < 0.3:
            mass_hints.append("Light CB electrons (high mobility, likely covalent)")
        elif abs(cbm_mass) > 3:
            mass_hints.append("Heavy CB electrons (low mobility, likely ionic)")
    evidence.extend(mass_hints)

    # Bonding type refinement
    if "covalent" in " ".join(mass_hints).lower() and "ionic" not in " ".join(mass_hints).lower():
        detail += " (covalent bonding character)"
    elif "ionic" in " ".join(mass_hints).lower():
        detail += " (ionic bonding character)"
    elif 1.0 <= gap_ev <= 2.5 and is_direct:
        detail += " (mixed covalent-ionic, optoelectronic candidate)"

    # Spacegroup-based hints
    if spacegroup is not None:
        if spacegroup in (225, 221, 216, 227):  # Fm-3m, Pm-3m, F-43m, Fd-3m
            detail += " [high-symmetry cubic]"

    return primary, detail, evidence


# ═════════════════════════════════════════════════════════════════════════════
#  Main classifier class
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MaterialClassification:
    """Complete material classification result."""
    # Elements
    formula: str = ""
    elements: List[Dict[str, Any]] = field(default_factory=list)
    element_source: str = ""  # "mp_metadata" | "parsed_from_id" | "unknown"

    # Crystal structure
    spacegroup_number: Optional[int] = None
    crystal_system: str = ""
    bravais_lattice: str = ""
    point_group: str = ""
    structure_source: str = ""  # "mp_metadata" | "local_manifest" | "inferred"

    # Crystal type
    crystal_type: str = ""        # Metal/Semiconductor/Insulator
    crystal_type_detail: str = "" # Detailed description
    type_evidence: List[str] = field(default_factory=list)

    # Confidence
    confidence: float = 0.0  # 0-1

    def to_text(self) -> str:
        """Format as human-readable text block."""
        lines = [
            "╔══════════════════════════════════╗",
            "║   MATERIAL IDENTIFICATION        ║",
            "╚══════════════════════════════════╝",
            "",
        ]
        # Elements
        if self.formula:
            el_str = "".join(
                f"{e['symbol']}{e['count'] if e['count']>1 else ''}"
                for e in self.elements
            )
            names = ", ".join(f"{e['name']}({e['symbol']})" for e in self.elements)
            lines.append(f"  Formula:        {self.formula}  ({el_str})")
            lines.append(f"  Elements:       {names}")
            lines.append(f"  Source:         {self.element_source}")

        # Crystal structure
        if self.crystal_system:
            lines.append("")
            lines.append(f"  Crystal system: {self.crystal_system}")
            lines.append(f"  Bravais lattice:{self.bravais_lattice}")
            lines.append(f"  Space group:    #{self.spacegroup_number} ({self.point_group})")
            lines.append(f"  Source:         {self.structure_source}")

        # Crystal type
        if self.crystal_type:
            lines.append("")
            lines.append(f"  Crystal type:   {self.crystal_type}")
            lines.append(f"  Detail:         {self.crystal_type_detail}")
            if self.type_evidence:
                lines.append("  Evidence:")
                for ev in self.type_evidence:
                    lines.append(f"    • {ev}")

        lines.append("")
        lines.append(f"  Confidence:     {self.confidence:.0%}")
        return "\n".join(lines)


class MaterialClassifier:
    """Classify a material from its band structure and available metadata."""

    def __init__(self):
        self._mp_meta: Dict[str, Dict] = {}
        self._local_meta: Dict[str, Dict] = {}
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata caches."""
        # Materials Project metadata
        mp_path = Path(__file__).resolve().parents[2] / "data_cache" / "mp_metadata.json"
        try:
            data = json.loads(mp_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    mid = item.get("material_id", "")
                    if mid:
                        self._mp_meta[str(mid)] = item
        except Exception:
            pass

        # Local OOD manifest
        manifest_path = Path(__file__).resolve().parents[2] / "data_cache" / "ood_tensors" / "ood_split_manifest.json"
        try:
            man = json.loads(manifest_path.read_text(encoding="utf-8"))
            for s in man.get("samples", []):
                mid = s.get("material_id", "")
                if mid:
                    self._local_meta[str(mid)] = s
        except Exception:
            pass

    def classify(
        self,
        gap_ev: float = 0.0,
        is_direct: bool = False,
        vbm_mass: Optional[float] = None,
        cbm_mass: Optional[float] = None,
        material_id: str = "",
        file_path: str = "",
    ) -> MaterialClassification:
        """Classify a material from all available data sources."""
        result = MaterialClassification()
        confidence_parts = []

        # ── 1. Try Materials Project metadata ──
        mp_entry = self._mp_meta.get(material_id)
        if not mp_entry:
            # Try to extract material_id from filename
            basename = Path(file_path).stem if file_path else ""
            mp_match = re.search(r'mp-\d+', basename, re.IGNORECASE)
            if mp_match:
                mp_entry = self._mp_meta.get(mp_match.group())

        # ── 2. Try local manifest ──
        local_entry = self._local_meta.get(material_id) if material_id else None

        # ── 3. Elements ──
        if mp_entry and "formula_pretty" in mp_entry:
            formula = mp_entry["formula_pretty"]
            result.formula = formula
            result.elements = parse_formula(formula)
            result.element_source = "Materials Project database"
            confidence_parts.append(0.95)
        else:
            # Try to parse formula from material_id
            parsed = self._parse_id_for_formula(material_id)
            if parsed:
                result.formula = parsed
                result.elements = parse_formula(parsed)
                result.element_source = "parsed from material ID (verify manually)"
                confidence_parts.append(0.5)
            else:
                result.element_source = "unknown — provide material_id or mp-ID for lookup"
                confidence_parts.append(0.0)

        # ── 4. Crystal structure ──
        sg = None
        if mp_entry and "spacegroup_number" in mp_entry:
            sg = mp_entry["spacegroup_number"]
            result.structure_source = "Materials Project database"
            confidence_parts.append(0.95)
        elif local_entry and "spacegroup_number" in local_entry:
            sg = local_entry["spacegroup_number"]
            result.structure_source = "local OOD manifest"
            confidence_parts.append(0.85)
        else:
            result.structure_source = "unknown — provide material_id for lookup"
            confidence_parts.append(0.0)

        if sg is not None:
            system_info = _SPACEGROUP_TO_SYSTEM.get(int(sg))
            if system_info:
                result.spacegroup_number = int(sg)
                result.crystal_system, result.bravais_lattice, result.point_group = system_info

        # ── 5. Crystal type (always available from physics) ──
        primary, detail, evidence = infer_crystal_type(
            gap_ev, is_direct, vbm_mass, cbm_mass, sg
        )
        result.crystal_type = primary
        result.crystal_type_detail = detail
        result.type_evidence = evidence
        confidence_parts.append(0.80)  # physics-based inference

        # ── Compute overall confidence ──
        result.confidence = float(np.mean([c for c in confidence_parts if c > 0]) if any(
            c > 0 for c in confidence_parts) else 0.0)

        return result

    def _parse_id_for_formula(self, material_id: str) -> str:
        """Try to extract chemical formula from a material ID string."""
        if not material_id:
            return ""
        # If it looks like a formula (e.g., CsPbI3, MoS2)
        if re.match(r'^[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+$', material_id.strip()):
            return material_id.strip()
        # If it contains a formula-like substring
        match = re.search(r'([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*){1,})', material_id)
        if match:
            return match.group(1)
        return ""


# ── Quick CLI test ──────────────────────────────────────────────────────────

def main():
    """Test the classifier with sample data."""
    clf = MaterialClassifier()

    # Test 1: Known MP material
    result = clf.classify(
        gap_ev=2.30, is_direct=True, vbm_mass=0.3, cbm_mass=0.25,
        material_id="mp-32800"
    )
    print(result.to_text())
    print()

    # Test 2: Material ID looks like formula
    result2 = clf.classify(
        gap_ev=1.52, is_direct=True, vbm_mass=0.4, cbm_mass=0.2,
        material_id="CsPbI3"
    )
    print(result2.to_text())
    print()

    # Test 3: Unknown material
    result3 = clf.classify(
        gap_ev=0.0, is_direct=False,
        material_id="unknown_sample"
    )
    print(result3.to_text())


if __name__ == "__main__":
    main()
