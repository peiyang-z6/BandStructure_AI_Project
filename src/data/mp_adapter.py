"""
Materials Project data adapter.

Constitution notes:
- Keeps Materials Project metadata needed by the spacegroup splitter.
- Stores line-mode band structures in the existing HDF5/JSON cache layout.
- Uses single-request fetching with rate limiting and local JSON cache.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
from pymatgen.electronic_structure.core import Spin
from pymatgen.ext.matproj import MPRester as LegacyMPRester

try:
    from mp_api.client import MPRester as NewMPRester

    HAS_NEW_API = True
except ImportError:
    NewMPRester = None
    HAS_NEW_API = False


def _read_env_file(env_path: str = "configs/api_keys.env") -> Dict[str, str]:
    root_env = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", env_path)
    )
    values: Dict[str, str] = {}
    if not os.path.exists(root_env):
        return values

    with open(root_env, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key.startswith("MP_API_KEY") and value:
                values[key] = value
    return values


def load_api_keys(env_path: str = "configs/api_keys.env") -> Dict[str, Optional[str]]:
    """Load new, legacy, and fallback Materials Project API keys."""
    file_values = _read_env_file(env_path)

    default_key = os.getenv("MP_API_KEY") or file_values.get("MP_API_KEY")
    new_key = os.getenv("MP_API_KEY_NEW") or file_values.get("MP_API_KEY_NEW")
    legacy_key = os.getenv("MP_API_KEY_LEGACY") or file_values.get("MP_API_KEY_LEGACY")

    default_is_new_key = bool(default_key and len(default_key.strip()) == 32)

    if not new_key and default_is_new_key:
        new_key = default_key
    if not legacy_key and default_key and not default_is_new_key:
        legacy_key = default_key

    if not new_key and not legacy_key:
        raise ValueError(
            "No Materials Project key found. Set MP_API_KEY_NEW, "
            "MP_API_KEY_LEGACY, or MP_API_KEY."
        )

    return {
        "new": new_key.strip() if new_key else None,
        "legacy": legacy_key.strip() if legacy_key else None,
        "default": default_key.strip() if default_key else None,
    }


def load_api_key(env_path: str = "configs/api_keys.env") -> str:
    """Backward-compatible single-key loader."""
    keys = load_api_keys(env_path)
    return keys["default"] or keys["new"] or keys["legacy"]


class MPAdapter:
    """Robust adapter around the current mp-api and pymatgen fallback APIs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        new_api_key: Optional[str] = None,
        legacy_api_key: Optional[str] = None,
        cache_dir: str = "./data_cache",
        rate_limit_delay: float = 1.0,
    ):
        if api_key and not new_api_key and len(api_key.strip()) == 32:
            new_api_key = api_key
        if api_key and not legacy_api_key and len(api_key.strip()) != 32:
            legacy_api_key = api_key

        self.api_key = api_key or new_api_key or legacy_api_key
        self.new_api_key = new_api_key
        self.legacy_api_key = legacy_api_key
        self.cache_dir = cache_dir
        self.rate_limit_delay = rate_limit_delay
        self.json_cache_dir = os.path.join(cache_dir, "json_cache")

        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(self.json_cache_dir, exist_ok=True)

        self.legacy_rester = None
        if legacy_api_key:
            try:
                self.legacy_rester = LegacyMPRester(legacy_api_key)
            except Exception as exc:
                print(f"[WARN] Legacy Materials Project API disabled: {exc}")

        self.new_rester = None
        if HAS_NEW_API and new_api_key:
            try:
                self.new_rester = NewMPRester(new_api_key, mute_progress_bars=True)
            except Exception as exc:
                print(f"[WARN] New Materials Project API disabled: {exc}")
                if self.legacy_rester is not None:
                    print("[WARN] Falling back to legacy pymatgen MPRester.")

        if self.new_rester is None and self.legacy_rester is None:
            raise ValueError("No usable Materials Project API client could be initialized")

    def close(self) -> None:
        """Close API sessions when the installed client exposes close methods."""
        for rester in (self.new_rester, self.legacy_rester):
            close = getattr(rester, "close", None)
            if callable(close):
                close()

    def _get_json_cache_path(self, material_id: str) -> str:
        return os.path.join(self.json_cache_dir, f"{material_id}.json")

    def _load_from_json_cache(self, material_id: str) -> Optional[Dict[str, Any]]:
        path = self._get_json_cache_path(material_id)
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

        if "kpoints" in cached_data:
            cached_data["kpoints"] = np.array(cached_data["kpoints"], dtype=np.float32)
        if "energies" in cached_data:
            cached_data["energies"] = np.array(cached_data["energies"], dtype=np.float32)

        return cached_data

    def _save_to_json_cache(self, material_id: str, data: Dict[str, Any]) -> None:
        path = self._get_json_cache_path(material_id)
        tmp_path = path + ".tmp"

        serializable_data = {}
        for key, value in data.items():
            serializable_data[key] = self._to_jsonable(value)

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f)
        os.replace(tmp_path, path)

    def _to_jsonable(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(v) for v in value]
        return value

    def fetch_metadata(
        self,
        query_params: Optional[Dict[str, Any]] = None,
        fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch summary metadata used by downloader and spacegroup splitter."""
        safe_query = dict(query_params or {})
        safe_query.pop("has_bandstructure", None)

        if fields is None:
            fields = [
                "material_id",
                "formula_pretty",
                "symmetry",
                "band_gap",
                "is_gap_direct",
                "efermi",
                "num_sites",
                "theoretical",
            ]

        if self.new_rester is None:
            return self._fetch_metadata_legacy(safe_query)

        try:
            summary = self._get_summary_rester()
            docs = summary.search(**safe_query, fields=fields)
        except TypeError as exc:
            cleaned_query = self._drop_unsupported_summary_filters(safe_query)
            print(f"[WARN] Summary query rejected one filter: {exc}")
            print(f"[WARN] Retrying with: {cleaned_query}")
            summary = self._get_summary_rester()
            docs = summary.search(**cleaned_query, fields=fields)
        except Exception as exc:
            print(f"[ERROR] Metadata query failed: {type(exc).__name__}: {exc}")
            if self.legacy_rester is not None:
                print("[WARN] Retrying metadata query with legacy API.")
                return self._fetch_metadata_legacy(safe_query)
            return []

        metadata_list = []
        for doc in docs:
            doc_dict = self._doc_to_dict(doc)
            material_id = str(doc_dict.get("material_id", ""))
            if not material_id:
                continue

            symmetry = doc_dict.get("symmetry") or {}
            if hasattr(symmetry, "model_dump"):
                symmetry = symmetry.model_dump()
            elif hasattr(symmetry, "dict"):
                symmetry = symmetry.dict()

            sg_number = doc_dict.get("spacegroup_number")
            if sg_number is None and isinstance(symmetry, dict):
                sg_number = symmetry.get("number")

            metadata_list.append(
                {
                    "material_id": material_id,
                    "formula_pretty": doc_dict.get("formula_pretty"),
                    "spacegroup_number": sg_number,
                    "band_gap": doc_dict.get("band_gap"),
                    "is_direct": doc_dict.get("is_gap_direct"),
                    "efermi": doc_dict.get("efermi"),
                    "num_sites": doc_dict.get("num_sites"),
                    "theoretical": doc_dict.get("theoretical"),
                }
            )

        if not metadata_list and self.legacy_rester is not None:
            print("[WARN] New API returned no metadata; retrying with legacy API.")
            return self._fetch_metadata_legacy(safe_query)

        return metadata_list

    def _fetch_metadata_legacy(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch metadata through the legacy API when the user has an old key."""
        if self.legacy_rester is None:
            print("[ERROR] Legacy metadata query requested, but no legacy key/client is available.")
            return []

        criteria: Dict[str, Any] = {}

        band_gap = query_params.get("band_gap")
        if isinstance(band_gap, (list, tuple)) and len(band_gap) == 2:
            criteria["band_gap"] = {"$gte": band_gap[0], "$lte": band_gap[1]}

        num_sites = query_params.get("num_sites")
        if isinstance(num_sites, (list, tuple)) and len(num_sites) == 2:
            criteria["nsites"] = {"$gte": num_sites[0], "$lte": num_sites[1]}

        if query_params.get("material_ids"):
            criteria["material_id"] = {"$in": query_params["material_ids"]}

        criteria["has_bandstructure"] = True

        properties = [
            "material_id",
            "pretty_formula",
            "spacegroup.number",
            "band_gap",
            "efermi",
            "nsites",
        ]

        try:
            docs = self.legacy_rester.query(criteria=criteria, properties=properties)
        except Exception as exc:
            if "has_bandstructure" not in criteria:
                print(f"[ERROR] Legacy metadata query failed: {type(exc).__name__}: {exc}")
                return []
            print(f"[WARN] Legacy query rejected has_bandstructure: {exc}")
            criteria.pop("has_bandstructure", None)
            try:
                docs = self.legacy_rester.query(criteria=criteria, properties=properties)
            except Exception as retry_exc:
                print(
                    "[ERROR] Legacy metadata query failed after retry: "
                    f"{type(retry_exc).__name__}: {retry_exc}"
                )
                return []

        metadata_list = []
        for doc in docs:
            material_id = doc.get("material_id")
            if not material_id:
                continue

            spacegroup_number = doc.get("spacegroup.number")
            if spacegroup_number is None and isinstance(doc.get("spacegroup"), dict):
                spacegroup_number = doc["spacegroup"].get("number")

            metadata_list.append(
                {
                    "material_id": str(material_id),
                    "formula_pretty": doc.get("pretty_formula"),
                    "spacegroup_number": spacegroup_number,
                    "band_gap": doc.get("band_gap"),
                    "is_direct": doc.get("is_gap_direct"),
                    "efermi": doc.get("efermi"),
                    "num_sites": doc.get("nsites"),
                    "theoretical": doc.get("theoretical"),
                }
            )

        return metadata_list

    def _get_summary_rester(self):
        if hasattr(self.new_rester, "materials") and hasattr(
            self.new_rester.materials, "summary"
        ):
            return self.new_rester.materials.summary
        if hasattr(self.new_rester, "summary"):
            return self.new_rester.summary
        raise RuntimeError("Installed mp-api client has no summary rester")

    def _drop_unsupported_summary_filters(self, query: Dict[str, Any]) -> Dict[str, Any]:
        supported = {
            "material_ids",
            "chemsys",
            "elements",
            "exclude_elements",
            "formula",
            "band_gap",
            "num_sites",
            "num_elements",
            "is_gap_direct",
            "is_metal",
            "theoretical",
        }
        return {key: value for key, value in query.items() if key in supported}

    def _doc_to_dict(self, doc: Any) -> Dict[str, Any]:
        if hasattr(doc, "model_dump"):
            return doc.model_dump()
        if hasattr(doc, "dict"):
            return doc.dict()
        if isinstance(doc, dict):
            return doc
        return {
            key: getattr(doc, key)
            for key in dir(doc)
            if not key.startswith("_") and not callable(getattr(doc, key))
        }

    def fetch_band_structure(self, material_id: str) -> Dict[str, Any]:
        """Fetch and parse a line-mode band structure for one material."""
        cached_data = self._load_from_json_cache(material_id)
        if cached_data:
            return cached_data

        bs_obj = None
        errors = []

        if self.new_rester is not None:
            bs_obj, err = self._fetch_band_structure_new_api(material_id)
            if err:
                errors.append(err)

        if bs_obj is None:
            bs_obj, err = self._fetch_band_structure_legacy_api(material_id)
            if err:
                errors.append(err)

        time.sleep(self.rate_limit_delay)

        if bs_obj is None:
            return {
                "material_id": material_id,
                "error": "; ".join(errors) or "line-mode band structure unavailable",
            }

        if not getattr(bs_obj, "branches", None):
            return {"material_id": material_id, "error": "line-mode band structure unavailable"}

        try:
            result = self._parse_band_structure(material_id, bs_obj)
        except Exception as exc:
            return {"material_id": material_id, "error": f"Parse Error: {type(exc).__name__}: {exc}"}

        self._save_to_json_cache(material_id, result)
        return result

    def _fetch_band_structure_new_api(self, material_id: str):
        try:
            if hasattr(self.new_rester, "get_bandstructure_by_material_id"):
                return (
                    self.new_rester.get_bandstructure_by_material_id(
                        material_id, line_mode=True
                    ),
                    None,
                )

            materials = getattr(self.new_rester, "materials", None)
            bs_rester = getattr(materials, "electronic_structure_bandstructure", None)
            if bs_rester is not None:
                return (
                    bs_rester.get_bandstructure_from_material_id(
                        material_id=material_id, line_mode=True
                    ),
                    None,
                )

            electronic_structure = getattr(self.new_rester, "electronic_structure", None)
            if electronic_structure is not None and hasattr(
                electronic_structure, "get_bandstructure_from_material_id"
            ):
                return (
                    electronic_structure.get_bandstructure_from_material_id(
                        material_id=material_id, line_mode=True
                    ),
                    None,
                )

            return None, "New API has no band structure route"
        except Exception as exc:
            return None, f"New API failed for {material_id}: {type(exc).__name__}: {exc}"

    def _fetch_band_structure_legacy_api(self, material_id: str):
        if self.legacy_rester is None:
            return None, "Legacy API unavailable"

        try:
            return (
                self.legacy_rester.get_bandstructure_by_material_id(
                    material_id, line_mode=True
                ),
                None,
            )
        except Exception as exc:
            return None, f"Legacy API failed for {material_id}: {type(exc).__name__}: {exc}"

    def _parse_band_structure(self, material_id: str, bs_obj: Any) -> Dict[str, Any]:
        kpoints = np.array([kp.frac_coords for kp in bs_obj.kpoints], dtype=np.float32)

        if bs_obj.is_spin_polarized:
            energies = np.stack(
                [
                    np.array(bs_obj.bands[Spin.up], dtype=np.float32),
                    np.array(bs_obj.bands[Spin.down], dtype=np.float32),
                ],
                axis=0,
            )
            is_spin_polarized = True
            num_bands = energies.shape[-2]
        else:
            energies = np.array(bs_obj.bands[Spin.up], dtype=np.float32)
            is_spin_polarized = False
            num_bands = energies.shape[0]

        labels = []
        for idx, kp in enumerate(bs_obj.kpoints):
            if kp.label:
                labels.append(
                    {
                        "index": idx,
                        "label": str(kp.label),
                        "frac_coords": kp.frac_coords.tolist(),
                    }
                )

        branches = []
        for branch in bs_obj.branches:
            branches.append(
                {
                    "start_index": int(branch["start_index"]),
                    "end_index": int(branch["end_index"]),
                    "name": branch.get("name", ""),
                }
            )

        return {
            "material_id": material_id,
            "kpoints": kpoints,
            "energies": energies,
            "efermi": float(bs_obj.efermi),
            "kpath_labels": labels,
            "branches": branches,
            "is_spin_polarized": is_spin_polarized,
            "num_bands": int(num_bands),
            "num_kpoints": int(energies.shape[-1]),
        }

    def save_to_hdf5(self, bs_data: Dict[str, Any]) -> None:
        """Append one parsed band structure to mp_bands.h5 atomically per group."""
        material_id = bs_data["material_id"]
        h5_path = os.path.join(self.cache_dir, "mp_bands.h5")

        with h5py.File(h5_path, "a") as f:
            if material_id in f:
                del f[material_id]

            grp = f.create_group(material_id)
            for key, value in bs_data.items():
                if key == "metadata":
                    meta_grp = grp.create_group("metadata")
                    for meta_key, meta_value in value.items():
                        if meta_value is not None:
                            meta_grp.attrs[meta_key] = meta_value
                            grp.attrs[meta_key] = meta_value
                    if value.get("spacegroup_number") is not None:
                        grp.attrs["spacegroup"] = value["spacegroup_number"]
                    if value.get("is_direct") is not None:
                        grp.attrs["is_direct"] = value["is_direct"]
                    continue

                if key == "kpath_labels":
                    grp.attrs[key] = json.dumps(self._to_jsonable(value), ensure_ascii=False)
                    continue

                if isinstance(value, np.ndarray):
                    grp.create_dataset(key, data=value)
                elif isinstance(value, (list, tuple, dict)):
                    grp.attrs[key] = json.dumps(self._to_jsonable(value), ensure_ascii=False)
                elif value is not None:
                    grp.attrs[key] = value

    def save_metadata_to_json(self, metadata_list: List[Dict[str, Any]]) -> None:
        """Save metadata list without duplicating material IDs."""
        json_path = os.path.join(self.cache_dir, "mp_metadata.json")
        existing: List[Dict[str, Any]] = []

        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                existing = json.load(f)

        by_id = {item["material_id"]: item for item in existing if item.get("material_id")}
        for item in metadata_list:
            material_id = item.get("material_id")
            if material_id:
                by_id[material_id] = item

        merged = list(by_id.values())
        tmp_path = json_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, json_path)
