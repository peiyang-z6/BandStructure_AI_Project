"""
Publication-grade band-structure visualization helpers.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _curvature_at_extremum(band: np.ndarray, mode: str) -> tuple[int, float]:
    band = np.asarray(band, dtype=np.float32)
    idx = int(np.argmax(band) if mode == "max" else np.argmin(band))
    start = max(0, idx - 2)
    end = min(len(band), idx + 3)
    if end - start < 3:
        start = max(0, min(idx, len(band) - 3))
        end = min(len(band), start + 3)
    k_local = np.arange(start, end, dtype=np.float32) - float(idx)
    e_local = band[start:end].astype(np.float32)
    if len(k_local) < 3:
        return idx, 0.0
    coeff = np.polyfit(k_local, e_local, deg=2)
    curvature = 2.0 * float(coeff[0])
    return idx, curvature


def _effective_mass_from_curvature(curvature: float) -> float:
    if abs(curvature) < 1e-6:
        return float("inf") if curvature >= 0 else float("-inf")
    return float(1.0 / curvature)


def _format_mass(value: float) -> str:
    if not np.isfinite(value):
        return r"\infty" if value > 0 else r"-\infty"
    return f"{value:.2f}"


def _annotate_effective_mass(
    ax,
    k_axis: np.ndarray,
    true_band: np.ndarray,
    pred_band: np.ndarray | None,
    mode: str,
    label: str,
    text_offset: tuple[int, int],
) -> None:
    if pred_band is None:
        return

    true_idx, true_curv = _curvature_at_extremum(true_band, mode)
    pred_idx, pred_curv = _curvature_at_extremum(pred_band, mode)
    true_mass = _effective_mass_from_curvature(true_curv)
    pred_mass = _effective_mass_from_curvature(pred_curv)

    same_sign = np.sign(true_curv) == np.sign(pred_curv)
    if abs(true_curv) < 1e-6 or abs(pred_curv) < 1e-6:
        same_sign = abs(true_curv - pred_curv) < 1e-6
    color = "#2CA02C" if same_sign else "#D62728"
    y_value = float(true_band[true_idx])
    pred_y = float(pred_band[pred_idx])

    ax.scatter([true_idx], [y_value], s=22, color=color, zorder=5)
    ax.scatter([pred_idx], [pred_y], s=22, facecolors="none", edgecolors=color, linewidths=1.1, zorder=5)
    ax.annotate(
        rf"{label}: $m^*_{{pred}}={_format_mass(pred_mass)},\ m^*_{{true}}={_format_mass(true_mass)}$",
        xy=(k_axis[true_idx], y_value),
        xytext=text_offset,
        textcoords="offset points",
        fontsize=8,
        color=color,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "->", "color": color, "lw": 0.8},
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": color, "alpha": 0.86},
    )


def latex_k_label(label: str) -> str:
    cleaned = str(label).strip()
    if cleaned in {"G", "Gamma", "\\Gamma"}:
        return r"$\Gamma$"
    if cleaned.startswith("\\"):
        return f"${cleaned}$"
    return cleaned


def normalize_kpath_labels(kpath_labels: Any, seq_len: int) -> List[Dict[str, Any]]:
    if isinstance(kpath_labels, str):
        try:
            kpath_labels = json.loads(kpath_labels)
        except json.JSONDecodeError:
            kpath_labels = []
    if not isinstance(kpath_labels, Iterable):
        kpath_labels = []

    ticks: List[Dict[str, Any]] = []
    seen = set()
    for item in kpath_labels:
        if isinstance(item, dict):
            idx = item.get("index")
            label = item.get("label")
        elif isinstance(item, Sequence) and len(item) >= 2:
            idx, label = item[0], item[1]
        else:
            continue
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            continue
        idx = max(0, min(seq_len - 1, idx))
        key = (idx, str(label))
        if key in seen:
            continue
        seen.add(key)
        ticks.append({"index": idx, "label": latex_k_label(str(label))})

    if not ticks:
        ticks = [
            {"index": 0, "label": r"$\Gamma$"},
            {"index": seq_len - 1, "label": r"$\Gamma$"},
        ]
    return ticks


def plot_band_overlay(
    ax,
    true_vbm: np.ndarray,
    true_cbm: np.ndarray,
    pred_vbm: np.ndarray | None = None,
    pred_cbm: np.ndarray | None = None,
    kpath_labels: Any = None,
    title: str | None = None,
) -> None:
    seq_len = len(true_vbm)
    k_axis = np.arange(seq_len)

    ax.plot(k_axis, true_vbm, color="black", linewidth=1.5, label="DFT VBM")
    ax.plot(k_axis, true_cbm, color="black", linewidth=1.5, alpha=0.65, label="DFT CBM")
    if pred_vbm is not None:
        ax.plot(k_axis, pred_vbm, color="#1F77B4", linestyle="--", linewidth=1.35, label="Model VBM")
    if pred_cbm is not None:
        ax.plot(k_axis, pred_cbm, color="#D62728", linestyle="--", linewidth=1.35, label="Model CBM")

    _annotate_effective_mass(ax, k_axis, true_vbm, pred_vbm, "max", "VBM", (10, 18))
    _annotate_effective_mass(ax, k_axis, true_cbm, pred_cbm, "min", "CBM", (10, -18))

    ticks = normalize_kpath_labels(kpath_labels, seq_len)
    for tick in ticks:
        ax.axvline(tick["index"], color="0.72", linestyle="--", linewidth=0.8, zorder=0)
    ax.axhline(0.0, color="#D62728", linestyle="--", linewidth=1.05, label=r"$E_F$")
    ax.set_xticks([tick["index"] for tick in ticks])
    ax.set_xticklabels([tick["label"] for tick in ticks])
    ax.set_xlim(0, seq_len - 1)
    ax.set_ylabel("Energy (eV)")
    if title:
        ax.set_title(title)
    ax.grid(axis="y", alpha=0.18)


def save_band_overlay_grid(
    output_path: str,
    true_tensors: np.ndarray,
    recon_tensors: np.ndarray,
    material_ids: Sequence[str],
    kpath_labels_by_material: Dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    selected: Sequence[int],
) -> None:
    rows = max(1, len(selected))
    fig, axes = plt.subplots(rows, 1, figsize=(8.2, max(3.0, 2.7 * rows)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, idx in zip(axes, selected):
        material_id = str(material_ids[idx])
        plot_band_overlay(
            ax,
            true_tensors[idx, 0, :, 0],
            true_tensors[idx, 1, :, 0],
            recon_tensors[idx, 0, :, 0],
            recon_tensors[idx, 1, :, 0],
            kpath_labels_by_material.get(material_id),
            f"{material_id} | DFT gap={y_true[idx]:.3f} eV, pred={y_pred.reshape(-1)[idx]:.3f} eV",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)))
    axes[-1].set_xlabel("High-symmetry k-path")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
