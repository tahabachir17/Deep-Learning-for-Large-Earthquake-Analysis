"""Plotting utilities for notebook-style evaluation figures."""

from __future__ import annotations

from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import rms


def plot_figure11(results_df: pd.DataFrame, event_order: List[str], output_path: Optional[str] = None, cmap: str = "viridis") -> None:
    mpl.rcParams.update({"font.family": "serif", "font.size": 12, "figure.dpi": 150})
    present_events = results_df["event_id"].unique()
    event_order = [event for event in event_order if event in present_events]
    case_order = sorted(results_df["case"].unique())
    norm = mpl.colors.Normalize(vmin=float(results_df["median_distance_deg"].min()), vmax=float(results_df["median_distance_deg"].max()))
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(len(case_order), 1, figsize=(10, 3.8 * len(case_order)), sharex=True, sharey=True)
    if len(case_order) == 1:
        axes = [axes]
    y_lim = (-1.05, 1.30)
    rms_y = -0.90
    scatter = None
    for ax, case in zip(axes, case_order):
        case_df = results_df[results_df["case"] == case]
        violin_data = []
        violin_pos = []
        for index, event_id in enumerate(event_order):
            values = case_df.loc[case_df["event_id"] == event_id, "error"].dropna().values
            if len(values) > 1:
                violin_data.append(values)
                violin_pos.append(index)
        if violin_data:
            violin = ax.violinplot(violin_data, positions=violin_pos, widths=0.70, showmedians=True, showextrema=True)
            for body in violin["bodies"]:
                body.set_facecolor("#FFF2CC")
                body.set_edgecolor("#F6B26B")
                body.set_alpha(0.65)
        for index, event_id in enumerate(event_order):
            sub = case_df[case_df["event_id"] == event_id]
            if sub.empty:
                continue
            jitter = rng.uniform(-0.14, 0.14, size=len(sub))
            scatter = ax.scatter(np.full(len(sub), index) + jitter, sub["error"].values, c=sub["median_distance_deg"].values, cmap=cmap, norm=norm, s=18 if len(sub) > 300 else 30, alpha=0.7, edgecolors="k", linewidths=0.3)
            ax.text(index, rms_y, f"{rms(sub['error']):.2f}", ha="center", va="center", fontsize=10, fontweight="bold")
        ax.axhline(0, linestyle="--", linewidth=1.2, color="#AAAAAA", alpha=0.85)
        ax.set_ylim(*y_lim)
        ax.set_ylabel("Error (Mw)")
        ax.set_title(case)
    axes[-1].set_xticks(range(len(event_order)))
    axes[-1].set_xticklabels(event_order, rotation=20, ha="right")
    if scatter is not None:
        fig.colorbar(scatter, ax=axes, label="Median epicentral distance (deg)")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()
