#!/usr/bin/env python3
"""make_toc_graphic.py -- table-of-contents graphic (8 cm x 4 cm, 600 dpi TIFF) from the result tables.

Usage: python make_toc_graphic.py --data DIR --out TOC_graphic.tif
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

WINDOW = (-32.0, -25.8)
CANDIDATES = ["M0107", "M0116", "M0119", "M0120"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default="TOC_graphic.tif")
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 6, "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.5})
    df = pd.read_csv(f"{a.data}/library_results.csv").merge(pd.read_csv(f"{a.data}/dipoles.csv"), on="id")
    o = df[df.scaffold == "oxane"]
    fig, ax = plt.subplots(figsize=(8 / 2.54, 4 / 2.54))
    ax.axvspan(*WINDOW, color="#009E73", alpha=.13, lw=0)
    ax.scatter(o.ESPmin_kcal_mol, o.dipole_debye, c="#BBBBBB", s=3, linewidth=0)
    w = o[o.id.isin(CANDIDATES)]
    ax.scatter(w.ESPmin_kcal_mol, w.dipole_debye, c="#D55E00", s=10, edgecolor="black", linewidth=.3, zorder=3)
    p = o[o.id == "M0027"]
    ax.scatter(p.ESPmin_kcal_mol, p.dipole_debye, marker="*", c="black", s=14)
    ax.annotate("THP", (p.ESPmin_kcal_mol.iloc[0], p.dipole_debye.iloc[0]), xytext=(2, 3), textcoords="offset points", fontsize=5.5)
    ax.text(-28.9, 0.32, "ESP$_{min}$ window", color="#009E73", ha="center", fontsize=5.5)
    ax.annotate("4 candidates", xy=(-28.37, 3.83), xytext=(-24.0, 4.05), color="#D55E00", fontsize=5.5,
                arrowprops=dict(arrowstyle="-", color="#D55E00", lw=0.4))
    ax.set_xlim(-38.5, -19.5); ax.set_ylim(0.15, 4.4); ax.set_xticks([-35, -30, -25, -20]); ax.set_yticks([1, 2, 3, 4])
    ax.set_xlabel("ESP$_{min}$ / kcal mol$^{-1}$", labelpad=1); ax.set_ylabel("Dipole / D", labelpad=1)
    ax.tick_params(length=2, pad=1)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.25, top=0.97)
    fig.savefig(a.out, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    print("written", a.out)


if __name__ == "__main__":
    main()
