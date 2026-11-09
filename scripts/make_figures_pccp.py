#!/usr/bin/env python3
"""make_figures_pccp.py -- the same figures sized for PCCP (8.3 cm / 17.1 cm columns), also written as 600 dpi TIFF.

Usage (folder with library_results.csv, dipoles.csv, pa.csv, lumo.csv):
    python make_figures.py [--data DIR] [--out figures]
Writes figures/fig1..fig5 as PDF (vector) and PNG (300 dpi).
Okabe-Ito colour-blind-safe colours; every group is also distinguished by marker shape.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "font.family": "DejaVu Sans",
})
BLUE, ORANGE, GREEN, VERMIL, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#7F7F7F"

FRONT_IDS = ["M0106", "M0125", "M0130", "M0102", "M0120"]
WINDOW_IDS = ["M0107", "M0116", "M0119", "M0120"]
WINDOW = (-32.0, -25.8)
PARENT = "M0027"
NIST = {"THF": (204.4, 196.49), "THP": (205.3, 196.65)}  # (computed electronic PA, NIST experimental PA)


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out, f"{name}.tif"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("written", name)


def load(d):
    lib = pd.read_csv(os.path.join(d, "library_results.csv"))
    for f in ("dipoles.csv", "pa.csv", "lumo.csv"):
        lib = lib.merge(pd.read_csv(os.path.join(d, f)).drop(columns=["ring_opened"], errors="ignore"),
                        on="id", how="left")
    return lib


def fig1(df, out):
    fig, ax = plt.subplots(figsize=(8.3/2.54, 2.6))
    rng = np.random.default_rng(1)
    for i, (sc, c, m) in enumerate([("oxetane", VERMIL, "s"), ("oxane", BLUE, "o")]):
        s = df[df.scaffold == sc]
        ax.scatter(i + rng.uniform(-.18, .18, len(s)), s.RSE_release_kcal_mol, c=c, marker=m,
                   s=14, alpha=.8, edgecolor="white", linewidth=.3, label=f"{sc} (n={len(s)})")
    ax.axhline(15, color=GREY, lw=.7, ls="--")
    ax.text(1.45, 15.8, "RSE = 15 kcal mol$^{-1}$", ha="right", color=GREY, fontsize=7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["oxetane", "oxane"]); ax.set_xlim(-.5, 1.5)
    ax.set_ylabel("Ring-strain release / kcal mol$^{-1}$")
    ax.legend(frameon=False, loc="center", bbox_to_anchor=(0.5, 0.42))
    save(fig, out, "fig3_ring_strain")


def fig2(df, out):
    o = df[df.scaffold == "oxane"]
    fig, axs = plt.subplots(1, 2, figsize=(17.1/2.54, 2.6))
    for ax, col, lab in [(axs[0], "ESPmin_kcal_mol", "ESP$_{min}$ / kcal mol$^{-1}$"),
                         (axs[1], "Ebind_global_kcal_mol", "Global Na$^+$ binding energy / kcal mol$^{-1}$")]:
        g = [o[o.n_F_atoms == k][col].values for k in range(6)]
        ax.boxplot(g, positions=range(6), widths=.55, showfliers=False,
                   medianprops=dict(color=BLUE), boxprops=dict(color=GREY), whiskerprops=dict(color=GREY),
                   capprops=dict(color=GREY))
        for k, v in enumerate(g):
            ax.scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-.15, .15, len(v)), v,
                       s=6, c=BLUE, alpha=.45, linewidth=0)
        ax.set_xlabel("Number of F atoms"); ax.set_ylabel(lab)
    axs[0].set_title("(a)", loc="left")
    axs[1].set_title("(b)", loc="left")
    fig.tight_layout()
    save(fig, out, "fig1_F_dependence")


def fig3(df, out):
    d = df.dropna(subset=["Ebind_noF_kcal_mol"])
    d = d[d.Ebind_noF_kcal_mol.notna()]
    fig, axs = plt.subplots(1, 2, figsize=(17.1/2.54, 2.8))
    for ax, col, c, mk, ttl in [(axs[0], "Ebind_noF_kcal_mol", BLUE, "o", "(a)"),
                                (axs[1], "Ebind_global_kcal_mol", ORANGE, "s", "(b)")]:
        ax.scatter(d.ESPmin_kcal_mol, d[col], c=c, marker=mk, s=14, alpha=.85, edgecolor="white", linewidth=.3)
        r = np.corrcoef(d.ESPmin_kcal_mol, d[col])[0, 1]
        k = np.polyfit(d.ESPmin_kcal_mol, d[col], 1)
        xs = np.linspace(d.ESPmin_kcal_mol.min(), d.ESPmin_kcal_mol.max(), 2)
        ax.plot(xs, np.polyval(k, xs), color=GREY, lw=.8)
        ax.text(.04, .94, f"r = {r:.2f}  (n = {len(d)})", transform=ax.transAxes, va="top")
        ax.set_xlabel("ESP$_{min}$ / kcal mol$^{-1}$"); ax.set_ylabel("Na$^+$ binding energy / kcal mol$^{-1}$")
        ax.set_title(ttl, loc="left")
    fig.tight_layout()
    save(fig, out, "fig2_chelation_diagnostic")


def fig4(df, out):
    o = df[df.scaffold == "oxane"]
    fig, ax = plt.subplots(figsize=(7.7/2.54, 3.2))
    ax.axvspan(*WINDOW, color=GREEN, alpha=.12, lw=0)
    sc = ax.scatter(o.ESPmin_kcal_mol, o.dipole_debye, c=o.proton_affinity_kcal_mol, cmap="viridis_r",
                    s=16, edgecolor="white", linewidth=.3)
    f = o[o.id.isin(FRONT_IDS)]
    ax.scatter(f.ESPmin_kcal_mol, f.dipole_debye, s=46, facecolor="none", edgecolor=VERMIL, linewidth=1.1,
               marker="D", label="3-objective front")
    w = o[o.id.isin(WINDOW_IDS)]
    ax.scatter(w.ESPmin_kcal_mol, w.dipole_debye, s=40, facecolor="none", edgecolor="black", linewidth=1.0,
               marker="s", label="in window")
    p = o[o.id == PARENT]
    ax.scatter(p.ESPmin_kcal_mol, p.dipole_debye, s=50, c="black", marker="*", label="parent oxane")
    for _, r in pd.concat([f, w]).drop_duplicates("id").iterrows():
        ax.annotate(r.id[1:], (r.ESPmin_kcal_mol, r.dipole_debye), xytext=(3, 3), textcoords="offset points",
                    fontsize=6)
    ax.set_xlabel("ESP$_{min}$ / kcal mol$^{-1}$"); ax.set_ylabel("Dipole moment / D")
    cb = fig.colorbar(sc, ax=ax, pad=.02); cb.set_label("Proton affinity / kcal mol$^{-1}$", fontsize=7)
    ax.set_xlim(-38.5, -18.2)
    ax.text(np.mean(WINDOW), 3.6, "ESP$_{min}$\nwindow", ha="center", va="top", fontsize=6.5, color=GREEN)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6.5, handletextpad=0.3, columnspacing=0.8)
    save(fig, out, "fig4_window_and_fronts")


def fig5(df, out):
    o = df[df.scaffold == "oxane"]
    fig, axs = plt.subplots(1, 2, figsize=(17.1/2.54, 2.6), gridspec_kw=dict(width_ratios=[1, 1.3]))
    ax = axs[0]
    x = np.arange(2); w = .36
    ax.bar(x - w / 2, [v[1] for v in NIST.values()], w, color=GREY, label="NIST")
    ax.bar(x + w / 2, [v[0] for v in NIST.values()], w, color=BLUE, label="this work (electronic)")
    for i, v in enumerate(NIST.values()):
        ax.text(i - w / 2, v[1] + .3, f"{v[1]:.1f}", ha="center", fontsize=6.5)
        ax.text(i + w / 2, v[0] + .3, f"{v[0]:.1f}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(list(NIST)); ax.set_ylim(185, 216)
    ax.set_ylabel("Proton affinity / kcal mol$^{-1}$"); ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=1)
    ax.set_title("(a)", loc="left")
    ax = axs[1]
    ax.scatter(o.ESPmin_kcal_mol, o.proton_affinity_kcal_mol, s=12, c=BLUE, alpha=.8, edgecolor="white",
               linewidth=.3)
    r = np.corrcoef(o.ESPmin_kcal_mol, o.proton_affinity_kcal_mol)[0, 1]
    ax.text(.04, .06, f"r = {r:.2f}  (n = {len(o)})", transform=ax.transAxes)
    ax.set_xlabel("ESP$_{min}$ / kcal mol$^{-1}$"); ax.set_ylabel("Proton affinity / kcal mol$^{-1}$")
    ax.set_title("(b)", loc="left")
    fig.tight_layout()
    save(fig, out, "fig5_PA_validation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load(a.data)
    for f in (fig1, fig2, fig3, fig4, fig5):
        f(df, a.out)


if __name__ == "__main__":
    main()
