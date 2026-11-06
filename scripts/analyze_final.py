#!/usr/bin/env python3
"""
analyze_final.py -- the analysis we actually need after looking at the real library data.

Why a second analysis script?  The first run (analyze_library.py) showed three things:
  1. Global (gas-phase) Na+ binding energies of fluorinated molecules are dominated by F...Na+ chelation
     (Pearson r with the F-free binding energy only 0.22, n = 40), so they are a poor desolvation descriptor.
  2. ESPmin (the most negative potential on the 0.001 a.u. surface, M06-2X/def2-TZVP/ddCOSMO) tracks the
     F-free binding energy closely (Pearson r = 0.97, n = 40) and is available for all molecules.
  3. The ring-strain release separates the rings perfectly (oxetane 24.0-28.4, oxane -2.1-6.0 kcal/mol) but
     has no useful resolution INSIDE the oxane family (sd 1.4 kcal/mol, smaller than the protocol uncertainty).
So here:  strained rings are removed by a constraint (RSE > --max-rse), and the oxanes are ranked on
  f1 = ESPmin (less negative = weaker O site = easier desolvation)      maximise
  f2 = dipole moment (rank-only polarity proxy)                          maximise
An optional acid-initiation descriptor (proton affinity, from compute_pa.py) is added when available:
  f3 = proton affinity (lower = harder to protonate = more resistant to acid-initiated ring opening)   minimise

Usage
    python analyze_final.py --results library_results.csv --dipoles dipoles.csv [--pa pa.csv] --out analysis_final
"""
import argparse
import os

import numpy as np
import pandas as pd

from analyze_library import non_dominated_ranks


def load(results, dipoles, pa):
    df = pd.read_csv(results).merge(pd.read_csv(dipoles)[["id", "dipole_debye"]], on="id", how="left")
    if pa and os.path.exists(pa):
        df = df.merge(pd.read_csv(pa)[["id", "proton_affinity_kcal_mol"]], on="id", how="left")
    return df


def front(df, cols_max, cols_min=()):
    cols = list(cols_max) + list(cols_min)
    d = df.dropna(subset=cols).copy()
    F = np.column_stack([-d[c].to_numpy(float) for c in cols_max] + [d[c].to_numpy(float) for c in cols_min])
    d["rank"] = non_dominated_ranks(F)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="library_results.csv")
    ap.add_argument("--dipoles", default="dipoles.csv")
    ap.add_argument("--pa", default="pa.csv")
    ap.add_argument("--max-rse", type=float, default=15.0, help="rings with RSE above this are 'strained' and excluded")
    ap.add_argument("--esp-margin", type=float, default=4.0, help="kcal/mol; about 2x the 1.65 kcal/mol ESPmin scatter against Niu's table")
    ap.add_argument("--dipole-margin", type=float, default=0.3, help="Debye")
    ap.add_argument("--out", default="analysis_final")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load(a.results, a.dipoles, a.pa)
    L = []

    strained = df[df.RSE_release_kcal_mol > a.max_rse]
    ok = df[df.RSE_release_kcal_mol <= a.max_rse].copy()
    L.append(f"ring-strain constraint: RSE > {a.max_rse:g} kcal/mol removes {len(strained)} molecules "
             f"({', '.join(sorted(strained.scaffold.unique())) or 'none'}); {len(ok)} remain")
    if len(strained):
        L.append(f"  strained RSE range {strained.RSE_release_kcal_mol.min():.1f}..{strained.RSE_release_kcal_mol.max():.1f};"
                 f" retained RSE range {ok.RSE_release_kcal_mol.min():.1f}..{ok.RSE_release_kcal_mol.max():.1f} kcal/mol")

    # parent reference (unsubstituted oxane)
    par = ok[ok.ring_smiles == "C1CCCCO1"]
    par = par.iloc[0] if len(par) else None
    if par is not None:
        L.append(f"parent oxane {par.id}: ESPmin {par.ESPmin_kcal_mol:.2f}  dipole {par.dipole_debye:.2f}"
                 + (f"  PA {par.proton_affinity_kcal_mol:.1f}" if "proton_affinity_kcal_mol" in ok and pd.notna(par.get("proton_affinity_kcal_mol")) else ""))

    # chelation diagnostic
    s = df.dropna(subset=["Ebind_noF_kcal_mol"])
    if len(s) > 5:
        L.append(f"\nchelation diagnostic (n={len(s)} with an F-free reference):")
        L.append(f"  Pearson(F-free binding, ESPmin)   = {s[['Ebind_noF_kcal_mol','ESPmin_kcal_mol']].corr().iloc[0,1]:.2f}")
        L.append(f"  Pearson(global binding, ESPmin)   = {s[['Ebind_global_kcal_mol','ESPmin_kcal_mol']].corr().iloc[0,1]:.2f}")
        L.append(f"  Pearson(global, F-free binding)   = {s[['Ebind_global_kcal_mol','Ebind_noF_kcal_mol']].corr().iloc[0,1]:.2f}")

    # ESPmin vs number of fluorine atoms
    L.append("\nESPmin by number of F atoms (retained molecules):")
    g = ok.groupby("n_F_atoms").ESPmin_kcal_mol.agg(["mean", "count"]).round(2)
    L.append(g.to_string())

    # main front (two objectives, plus PA if present)
    has_pa = "proton_affinity_kcal_mol" in ok and ok.proton_affinity_kcal_mol.notna().sum() > 0.8 * len(ok)
    cols_min = ["proton_affinity_kcal_mol"] if has_pa else []
    d = front(ok, ["ESPmin_kcal_mol", "dipole_debye"], cols_min)
    fr = d[d["rank"] == 1].sort_values("ESPmin_kcal_mol", ascending=False)
    fr.to_csv(os.path.join(a.out, "front_main.csv"), index=False)
    d.sort_values(["rank", "ESPmin_kcal_mol"], ascending=[True, False]).to_csv(os.path.join(a.out, "ranks_main.csv"), index=False)
    L.append(f"\nmain front ({'ESPmin, dipole, proton affinity' if has_pa else 'ESPmin, dipole'}): {len(fr)} of {len(d)} molecules")
    show = ["id", "ring_smiles", "n_F_atoms", "ESPmin_kcal_mol", "dipole_debye"] + (["proton_affinity_kcal_mol"] if has_pa else []) + ["Ebind_global_kcal_mol", "RSE_release_kcal_mol"]
    L.append(fr[show].round(2).to_string(index=False))

    # molecules that beat the parent by margins that exceed the protocol scatter
    if par is not None:
        cond = (ok.ESPmin_kcal_mol > par.ESPmin_kcal_mol + a.esp_margin) & (ok.dipole_debye > par.dipole_debye + a.dipole_margin)
        if has_pa:
            cond &= ok.proton_affinity_kcal_mol < par.proton_affinity_kcal_mol
        good = ok[cond].sort_values("dipole_debye", ascending=False)
        good.to_csv(os.path.join(a.out, "better_than_parent.csv"), index=False)
        L.append(f"\nbetter than the parent oxane by margins (ESPmin +{a.esp_margin:g} kcal/mol, dipole +{a.dipole_margin:g} D"
                 + (", lower proton affinity" if has_pa else "") + f"): {len(good)} of {len(ok)}")
        L.append("  number of F atoms among them: " + str(good.n_F_atoms.value_counts().sort_index().to_dict()))

    # sensitivity of the front
    base = set(fr.id)
    variants = {}
    v = front(ok, ["Ebind_global_kcal_mol", "dipole_debye"], cols_min); variants["desolvation = global binding energy (chelation included)"] = set(v[v["rank"] == 1].id)
    if "Volume_A3" in ok:
        ok2 = ok.assign(dipole_per_vol=ok.dipole_debye / ok.Volume_A3)
        v = front(ok2, ["ESPmin_kcal_mol", "dipole_per_vol"], cols_min); variants["polarity = dipole per volume"] = set(v[v["rank"] == 1].id)
    v = front(ok, ["ESPmin_kcal_mol", "dipole_debye"], ["RSE_release_kcal_mol"] + cols_min); variants["RSE added as a third objective"] = set(v[v["rank"] == 1].id)
    v = front(ok, ["ESPmin_kcal_mol"], cols_min); variants["ESPmin only (no polarity objective)"] = set(v[v["rank"] == 1].id)
    L.append("\nsensitivity of the main front (Jaccard overlap with the main front):")
    rows = []
    for k, sset in variants.items():
        j = len(base & sset) / max(1, len(base | sset))
        rows.append({"variant": k, "size": len(sset), "overlap": len(base & sset), "jaccard": round(j, 2)})
        L.append(f"  {k:55s} size {len(sset):3d} overlap {len(base & sset):3d} jaccard {j:.2f}")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "sensitivity_final.csv"), index=False)
    robust = set(base)
    for sset in variants.values():
        robust &= sset
    L.append(f"members that stay on the front in EVERY variant: {sorted(robust)}")

    # trade-off check between the objectives
    L.append(f"\ncorrelation between objectives (retained molecules): Pearson(ESPmin, dipole) = "
             f"{ok[['ESPmin_kcal_mol','dipole_debye']].corr().iloc[0,1]:.2f}")
    if has_pa:
        L.append(f"  Pearson(ESPmin, proton affinity) = {ok[['ESPmin_kcal_mol','proton_affinity_kcal_mol']].corr().iloc[0,1]:.2f}"
                 f"   Pearson(dipole, proton affinity) = {ok[['dipole_debye','proton_affinity_kcal_mol']].corr().iloc[0,1]:.2f}")

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        ax = axes[0]
        ax.scatter(ok.ESPmin_kcal_mol, ok.dipole_debye, s=20, alpha=0.5, c="tab:blue", label="oxane derivatives")
        ax.scatter(fr.ESPmin_kcal_mol, fr.dipole_debye, s=75, facecolors="none", edgecolors="k", linewidths=1.4, label="main front")
        if par is not None:
            ax.scatter([par.ESPmin_kcal_mol], [par.dipole_debye], s=90, marker="*", c="tab:red", label="parent oxane")
        ax.set_xlabel("ESP$_{min}$ (kcal/mol), less negative = weaker O site"); ax.set_ylabel("dipole moment (D)")
        ax.legend(frameon=False, fontsize=8)
        ax = axes[1]
        sc = df.groupby("n_F_atoms")
        for sname, gg in df.groupby("scaffold"):
            ax.scatter(gg.Ebind_global_kcal_mol, gg.ESPmin_kcal_mol, s=18, alpha=0.55, label=sname)
        ax.set_xlabel("global Na$^+$ binding energy (kcal/mol)"); ax.set_ylabel("ESP$_{min}$ (kcal/mol)")
        ax.set_title("gas-phase binding is not a clean O-site descriptor", fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_final.png"), dpi=200); plt.close(fig)
    except Exception as err:
        L.append(f"(figure skipped: {err})")

    text = "\n".join(L) + "\n"
    open(os.path.join(a.out, "summary_final.txt"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
