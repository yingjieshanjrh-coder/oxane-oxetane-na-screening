#!/usr/bin/env python3
"""
extra_analysis.py -- reproduces the secondary numbers quoted in the manuscript.

Inputs (all produced by the earlier scripts): library_results.csv, dipoles.csv, pa.csv, lumo.csv
Needs run_library.py and enumerate_library.py in the same folder (only for the substituent positions).

Usage
    python extra_analysis.py --results library_results.csv --dipoles dipoles.csv --pa pa.csv --lumo lumo.csv
Prints a report; also writes extra_analysis_report.txt.
"""
import argparse

import numpy as np
import pandas as pd

import run_library as rl
from analyze_library import non_dominated_ranks

# Niu et al. (Angew. Chem. 2026) SI Table S1 and this work's values for the same seven dioxolanes (validation run)
NIU = {"DOL": -32.20, "2MeDOL": -32.75, "4MeDOL": -33.03, "24diMeDOL": -33.57, "TFDOL": -27.40, "22diMeDOL": -32.81, "OMeDOL": -39.45}
OURS = {"DOL": -30.03, "2MeDOL": -33.02, "4MeDOL": -32.05, "24diMeDOL": -33.15, "TFDOL": -25.20, "22diMeDOL": -31.73, "OMeDOL": -35.03}


def feats(cfg):
    """Number of F atoms at 3, 4, 5 bonds from the ring oxygen, and number of methyl groups."""
    nF = {"F": 1, "CH3": 0, "CF3": 3, "CH2F": 1, "CHF2": 2}
    n = len(cfg)
    f = {3: 0, 4: 0, 5: 0}
    me = 0
    for i, subs in enumerate(cfg):
        r = min(i, n - 1 - i) + 1                 # ring-bond distance O -> this carbon (alpha = 1)
        for s in subs:
            if s == "F":
                f[r + 1] = f.get(r + 1, 0) + 1
            elif s == "CH3":
                me += 1
            else:
                f[r + 2] = f.get(r + 2, 0) + nF[s]
    return {"F_d3": f[3], "F_d4": f[4], "F_d5": f[5], "n_CH3": me}


def fit(d, y, cols):
    A = np.column_stack([np.ones(len(d))] + [d[c] for c in cols])
    coef = np.linalg.lstsq(A, d[y], rcond=None)[0]
    pred = A @ coef
    r2 = 1 - ((d[y] - pred) ** 2).sum() / ((d[y] - d[y].mean()) ** 2).sum()
    return coef, r2, float(np.sqrt(((d[y] - pred) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="library_results.csv")
    ap.add_argument("--dipoles", default="dipoles.csv")
    ap.add_argument("--pa", default="pa.csv")
    ap.add_argument("--lumo", default="lumo.csv")
    a = ap.parse_args()
    L = []
    P = L.append

    df = (pd.read_csv(a.results).merge(pd.read_csv(a.dipoles)[["id", "dipole_debye"]], on="id")
          .merge(pd.read_csv(a.pa)[["id", "proton_affinity_kcal_mol"]], on="id")
          .merge(pd.read_csv(a.lumo), on="id"))
    P(f"molecules: {len(df)} ({(df.scaffold=='oxetane').sum()} oxetanes, {(df.scaffold=='oxane').sum()} oxanes)")
    ox = df[df.scaffold == "oxane"].copy()
    par = ox[ox.ring_smiles == "C1CCCCO1"].iloc[0]
    P(f"parent oxane {par.id}: ESPmin {par.ESPmin_kcal_mol:.2f}  dipole {par.dipole_debye:.2f}  PA {par.proton_affinity_kcal_mol:.1f}  LUMO {par.lumo_eV:.2f}")

    P("\n== ring strain ==")
    for sc, g in df.groupby("scaffold"):
        P(f"{sc}: RSE {g.RSE_release_kcal_mol.min():.1f}..{g.RSE_release_kcal_mol.max():.1f}  sd {g.RSE_release_kcal_mol.std():.2f}")

    P("\n== chelation (molecules with an F-free reference) ==")
    s = df.dropna(subset=["Ebind_noF_kcal_mol"])
    P(f"n with F-free reference: {len(s)} of {len(df)}")
    for nm, (x, y) in {"F-free vs ESPmin": ("Ebind_noF_kcal_mol", "ESPmin_kcal_mol"), "global vs ESPmin": ("Ebind_global_kcal_mol", "ESPmin_kcal_mol"),
                       "global vs F-free": ("Ebind_global_kcal_mol", "Ebind_noF_kcal_mol")}.items():
        P(f"  Pearson {nm}: {s[[x, y]].corr().iloc[0, 1]:.2f}")
    f = s[s.n_F_atoms > 0]
    gain = (f.Ebind_global_kcal_mol - f.Ebind_noF_kcal_mol)
    P(f"  fluorinated with both values: n={len(f)}  chelation gain mean {-gain.mean():.1f}  max {-gain.min():.1f} kcal/mol")

    P("\n== ESPmin by number of F (oxanes) ==")
    P(ox.groupby("n_F_atoms").ESPmin_kcal_mol.agg(["mean", "count"]).round(2).to_string())
    P("mean global Ebind by number of F (oxanes), to show it is not monotonic:")
    P(ox.groupby("n_F_atoms").Ebind_global_kcal_mol.mean().round(2).to_string())

    P("\n== additive model (oxanes) ==")
    lib = {x["id"]: x for x in rl.build_library()}
    X = pd.DataFrame([feats(lib[i]["config"]) for i in ox.id]); X["id"] = ox.id.values
    d = ox.merge(X, on="id")
    assert ((d.F_d3 + d.F_d4 + d.F_d5) == d.n_F_atoms).all(), "feature check failed"
    cols = ["F_d3", "F_d4", "F_d5", "n_CH3"]
    for y in ["ESPmin_kcal_mol", "proton_affinity_kcal_mol", "dipole_debye"]:
        c, r2, rms = fit(d, y, cols)
        P(f"{y}: R2 {r2:.2f}  rms {rms:.2f} | per F at 3/4/5 bonds {c[1]:+.2f} {c[2]:+.2f} {c[3]:+.2f} | per CH3 {c[4]:+.2f}")
    r2dip = fit(d.assign(nF=d.n_F_atoms), "dipole_debye", ["nF"])[1]
    P(f"dipole vs number of F: R2 {r2dip:.2f}")

    P("\n== correlations (oxanes) ==")
    P(f"ESPmin-dipole {ox[['ESPmin_kcal_mol','dipole_debye']].corr().iloc[0,1]:.2f}   ESPmin-PA {ox[['ESPmin_kcal_mol','proton_affinity_kcal_mol']].corr().iloc[0,1]:.2f}"
      f"   dipole-PA {ox[['dipole_debye','proton_affinity_kcal_mol']].corr().iloc[0,1]:.2f}   HOMO-PA {ox[['homo_eV','proton_affinity_kcal_mol']].corr().iloc[0,1]:.2f}")

    P("\n== ESPmin window anchored on Niu et al. ==")
    off = np.mean([OURS[k] - NIU[k] for k in NIU])
    lo, hi = NIU["24diMeDOL"] + off, NIU["TFDOL"] + off
    P(f"mean offset (ours - Niu) over 7 dioxolanes: {off:+.2f}; window {lo:.1f} .. {hi:.1f} kcal/mol")
    w = ox[(ox.ESPmin_kcal_mol >= lo) & (ox.ESPmin_kcal_mol <= hi)].copy()
    P(f"oxanes inside the window: {len(w)} of {len(ox)}; parent {par.ESPmin_kcal_mol:.2f} is {'inside' if lo <= par.ESPmin_kcal_mol <= hi else 'outside (too strong)'}")
    F = np.column_stack([w.proton_affinity_kcal_mol, -w.dipole_debye])
    w["rk"] = non_dominated_ranks(F)
    wf = w[w.rk == 1].sort_values("proton_affinity_kcal_mol")
    P("window Pareto (lower PA, larger dipole):")
    P(wf[["id", "ring_smiles", "n_F_atoms", "ESPmin_kcal_mol", "dipole_debye", "proton_affinity_kcal_mol", "lumo_eV"]].round(2).to_string(index=False))

    P("\n== main (constrained) front: ESPmin, dipole, PA ==")
    Fm = np.column_stack([-ox.ESPmin_kcal_mol, -ox.dipole_debye, ox.proton_affinity_kcal_mol])
    ox["rank"] = non_dominated_ranks(Fm)
    fr = ox[ox["rank"] == 1].sort_values("ESPmin_kcal_mol", ascending=False)
    P(fr[["id", "n_F_atoms", "ESPmin_kcal_mol", "dipole_debye", "proton_affinity_kcal_mol", "lumo_eV"]].round(2).to_string(index=False))
    P(f"front molecules weaker than the TFDOL level ({hi:.1f}): {sorted(fr[fr.ESPmin_kcal_mol > hi].id)}")
    good = ox[(ox.ESPmin_kcal_mol > par.ESPmin_kcal_mol + 4) & (ox.dipole_debye > par.dipole_debye + 0.3)
              & (ox.proton_affinity_kcal_mol < par.proton_affinity_kcal_mol)]
    P(f"better than the parent by margins (ESPmin +4, dipole +0.3, lower PA): {len(good)} of {len(ox)}")

    P("\n== frontier orbitals (oxanes) ==")
    P(ox.groupby("n_F_atoms")[["lumo_eV", "homo_eV"]].mean().round(2).to_string())
    P("PA of front members: %.1f .. %.1f ; window members: %.1f .. %.1f" % (fr.proton_affinity_kcal_mol.min(), fr.proton_affinity_kcal_mol.max(),
                                                                        wf.proton_affinity_kcal_mol.min(), wf.proton_affinity_kcal_mol.max()))
    text = "\n".join(L) + "\n"
    open("extra_analysis_report.txt", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
