#!/usr/bin/env python3
"""
analyze_library.py -- Pareto analysis of the finished library (no chemistry software needed).

Inputs
  library_results.csv   from   python run_library.py --collect
  dipoles.csv           from   python compute_dipole.py        (salt-dissociation proxy, rank only)

Objectives (all converted internally to "minimise"):
  f1  desolvation      Na+ binding energy (global minimum), WEAKER is better   -> maximise Ebind (less negative)
  f2  salt dissociation  gas-phase dipole moment, LARGER is better (rank-only proxy)  -> maximise
  f3  ring-opening stability  ring-strain release energy, LOWER is better     -> minimise

Outputs (written to --out, default analysis/)
  pareto_table.csv        every molecule with its non-dominated rank (1 = Pareto front)
  pareto_front.csv        rank-1 molecules only
  sensitivity.csv         which molecules stay on the front when the definitions are changed
  summary.txt             plain-text summary used for the paper (numbers only, no interpretation)
  fig_pareto_pairs.png    three 2-D projections, front highlighted

Usage
    python analyze_library.py --results library_results.csv --dipoles dipoles.csv --out analysis
    python analyze_library.py --selftest                   # runs on synthetic data to check the code
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- Pareto machinery
def non_dominated_ranks(F):
    """F: (n, m) array, all objectives to MINIMISE. Returns integer ranks (1 = non-dominated)."""
    F = np.asarray(F, float)
    n = len(F)
    ranks = np.zeros(n, dtype=int)
    remaining = np.arange(n)
    r = 1
    while len(remaining):
        sub = F[remaining]
        keep = []
        for i in range(len(remaining)):
            dominated = False
            for j in range(len(remaining)):
                if i != j and np.all(sub[j] <= sub[i]) and np.any(sub[j] < sub[i]):
                    dominated = True
                    break
            if not dominated:
                keep.append(i)
        ranks[remaining[keep]] = r
        remaining = np.delete(remaining, keep)
        r += 1
    return ranks


def objective_matrix(df, f1, f2, f3):
    """Columns -> minimisation form: [-Ebind, -dipole, RSE]."""
    return np.column_stack([-df[f1].to_numpy(float), -df[f2].to_numpy(float), df[f3].to_numpy(float)])


# ----------------------------------------------------------------- analysis
def load(results, dipoles):
    df = pd.read_csv(results)
    if dipoles and os.path.exists(dipoles):
        dp = pd.read_csv(dipoles)[["id", "dipole_debye"]]
        df = df.merge(dp, on="id", how="left")
    else:
        df["dipole_debye"] = np.nan
    return df


def quality_report(df):
    lines = [f"molecules in table: {len(df)}"]
    for c in ["Ebind_global_kcal_mol", "Ebind_noF_kcal_mol", "RSE_release_kcal_mol", "ESPmin_kcal_mol",
              "Volume_A3", "dipole_debye"]:
        if c in df:
            nmiss = int(df[c].isna().sum())
            lines.append(f"  {c:26s} missing: {nmiss:3d}   range: {df[c].min():8.2f} .. {df[c].max():8.2f}")
    if "errors" in df:
        ne = int(df["errors"].fillna("").astype(str).str.len().gt(0).sum())
        lines.append(f"  molecules with recorded errors: {ne}")
    # robust outlier flag (|z| > 4 with median/MAD) on the three objectives, computed WITHIN each scaffold
    # (the oxetane and oxane ring-strain values are bimodal by design and must not be mixed)
    for c in ["Ebind_global_kcal_mol", "RSE_release_kcal_mol", "dipole_debye"]:
        if c not in df:
            continue
        bad = []
        for sc, g in df.groupby("scaffold"):
            x = g[c].dropna()
            if len(x) < 6:
                continue
            mad = np.median(np.abs(x - x.median())) or 1e-9
            z = 0.6745 * (g[c] - x.median()) / mad
            bad += g.loc[z.abs() > 4, "id"].tolist()
        if bad:
            lines.append(f"  possible outliers in {c} (within scaffold): {sorted(bad)}")
    return "\n".join(lines)


def run_analysis(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    need = ["Ebind_global_kcal_mol", "dipole_debye", "RSE_release_kcal_mol"]
    use = df.dropna(subset=need).copy()
    dropped = len(df) - len(use)
    F = objective_matrix(use, *need)
    use["pareto_rank"] = non_dominated_ranks(F)
    use.sort_values(["pareto_rank", "RSE_release_kcal_mol"]).to_csv(os.path.join(outdir, "pareto_table.csv"), index=False)
    front = use[use["pareto_rank"] == 1]
    front.sort_values("RSE_release_kcal_mol").to_csv(os.path.join(outdir, "pareto_front.csv"), index=False)

    # ---- sensitivity: change the definitions and see who survives
    variants = {"baseline": (need[0], need[1], need[2])}
    if "Ebind_noF_kcal_mol" in use:
        use["Ebind_mixed"] = use["Ebind_noF_kcal_mol"].fillna(use["Ebind_global_kcal_mol"])
        variants["binding: F-free where available"] = ("Ebind_mixed", need[1], need[2])
    if "ESPmin_kcal_mol" in use:
        # a LESS negative ESPmin means a weaker Na+ site; use it in place of the binding energy
        variants["binding replaced by ESPmin"] = ("ESPmin_kcal_mol", need[1], need[2])
    if "Volume_A3" in use:
        variants["dissociation proxy: dipole per volume"] = (need[0], "dipole_per_vol", need[2])
        use["dipole_per_vol"] = use["dipole_debye"] / use["Volume_A3"]
    rows = []
    fronts = {}
    for name, (a, b, c) in variants.items():
        r = non_dominated_ranks(objective_matrix(use, a, b, c))
        fronts[name] = set(use.loc[r == 1, "id"])
    base = fronts["baseline"]
    for name, s in fronts.items():
        jac = len(base & s) / max(1, len(base | s))
        rows.append({"variant": name, "front_size": len(s), "overlap_with_baseline": len(base & s),
                     "jaccard": round(jac, 2)})
    robust = set.intersection(*fronts.values())
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "sensitivity.csv"), index=False)

    # ---- summary
    lines = ["PARETO SUMMARY (numbers only)", "", quality_report(df), "",
             f"molecules used (all three objectives present): {len(use)}   dropped: {dropped}",
             f"Pareto-front size (baseline): {len(front)}"]
    for sc, g in use.groupby("scaffold"):
        nfront = int((g["pareto_rank"] == 1).sum())
        lines.append(f"  {sc:8s}: {len(g):3d} molecules, {nfront:3d} on the front, "
                     f"RSE_release range {g['RSE_release_kcal_mol'].min():.1f}..{g['RSE_release_kcal_mol'].max():.1f} kcal/mol")
    lines += ["", "front members (baseline), sorted by ring-strain release:"]
    for _, r in front.sort_values("RSE_release_kcal_mol").iterrows():
        lines.append(f"  {r['id']}  {r['scaffold']:8s} {r['ring_smiles']:24s} Ebind {r['Ebind_global_kcal_mol']:7.2f}  "
                     f"dipole {r['dipole_debye']:5.2f}  RSE {r['RSE_release_kcal_mol']:5.2f}")
    lines += ["", "sensitivity (Jaccard overlap of the front with the baseline front):"]
    for r in rows:
        lines.append(f"  {r['variant']:42s} size {r['front_size']:3d}  overlap {r['overlap_with_baseline']:3d}  jaccard {r['jaccard']}")
    lines += ["", f"robust members (on the front in ALL variants): {sorted(robust)}"]
    open(os.path.join(outdir, "summary.txt"), "w").write("\n".join(lines) + "\n")

    # ---- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pairs = [("Ebind_global_kcal_mol", "dipole_debye"), ("Ebind_global_kcal_mol", "RSE_release_kcal_mol"),
                 ("dipole_debye", "RSE_release_kcal_mol")]
        labels = {"Ebind_global_kcal_mol": "Na$^+$ binding energy (kcal/mol)", "dipole_debye": "dipole moment (D)",
                  "RSE_release_kcal_mol": "ring-strain release (kcal/mol)"}
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
        colors = {"oxetane": "tab:red", "oxane": "tab:blue"}
        for ax, (a, b) in zip(axes, pairs):
            for sc, g in use.groupby("scaffold"):
                ax.scatter(g[a], g[b], s=22, alpha=0.55, c=colors.get(sc, "gray"), label=sc)
            ax.scatter(front[a], front[b], s=70, facecolors="none", edgecolors="k", linewidths=1.4, label="Pareto front")
            ax.set_xlabel(labels[a]); ax.set_ylabel(labels[b])
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig_pareto_pairs.png"), dpi=200)
        plt.close(fig)
    except Exception as err:      # plotting must never block the tables
        print("plotting skipped:", err)
    return use, front, robust


# ----------------------------------------------------------------- self-test on synthetic data
def selftest():
    rng = np.random.default_rng(1)
    n = 134
    sc = ["oxetane"] * 23 + ["oxane"] * 111
    df = pd.DataFrame({
        "id": [f"M{i:04d}" for i in range(1, n + 1)], "scaffold": sc,
        "ring_smiles": ["C1CCO1"] * n, "n_F_atoms": rng.integers(0, 4, n),
        "Ebind_global_kcal_mol": rng.normal(-33, 3, n),
        "Ebind_noF_kcal_mol": np.where(rng.random(n) < 0.6, rng.normal(-31, 3, n), np.nan),
        "RSE_release_kcal_mol": np.where(np.array(sc) == "oxetane", rng.normal(26, 1, n), rng.normal(3, 1, n)),
        "ESPmin_kcal_mol": rng.normal(-33, 4, n), "Volume_A3": rng.normal(110, 20, n),
        "dipole_debye": rng.normal(1.8, 0.5, n), "errors": [""] * n,
    })
    # brute-force check of the ranking function
    F = objective_matrix(df, "Ebind_global_kcal_mol", "dipole_debye", "RSE_release_kcal_mol")
    ranks = non_dominated_ranks(F)
    brute = np.array([not any(np.all(F[j] <= F[i]) and np.any(F[j] < F[i]) for j in range(n) if j != i) for i in range(n)])
    assert np.array_equal(ranks == 1, brute), "rank-1 set differs from brute force"
    # every rank-2 point must be dominated by some rank-1 point and not by another rank-2 point
    for i in np.where(ranks == 2)[0]:
        assert any(np.all(F[j] <= F[i]) and np.any(F[j] < F[i]) for j in np.where(ranks == 1)[0])
    use, front, robust = run_analysis(df, "/tmp/analysis_selftest")
    print("selftest OK: front size", len(front), "| robust members", len(robust),
          "| oxetanes on the front:", int((front["scaffold"] == "oxetane").sum()))
    print(open("/tmp/analysis_selftest/summary.txt").read()[:900])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="library_results.csv")
    ap.add_argument("--dipoles", default="dipoles.csv")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    df = load(a.results, a.dipoles)
    if df["dipole_debye"].isna().all():
        raise SystemExit("no dipole moments found: run  python compute_dipole.py  first")
    run_analysis(df, a.out)
    print(open(os.path.join(a.out, "summary.txt")).read())


if __name__ == "__main__":
    main()
