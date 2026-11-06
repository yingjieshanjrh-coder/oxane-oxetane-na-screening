#!/usr/bin/env python3
"""
compute_pa.py -- electronic proton affinity (PA) of the ring oxygen, an acid-initiation descriptor.

Why: the ring-strain release cannot rank molecules inside the oxane family (sd 1.4 kcal/mol).  The ring-opening
polymerization of cyclic ethers starts when a protic acid (formed from PF5/BF3-type Lewis acids and trace water)
protonates the ring oxygen.  A LOWER proton affinity means the oxygen is harder to protonate, so initiation is
harder.  Fluorination should lower it; methyl groups may raise it.  This is an electronic descriptor only: it does
not include steric effects on propagation, so it complements, and does not replace, the barrier calculations.

    PA = E(neutral) - E(protonated)    [kcal/mol, electronic energies, PBE0-D3(BJ)/def2-SVP, no ZPE/thermal terms]

Usage (same folder as validate_binding.py; needs run_lib/<id>_dftopt.xyz from run_library.py):
    python compute_pa.py --ids M0027            # test one molecule (parent oxane)
    python compute_pa.py --nproc 4              # all molecules, 4 single-core processes
    python compute_pa.py --ids M0027 M0116 --force --nproc 4   # recompute chosen molecules and save the protonated geometries
Output: pa.csv (id, proton_affinity_kcal_mol, ring_opened) and run_lib/pa_xyz/<id>_prot.xyz.
Finished molecules are skipped on re-runs unless --force is given.
FIX (2026-11): the first version flagged EVERY molecule as 'ring opened' because beta carbons (about 2.4 A from O)
were counted as bonded.  PA values were never affected; only the ring_opened column was wrong.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse
import csv
import glob
import json
import multiprocessing as mp
import time

import numpy as np

HARTREE_TO_KCAL = 627.5094740631


def read_xyz(path):
    L = open(path).read().strip().splitlines()
    n = int(L[0])
    sym, xyz = [], []
    for line in L[2:2 + n]:
        p = line.split()
        sym.append(p[0])
        xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return sym, np.array(xyz)


def protonate(sym, pos, r_oh=0.98):
    """Add H+ to the (single) ring oxygen along its lone-pair direction. Returns (sym2, pos2, o_index)."""
    o = [i for i, s in enumerate(sym) if s == "O"]
    if len(o) != 1:
        raise ValueError(f"expected exactly one oxygen, found {len(o)}")
    o = o[0]
    d = np.linalg.norm(pos - pos[o], axis=1)
    d[o] = 1e9
    nbrs = [i for i in np.argsort(d)[:2] if sym[i] == "C"]
    if len(nbrs) != 2:
        raise ValueError("oxygen is not bonded to two carbons")
    v = [(pos[i] - pos[o]) / np.linalg.norm(pos[i] - pos[o]) for i in nbrs]
    direction = -(v[0] + v[1])
    direction /= np.linalg.norm(direction)
    h = pos[o] + r_oh * direction
    return sym + ["H"], np.vstack([pos, h]), o


def ring_opened(sym, pos, o, bond_cutoff=1.9):
    """True if the oxygen has fewer than two bonded carbons (C-O shorter than 1.9 A).
    NOTE: beta carbons sit about 2.4 A from the oxygen and must NOT be counted as bonded."""
    d = np.linalg.norm(np.asarray(pos) - np.asarray(pos)[o], axis=1)
    bonded = [i for i, s_ in enumerate(sym) if s_ == "C" and d[i] < bond_cutoff]
    return len(bonded) < 2


def pa_of(path, basis="def2-svp", save_xyz=None):
    import validate_binding as vb
    vb.DFT_OPT = True
    sym, pos = read_xyz(path)
    e_neu = vb.dft_energy(sym, pos.tolist(), 0, basis)
    sym2, pos2, o = protonate(sym, pos)
    e_prot, sym3, pos3 = vb.dft_energy_opt(sym2, pos2.tolist(), +1, basis)
    pos3 = np.array(pos3)
    if save_xyz:
        with open(save_xyz, "w") as f:
            f.write(f"{len(sym3)}\nprotonated, DFT-optimised\n")
            for s_, c in zip(sym3, pos3):
                f.write(f"{s_} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
    return (e_neu - e_prot) * HARTREE_TO_KCAL, ring_opened(sym3, pos3, sym3.index("O"))


def job(args):
    mid, path, outdir, force = args
    out = os.path.join(outdir, mid + ".json")
    if os.path.exists(out) and not force:
        return mid, "skipped"
    t0 = time.time()
    try:
        xyz_dir = os.path.join(os.path.dirname(outdir), "pa_xyz")
        os.makedirs(xyz_dir, exist_ok=True)
        pa, opened = pa_of(path, save_xyz=os.path.join(xyz_dir, mid + "_prot.xyz"))
        json.dump({"id": mid, "proton_affinity_kcal_mol": round(pa, 2), "ring_opened": opened,
                   "seconds": round(time.time() - t0)}, open(out, "w"))
        return mid, f"PA = {pa:.1f} kcal/mol" + ("  (ring opened on protonation)" if opened else "")
    except Exception as err:
        return mid, "FAILED: " + type(err).__name__ + ": " + str(err)[:150]


def collect(outdir, csv_path):
    rows = []
    for f in sorted(glob.glob(os.path.join(outdir, "M*.json"))):
        d = json.load(open(f))
        rows.append({"id": d["id"], "proton_affinity_kcal_mol": d["proton_affinity_kcal_mol"], "ring_opened": d["ring_opened"]})
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "proton_affinity_kcal_mol", "ring_opened"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="run_lib")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--force", action="store_true", help="recompute even if a result exists")
    ap.add_argument("--out", default="pa.csv")
    a = ap.parse_args()
    outdir = os.path.join(a.workdir, "pa")
    os.makedirs(outdir, exist_ok=True)
    files = {os.path.basename(f).split("_")[0]: f for f in glob.glob(os.path.join(a.workdir, "M????_dftopt.xyz"))}
    ids = sorted(files)
    if a.ids:
        ids = [i for i in ids if i in set(a.ids)]
    print(f"{len(ids)} optimised ring geometries found")
    tasks = [(i, files[i], outdir, a.force) for i in ids]
    if a.nproc > 1:
        with mp.Pool(a.nproc) as pool:
            for mid, msg in pool.imap_unordered(job, tasks):
                print(mid, msg, flush=True)
    else:
        for t in tasks:
            print(*job(t), flush=True)
    print(f"wrote {a.out} with {collect(outdir, a.out)} molecules")


if __name__ == "__main__":
    main()
