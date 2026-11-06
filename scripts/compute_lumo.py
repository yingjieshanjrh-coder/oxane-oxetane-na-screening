#!/usr/bin/env python3
"""
compute_lumo.py -- HOMO and LUMO energies of every library molecule (reductive / oxidative stability descriptors).

Why: the screen so far ignores stability toward the sodium metal anode.  Strongly fluorinated ethers can be
reduced more easily (low LUMO), forming NaF-rich or organic-rich interphases (Gao et al. and Niu et al. both discuss the LUMO).
The LUMO energy of an isolated solvent is only a rough indicator, so it is used here to FLAG molecules, not to rank them.

Uses the DFT-optimised geometries that run_library.py already saved:  run_lib/<id>_dftopt.xyz
Level: PBE0-D3(BJ)/def2-SVP (same as the geometry optimisation), gas phase, density fitting.

Usage (run when the library calculation is finished, or earlier for the molecules already done):
    python compute_lumo.py --ids M0015            # test one molecule in the foreground
    python compute_lumo.py --nproc 4              # all available molecules, 4 single-core processes
Output: lumo.csv  (id, homo_eV, lumo_eV); finished molecules are skipped on re-runs.
"""
import os

# one thread per process: must be set before numpy / pyscf are imported
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse
import csv
import glob
import json
import multiprocessing as mp
import time


def read_xyz(path):
    L = open(path).read().strip().splitlines()
    n = int(L[0])
    sym, xyz = [], []
    for line in L[2:2 + n]:
        p = line.split()
        sym.append(p[0])
        xyz.append((float(p[1]), float(p[2]), float(p[3])))
    return sym, xyz


def orbitals_of(path, basis="def2-svp"):
    from pyscf import gto, dft
    import numpy as np
    sym, xyz = read_xyz(path)
    mol = gto.M(atom=list(zip(sym, xyz)), charge=0, spin=0, basis=basis, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "pbe0"
    try:
        mf.disp = "d3bj"
    except Exception:
        pass
    mf = mf.density_fit()
    mf.kernel()
    HARTREE_TO_EV = 27.211386245988
    occ = mf.mo_occ
    homo = float(mf.mo_energy[occ > 0].max()) * HARTREE_TO_EV
    lumo = float(mf.mo_energy[occ == 0].min()) * HARTREE_TO_EV
    return homo, lumo


def job(args):
    mid, path, outdir = args
    out = os.path.join(outdir, mid + ".json")
    if os.path.exists(out):
        return mid, "skipped"
    t0 = time.time()
    try:
        homo, lumo = orbitals_of(path)
        json.dump({"id": mid, "homo_eV": round(homo, 3), "lumo_eV": round(lumo, 3),
                   "seconds": round(time.time() - t0)}, open(out, "w"))
        return mid, f"HOMO {homo:.2f} eV  LUMO {lumo:.2f} eV"
    except Exception as err:
        return mid, "FAILED: " + type(err).__name__ + ": " + str(err)[:150]


def collect(outdir, csv_path):
    rows = []
    for f in sorted(glob.glob(os.path.join(outdir, "M*.json"))):
        d = json.load(open(f))
        rows.append({"id": d["id"], "homo_eV": d["homo_eV"], "lumo_eV": d["lumo_eV"]})
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "homo_eV", "lumo_eV"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="run_lib")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--out", default="lumo.csv")
    a = ap.parse_args()
    outdir = os.path.join(a.workdir, "lumo")
    os.makedirs(outdir, exist_ok=True)
    files = {}
    for f in glob.glob(os.path.join(a.workdir, "M????_dftopt.xyz")):
        files[os.path.basename(f).split("_")[0]] = f
    ids = sorted(files)
    if a.ids:
        ids = [i for i in ids if i in set(a.ids)]
    print(f"{len(ids)} optimised ring geometries found")
    tasks = [(i, files[i], outdir) for i in ids]
    if a.nproc > 1:
        with mp.Pool(a.nproc) as pool:
            for mid, msg in pool.imap_unordered(job, tasks):
                print(mid, msg, flush=True)
    else:
        for t in tasks:
            print(*job(t), flush=True)
    n = collect(outdir, a.out)
    print(f"wrote {a.out} with {n} molecules")


if __name__ == "__main__":
    main()
