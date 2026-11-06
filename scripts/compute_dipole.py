#!/usr/bin/env python3
"""
compute_dipole.py -- gas-phase dipole moment of every library molecule (salt-dissociation proxy).

Uses the DFT-optimised geometries that run_library.py already saved:  run_lib/<id>_dftopt.xyz
Level: PBE0-D3(BJ)/def2-SVP (same as the geometry optimisation), gas phase, density fitting.
The dipole is used ONLY as a rank-ordering proxy for salt-dissociation ability; it is not a dielectric constant.

Usage (run when the library calculation is finished, or earlier for the molecules already done):
    python compute_dipole.py --ids M0015            # test one molecule in the foreground
    python compute_dipole.py --nproc 4              # all available molecules, 4 single-core processes
Output: dipoles.csv  (id, dipole_debye); finished molecules are skipped on re-runs.
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


def dipole_of(path, basis="def2-svp"):
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
    d = mf.dip_moment(unit="Debye", verbose=0)
    return float(np.linalg.norm(d)), [float(x) for x in d]


def job(args):
    mid, path, outdir = args
    out = os.path.join(outdir, mid + ".json")
    if os.path.exists(out):
        return mid, "skipped"
    t0 = time.time()
    try:
        mag, vec = dipole_of(path)
        json.dump({"id": mid, "dipole_debye": round(mag, 4), "vector": [round(x, 4) for x in vec],
                   "seconds": round(time.time() - t0)}, open(out, "w"))
        return mid, f"{mag:.3f} D"
    except Exception as err:
        return mid, "FAILED: " + type(err).__name__ + ": " + str(err)[:150]


def collect(outdir, csv_path):
    rows = []
    for f in sorted(glob.glob(os.path.join(outdir, "M*.json"))):
        d = json.load(open(f))
        rows.append({"id": d["id"], "dipole_debye": d["dipole_debye"]})
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "dipole_debye"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="run_lib")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--out", default="dipoles.csv")
    a = ap.parse_args()
    outdir = os.path.join(a.workdir, "dipole")
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
