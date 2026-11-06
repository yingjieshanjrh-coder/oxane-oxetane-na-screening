#!/usr/bin/env python3
"""
validate_binding.py  --  pipeline check against published reference molecules.

What it does
------------
1. Conformer search (RDKit ETKDG + MMFF, then xtb GFN2 optimisation).
2. Na+ complexes: Na+ is placed ~2.3 A from every oxygen along its lone-pair
   direction, optimised with xtb (charge +1), lowest-energy complex kept.
3. Single-point DFT energies with PySCF (PBE0, def2-SVP by default) for
   complex, solvent and Na+.  (xtb cannot handle a bare Na+ with zero valence
   electrons, so Na+ is always treated with PySCF.)
4. Binding energy  E_bind = E(complex) - E(solvent) - E(Na+)   [kcal/mol]
   Reported twice: Ebind_global (any complex) and Ebind_noF (only complexes
   with no Na-F contact < 3.0 A, i.e. O-site binding without F chelation).
   All 4 lowest-xtb complexes are re-ranked with DFT; see complex_log.csv.
5. Ring strain energy (RSE) for unsubstituted single-oxygen rings from the
   homodesmotic reaction
        ring + (n-1) C2H6  ->  diethyl ether + (n-2) C3H8
   RSE = E(reactants) - E(products)   (positive = strained).
   NOTE: this is MY definition for the three parent rings only.  Niu et al.
   used a different homodesmotic scheme (their Figure S6, ethyl methyl ether as
   the strain-releasing species).  Substituted rings need a scheme matched to
   theirs, so please send me Figure S6 before we extend RSE to the library.

Published numbers to compare with (rank order matters, not absolute values)
--------------------------------------------------------------------------
Gao et al., Adv. Funct. Mater. 2025 (PBE/DNP, DMol3), Na+ binding energy:
    EA -37.63, EDFA -38.38, DFEA -30.46 kcal/mol
Niu et al., Angew. Chem. 2026, RSE (their definition):
    DOL 4.16 > 4MeDOL 3.02 > 2MeDOL 1.94 > 24diMeDOL (lowest)
Their level of theory differs from this script, so differences of a few
kcal/mol are expected; the ORDER is the thing to check.

Usage
-----
    python validate_binding.py --quick          # 3 molecules, first trial
    python validate_binding.py --quick --dft-opt   # same, with DFT geometry optimisation
                                                   # (needs: pip install geometric)
    python validate_binding.py                  # full reference set
"""
import argparse
import csv
import math
import os
import re
import subprocess
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

HARTREE_TO_KCAL = 627.5094740631

REFERENCE_SET = {
    "EA": "CCOC(C)=O",
    "EDFA": "CCOC(=O)C(F)F",
    "DFEA": "CC(=O)OCC(F)F",
    "DOL": "C1COCO1",
    "2MeDOL": "CC1OCCO1",
    "4MeDOL": "CC1COCO1",
    "24diMeDOL": "CC1COC(C)O1",
    "oxetane": "C1COC1",
    "oxolane": "C1CCOC1",
    "oxane": "C1CCOCC1",
}
QUICK = ["EA", "DFEA", "oxolane"]
RSE_RINGS = {"oxetane": 3, "oxolane": 4, "oxane": 5}   # name -> ring carbons n


# ---------------------------------------------------------------- helpers
def formula_counts(smiles):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    out = {}
    for a in m.GetAtoms():
        out[a.GetSymbol()] = out.get(a.GetSymbol(), 0) + 1
    return out


def check_rse_balance(n):
    """Atom balance of  ring + (n-1) ethane -> DEE + (n-2) propane."""
    ring = {"C": n, "H": 2 * n, "O": 1}
    left = {"C": ring["C"] + 2 * (n - 1), "H": ring["H"] + 6 * (n - 1), "O": 1}
    right = {"C": 4 + 3 * (n - 2), "H": 10 + 8 * (n - 2), "O": 1}
    assert left == right, (n, left, right)


def write_xyz(path, symbols, coords, comment=""):
    with open(path, "w") as f:
        f.write(f"{len(symbols)}\n{comment}\n")
        for s, (x, y, z) in zip(symbols, coords):
            f.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")


def read_xyz(path):
    with open(path) as f:
        lines = f.read().strip().splitlines()
    n = int(lines[0])
    sym, xyz = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        sym.append(p[0])
        xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return sym, np.array(xyz)


def run_xtb(xyz_path, charge, workdir):
    """Optimise with GFN2-xTB. Returns (E_hartree, optimised symbols, coords)."""
    os.makedirs(workdir, exist_ok=True)
    cmd = ["xtb", os.path.abspath(xyz_path), "--gfn", "2",
           "--chrg", str(charge), "--opt", "tight"]
    res = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", res.stdout)
    opt = os.path.join(workdir, "xtbopt.xyz")
    if not m or not os.path.exists(opt):
        raise RuntimeError("xtb failed in %s\n%s" % (workdir, res.stdout[-800:] + res.stderr[-400:]))
    sym, xyz = read_xyz(opt)
    return float(m.group(1)), sym, xyz


# ---------------------------------------------------------------- conformers
def solvent_conformers(smiles, n_conf=30, keep=4, seed=7):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    cids = AllChem.EmbedMultipleConfs(mol, n_conf, ps)
    if len(cids) == 0:
        raise RuntimeError("embedding failed for " + smiles)
    AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    confs = [(cid, mol.GetConformer(cid).GetPositions()) for cid in cids]
    # crude diversity filter: drop conformers whose heavy-atom RMS to a kept one < 0.3 A
    heavy = [i for i, s in enumerate(syms) if s != "H"]
    kept = []
    for cid, pos in confs:
        if all(np.sqrt(np.mean(np.sum((pos[heavy] - k[heavy]) ** 2, axis=1))) > 0.3 for _, k in kept):
            kept.append((cid, pos))
        if len(kept) >= keep * 3:
            break
    return syms, [p for _, p in kept], mol


def best_solvent(name, smiles, workdir):
    syms, poses, mol = solvent_conformers(smiles)
    best = None
    for i, pos in enumerate(poses):
        xyz = os.path.join(workdir, f"{name}_s{i}.xyz")
        write_xyz(xyz, syms, pos, f"{name} conf {i}")
        e, s2, p2 = run_xtb(xyz, 0, os.path.join(workdir, f"{name}_s{i}"))
        if best is None or e < best[0]:
            best = (e, s2, p2)
    return best, mol


# ---------------------------------------------------------------- Na+ complexes
def lone_pair_direction(mol, o_idx, pos):
    nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(o_idx).GetNeighbors()]
    vecs = []
    for n in nbrs:
        v = pos[n] - pos[o_idx]
        vecs.append(v / np.linalg.norm(v))
    d = -np.sum(vecs, axis=0)
    return d / np.linalg.norm(d)


def all_complexes(name, smiles, solvent_pos_list, workdir, mol):
    """Place Na+ near every O of every provided solvent geometry; return ALL converged xtb complexes."""
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    o_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O"]
    found = []
    k = 0
    for pos in solvent_pos_list:
        for o in o_atoms:
            d = lone_pair_direction(mol, o, pos)
            na = pos[o] + 2.3 * d
            xyz = os.path.join(workdir, f"{name}_c{k}.xyz")
            write_xyz(xyz, syms + ["Na"], np.vstack([pos, na]), f"{name} + Na+ start {k}")
            try:
                e, s2, p2 = run_xtb(xyz, +1, os.path.join(workdir, f"{name}_c{k}"))
                found.append((e, s2, p2))
            except RuntimeError as err:
                print("  (skipped one complex start:", str(err).splitlines()[0], ")")
            k += 1
    if not found:
        raise RuntimeError("no complex converged for " + name)
    return found


def min_dist(symbols, coords, element):
    """Shortest distance from the Na atom (last atom) to any atom of the given element."""
    na = coords[-1]
    ds = [np.linalg.norm(coords[i] - na) for i, s in enumerate(symbols[:-1]) if s == element]
    return min(ds) if ds else float("nan")


# ---------------------------------------------------------------- DFT single points
def dft_energy(symbols, coords, charge, basis="def2-svp"):
    from pyscf import gto, dft
    atom = [(s, tuple(c)) for s, c in zip(symbols, coords)]
    mol = gto.M(atom=atom, charge=charge, spin=0, basis=basis, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "pbe0"
    try:
        mf.disp = "d3bj"
    except Exception:
        pass
    mf = mf.density_fit()
    return float(mf.kernel())


DFT_OPT = False   # set by --dft-opt


def dft_energy_opt(symbols, coords, charge, basis="def2-svp"):
    """DFT geometry optimisation (PySCF + geomeTRIC), then a final single point.
    Returns (E_hartree, symbols, coords).  Without --dft-opt this is just a single point.
    Needs:  pip install geometric
    """
    if (not DFT_OPT) or len(symbols) == 1:
        return dft_energy(symbols, coords, charge, basis), symbols, coords
    from pyscf import gto, dft
    from pyscf.geomopt.geometric_solver import optimize
    atom = [(sy, tuple(c)) for sy, c in zip(symbols, coords)]
    mol = gto.M(atom=atom, charge=charge, spin=0, basis=basis, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "pbe0"
    try:
        mf.disp = "d3bj"
    except Exception:
        pass
    mf = mf.density_fit()
    mol_eq = optimize(mf, maxsteps=150)
    new_coords = np.array(mol_eq.atom_coords(unit="Angstrom"))
    e = dft_energy(symbols, new_coords, charge, basis)
    return e, symbols, new_coords


def dft_energy_opt_constrained(symbols, coords, charge, basis, constraint_lines, workdir, tag):
    """DFT optimisation with geomeTRIC distance constraints ('distance i j value', 1-based).
    The result is a CONSTRAINED minimum (used for the F-free reference complex)."""
    from pyscf import gto, dft
    from pyscf.geomopt.geometric_solver import optimize
    os.makedirs(workdir, exist_ok=True)
    cpath = os.path.abspath(os.path.join(workdir, f"{tag}_constraints.txt"))
    with open(cpath, "w") as f:
        f.write("$set\n" + "\n".join(constraint_lines) + "\n")
    atom = [(sy, tuple(c)) for sy, c in zip(symbols, coords)]
    mol = gto.M(atom=atom, charge=charge, spin=0, basis=basis, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "pbe0"
    try:
        mf.disp = "d3bj"
    except Exception:
        pass
    mf = mf.density_fit()
    mol_eq = optimize(mf, maxsteps=200, constraints=cpath)
    new_coords = np.array(mol_eq.atom_coords(unit="Angstrom"))
    e = dft_energy(symbols, new_coords, charge, basis)
    return e, symbols, new_coords


_dispersion_warned = False


def check_dispersion():
    global _dispersion_warned
    try:
        import pyscf.dispersion  # noqa: F401
    except Exception:
        if not _dispersion_warned:
            print("WARNING: pyscf-dispersion not found; D3 dispersion may be missing.\n"
                  "         Binding energies will be systematically less negative.")
            _dispersion_warned = True


# ---------------------------------------------------------------- main pieces
F_CONTACT = 3.0   # A; Na-F shorter than this counts as an F contact (chelation)


def binding_energy(name, smiles, workdir, basis, log, n_keep=4):
    (e_x, s_sym, s_pos), mol = best_solvent(name, smiles, workdir)
    syms, poses, _ = solvent_conformers(smiles, keep=3)
    starts = [s_pos] + poses[:2]
    found = all_complexes(name, smiles, starts, workdir, mol)
    found.sort(key=lambda t: t[0])
    found = found[:(2 if DFT_OPT else n_keep)]
    e_s, _, _ = dft_energy_opt(s_sym, s_pos, 0, basis)
    e_na = dft_energy(["Na"], [[0.0, 0.0, 0.0]], +1, basis)
    best_all, best_nof = float("inf"), float("inf")
    best_geom = None
    for k, (e_xtb, c_sym, c_pos) in enumerate(found):
        e_c, c_sym, c_pos = dft_energy_opt(c_sym, c_pos, +1, basis)
        eb = (e_c - e_s - e_na) * HARTREE_TO_KCAL
        dO = min_dist(c_sym, c_pos, "O")
        dF = min_dist(c_sym, c_pos, "F")
        log.append({"name": name, "complex": k, "E_xtb_Eh": round(e_xtb, 6),
                    "Ebind_kcal_mol": round(eb, 2),
                    "NaO_min_A": round(dO, 3), "NaF_min_A": round(dF, 3)})
        if eb < best_all:
            best_all, best_geom = eb, (c_sym, c_pos)
        if math.isnan(dF) or dF > F_CONTACT:
            best_nof = min(best_nof, eb)

    # F-free reference: if every complex has an Na...F contact, push all Na-F distances
    # to 3.6 A (constrained optimisation) to see what O-site-only binding looks like.
    if best_nof == float("inf"):
        best_nof = float("nan")
        if DFT_OPT and best_geom is not None and "F" in best_geom[0]:
            c_sym, c_pos = best_geom
            na_idx = len(c_sym)                       # 1-based index of Na (last atom)
            lines = [f"distance {na_idx} {i + 1} 3.6" for i, sy in enumerate(c_sym[:-1]) if sy == "F"]
            try:
                e_c, c2, p2 = dft_energy_opt_constrained(
                    c_sym, c_pos, +1, basis, lines, workdir, name + "_noF")
                best_nof = (e_c - e_s - e_na) * HARTREE_TO_KCAL
                log.append({"name": name, "complex": "noF_constrained", "E_xtb_Eh": "",
                            "Ebind_kcal_mol": round(best_nof, 2),
                            "NaO_min_A": round(min_dist(c2, p2, "O"), 3),
                            "NaF_min_A": round(min_dist(c2, p2, "F"), 3)})
            except Exception as err:
                print("  (constrained F-free optimisation failed:", err, ")")
    return best_all, best_nof


def ring_strain(name, n, workdir, basis, cache):
    check_rse_balance(n)

    def e_of(key, smi):
        if key not in cache:
            (_, sym, pos), _ = best_solvent(key, smi, workdir)
            cache[key] = dft_energy_opt(sym, pos, 0, basis)[0]
        return cache[key]

    e_ring = e_of(name, REFERENCE_SET[name])
    e_eth = e_of("ethane", "CC")
    e_dee = e_of("DEE", "CCOCC")
    e_pro = e_of("propane", "CCC") if n > 2 else 0.0
    left = e_ring + (n - 1) * e_eth
    right = e_dee + (n - 2) * e_pro
    return (left - right) * HARTREE_TO_KCAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="run only 3 molecules")
    ap.add_argument("--workdir", default="run_validate")
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--dft-opt", action="store_true",
                    help="re-optimise geometries at the DFT level (slower, closer to published work)")
    args = ap.parse_args()
    global DFT_OPT
    DFT_OPT = args.dft_opt
    os.makedirs(args.workdir, exist_ok=True)
    check_dispersion()

    names = QUICK if args.quick else list(REFERENCE_SET)
    rows, cache, log = [], {}, []
    for name in names:
        smi = REFERENCE_SET[name]
        print(f"[{name}] binding energy ...", flush=True)
        try:
            eb_all, eb_nof = binding_energy(name, smi, args.workdir, args.basis, log)
        except Exception as err:
            print("  FAILED:", err)
            eb_all = eb_nof = float("nan")
        rse = float("nan")
        if name in RSE_RINGS:
            print(f"[{name}] ring strain ...", flush=True)
            try:
                rse = ring_strain(name, RSE_RINGS[name], args.workdir, args.basis, cache)
            except Exception as err:
                print("  FAILED:", err)
        rows.append({"name": name, "smiles": smi,
                     "Ebind_global_kcal_mol": round(eb_all, 2),
                     "Ebind_noF_kcal_mol": round(eb_nof, 2),
                     "RSE_kcal_mol": round(rse, 2)})
        print(f"  Ebind(global) = {eb_all:.2f}   Ebind(noF) = {eb_nof:.2f}   RSE = {rse:.2f}  kcal/mol")

    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if log:
        with open("complex_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
            w.writeheader()
            w.writerows(log)
        print("written: complex_log.csv  (every complex: energy, Na-O and Na-F distances)")
    print("\nwritten: results.csv")
    print("Compare ORDER with Gao: EA -37.63, EDFA -38.38, DFEA -30.46 (kcal/mol);\n      Gao Na-O distances: EA 2.139, DFEA 2.167 A (see complex_log.csv)")


if __name__ == "__main__":
    main()
