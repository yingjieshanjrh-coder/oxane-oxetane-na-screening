#!/usr/bin/env python3
"""
validate_niu.py -- one run that checks our pipeline against Niu et al. (Angew. Chem. 2026),
Supporting Information Table S1 and Figure S6.

What it reproduces (reference numbers are copied from the SI, Table S1):
  * Ring strain energy (their Figure S6 scheme):
        ring + CH3OCH2CH3  ->  CH3O-...-OCH2CH3 (ring opened)
        E_strain = E(product) - E(ring) - E(CH3OC2H5)         [kcal/mol]
    The SAME scheme is applied to oxetane / oxolane / oxane (product = CH3O-(CH2)n-OEt),
    so the library can later use exactly their definition.
  * ESP_min (kcal/mol), molecular volume (A^3) and ESP_min/Volume.

Their level: Gaussian 16, M06-2X/def2-TZVP, THF implicit solvent, ESP/volume with Multiwfn.
Ours:  geometry PBE0-D3(BJ)/def2-SVP (via validate_binding.py), then M06-2X/def2-TZVP single
points with ddCOSMO(THF, eps=7.43); ESP/volume from the 0.001 a.u. density isosurface.
Small differences are expected (geometry, solvent model, surface definition).

Usage (same folder as validate_binding.py; esp_volume.py is NOT needed any more):
    python validate_niu.py --workdir run_niu
    python validate_niu.py --workdir run_niu --only DOL 24diMeDOL        # quick test
    python validate_niu.py --workdir run_niu --selftest                  # find which stage fails
    python validate_niu.py --workdir run_niu --gas                       # without implicit solvent
Output: results_niu.csv (or the name given with --out)
"""
import argparse
import csv
import json
import logging
import os
import traceback

import numpy as np

HARTREE_TO_KCAL = 627.5094740631
EME = "COCC"   # ethyl methyl ether

# name: (ring SMILES, ring-opened product SMILES)
REACTIONS = {
    "DOL":       ("C1OCCO1",         "COCCOCOCC"),
    "2MeDOL":    ("CC1OCCO1",        "COCCOC(C)OCC"),
    "4MeDOL":    ("CC1COCO1",        "COC(C)COCOCC"),
    "24diMeDOL": ("CC1COC(C)O1",     "COC(C)COC(C)OCC"),
    "TFDOL":     ("FC(F)(F)C1COCO1", "COC(C(F)(F)F)COCOCC"),
    "22diMeDOL": ("CC1(C)OCCO1",     "COCCOC(C)(C)OCC"),
    "OMeDOL":    ("COC1OCCO1",       "COCCOC(OC)OCC"),
    # our own parent rings, same scheme (not in Niu's table):
    "oxetane":   ("C1COC1",          "COCCCOCC"),
    "oxolane":   ("C1CCOC1",         "COCCCCOCC"),
    "oxane":     ("C1CCOCC1",        "COCCCCCOCC"),
}

# Niu SI Table S1: RSE (kcal/mol), ESPmin (kcal/mol), Volume (A^3), ESPmin/Volume
TABLE_S1 = {
    "DOL":       (4.16535, -32.19844,  95.477, -0.33724),
    "2MeDOL":    (1.94086, -32.74598, 118.383, -0.27661),
    "4MeDOL":    (3.02392, -33.02941, 119.682, -0.27598),
    "24diMeDOL": (0.84587, -33.56676, 142.247, -0.23598),
    "TFDOL":     (1.76641, -27.40092, 134.472, -0.20377),
    "22diMeDOL": (0.96570, -32.80921, 141.591, -0.23172),
    "OMeDOL":    (2.73841, -39.45492, 129.485, -0.30471),
}
DEFAULT_SET = ["DOL", "2MeDOL", "4MeDOL", "24diMeDOL", "TFDOL", "22diMeDOL", "OMeDOL",
               "oxetane", "oxolane", "oxane"]


BOHR3_TO_A3 = 0.529177210903 ** 3


def read_cube(path):
    """Read a Gaussian cube file with orthogonal axes -> (data[nx,ny,nz], voxel volume in Bohr^3)."""
    with open(path) as f:
        f.readline()
        f.readline()
        natoms = abs(int(f.readline().split()[0]))
        dims, vecs = [], []
        for _ in range(3):
            p = f.readline().split()
            dims.append(int(p[0]))
            vecs.append([float(p[1]), float(p[2]), float(p[3])])
        for _ in range(natoms):
            f.readline()
        vals = np.array(f.read().split(), dtype=float)
    return vals.reshape(dims), abs(np.linalg.det(np.array(vecs)))


def esp_over_volume(rho, esp, voxel_bohr3, iso=1e-3, band=0.15):
    """(ESP_min [kcal/mol], Volume [A^3], ESP_min/Volume) on the rho = 0.001 a.u. isosurface."""
    volume = float((rho > iso).sum()) * voxel_bohr3 * BOHR3_TO_A3
    surf = np.abs(rho - iso) < band * iso
    if not surf.any():
        raise ValueError("no surface points found; refine the grid or widen band")
    esp_min = float(esp[surf].min()) * HARTREE_TO_KCAL
    return esp_min, volume, esp_min / volume


USE_SOLVENT = True


def m062x_sp(symbols, coords, basis="def2-tzvp", eps=7.43, want_dm=False):
    """M06-2X single point with ddCOSMO (THF-like). Returns E (Hartree) [, mol, dm]."""
    from pyscf import gto, dft, solvent
    atom = [(s, tuple(c)) for s, c in zip(symbols, coords)]
    mol = gto.M(atom=atom, charge=0, spin=0, basis=basis, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "m062x"
    mf.grids.level = 3
    mf = mf.density_fit()
    if USE_SOLVENT:
        mf = solvent.ddCOSMO(mf)
        mf.with_solvent.eps = eps
    e = float(mf.kernel())
    if want_dm:
        return e, mol, mf.make_rdm1()
    return e


def optimised_geometry(vb, tag, smiles, workdir, basis_opt):
    """DFT-optimised lowest conformer. Saved to <workdir>/<tag>_dftopt.xyz and reused on re-runs."""
    path = os.path.join(workdir, f"{tag}_dftopt.xyz")
    if os.path.exists(path):
        sym, pos = vb.read_xyz(path)
        return sym, np.array(pos)
    (_, sym, pos), _ = vb.best_solvent(tag, smiles, workdir)
    _, sym, pos = vb.dft_energy_opt(sym, pos, 0, basis_opt)
    vb.write_xyz(path, sym, pos, f"{tag} DFT-optimised")
    return sym, np.array(pos)


def esp_descriptors(mol, dm, tag, workdir, resolution):
    from pyscf.tools import cubegen
    rho_f = os.path.join(workdir, f"{tag}_rho.cube")
    esp_f = os.path.join(workdir, f"{tag}_esp.cube")
    cubegen.density(mol, rho_f, dm, resolution=resolution, margin=5.0)
    cubegen.mep(mol, esp_f, dm, resolution=resolution, margin=5.0)
    rho, vox = read_cube(rho_f)
    esp, _ = read_cube(esp_f)
    return esp_over_volume(rho, esp, vox)


def selftest(vb, workdir):
    """Run each stage on ethyl methyl ether and report exactly which one fails."""
    global USE_SOLVENT
    stages = []
    state = {}

    def stage(label):
        def deco(fn):
            stages.append((label, fn))
            return fn
        return deco

    @stage("1  conformer search + xtb")
    def _():
        (_, sym, pos), _m = vb.best_solvent("selftest", EME, workdir)
        state["sym"], state["pos"] = sym, pos

    @stage("2  DFT geometry optimisation (PBE0-D3/def2-SVP, geomeTRIC)")
    def _():
        _e, sym, pos = vb.dft_energy_opt(state["sym"], state["pos"], 0, "def2-svp")
        state["sym"], state["pos"] = sym, pos

    @stage("3  M06-2X/def2-TZVP single point, gas phase")
    def _():
        global USE_SOLVENT
        old = USE_SOLVENT
        USE_SOLVENT = False
        try:
            m062x_sp(state["sym"], state["pos"])
        finally:
            USE_SOLVENT = old

    @stage("4  M06-2X/def2-TZVP single point + ddCOSMO (THF)")
    def _():
        global USE_SOLVENT
        old = USE_SOLVENT
        USE_SOLVENT = True
        try:
            e, mol, dm = m062x_sp(state["sym"], state["pos"], want_dm=True)
            state["mol"], state["dm"] = mol, dm
        finally:
            USE_SOLVENT = old

    @stage("5  density / ESP cube files + surface analysis")
    def _():
        r = esp_descriptors(state["mol"], state["dm"], "selftest", workdir, 0.3)
        print("      (ESPmin, volume, ratio) =", tuple(round(x, 3) for x in r))

    for label, fn in stages:
        try:
            fn()
            print("PASS  " + label, flush=True)
        except Exception:
            print("FAIL  " + label)
            print(traceback.format_exc())
            print("Stopping here; please send this output.")
            return
    print("All stages passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="run_niu")
    ap.add_argument("--opt-basis", default="def2-svp", help="basis for the DFT geometry optimisation")
    ap.add_argument("--resolution", type=float, default=0.3)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default="results_niu.csv", help="name of the output csv (change it each run to avoid overwriting)")
    ap.add_argument("--selftest", action="store_true", help="test each stage on one small molecule and stop")
    ap.add_argument("--gas", action="store_true", help="skip the implicit-solvent model (use if ddCOSMO fails)")
    args = ap.parse_args()

    import validate_binding as vb
    vb.DFT_OPT = True
    vb.check_dispersion()
    os.makedirs(args.workdir, exist_ok=True)
    global USE_SOLVENT
    USE_SOLVENT = not args.gas
    if args.selftest:
        selftest(vb, args.workdir)
        return

    logging.getLogger("geometric").setLevel(logging.WARNING)   # quieter geomeTRIC output (best effort)

    cache_path = os.path.join(args.workdir, "energies_m062x.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    mode = "sol" if USE_SOLVENT else "gas"

    def energy_of(tag, smiles):
        key = f"{tag}|{mode}"
        if key not in cache:
            sym, pos = optimised_geometry(vb, tag, smiles, args.workdir, args.opt_basis)
            cache[key] = m062x_sp(sym, pos)
            json.dump(cache, open(cache_path, "w"))
        return cache[key]

    rows = []
    for name in (args.only or DEFAULT_SET):
        ring, prod = REACTIONS[name]
        print(f"[{name}] ring strain + ESP descriptors ...", flush=True)
        row = {"name": name, "RSE_ours": float("nan"), "ESPmin_ours": float("nan"),
               "Vol_ours": float("nan"), "ESPmin_over_V_ours": float("nan"), "error": ""}
        try:
            sym, pos = optimised_geometry(vb, name, ring, args.workdir, args.opt_basis)
            e_ring, mol, dm = m062x_sp(sym, pos, want_dm=True)
            e_prod = energy_of(name + "_open", prod)
            e_eme = energy_of("EME", EME)
            row["RSE_ours"] = round((e_prod - e_ring - e_eme) * HARTREE_TO_KCAL, 2)
            print("  ring strain (their scheme):", row["RSE_ours"], "kcal/mol", flush=True)
            try:
                emin, vol, ratio = esp_descriptors(mol, dm, name, args.workdir, args.resolution)
                row.update({"ESPmin_ours": round(emin, 2), "Vol_ours": round(vol, 1),
                            "ESPmin_over_V_ours": round(ratio, 4)})
            except Exception as err:
                print("  ESP step FAILED:", err)
                print(traceback.format_exc())
                row["error"] = ("ESP: " + type(err).__name__ + ": " + str(err))[:200].replace("\n", " ")
        except Exception as err:
            print("  FAILED:", err)
            print(traceback.format_exc())
            row["error"] = (type(err).__name__ + ": " + str(err))[:200].replace("\n", " ")
        ref = TABLE_S1.get(name)
        if ref:
            row.update({"RSE_Niu": round(ref[0], 2), "ESPmin_Niu": round(ref[1], 2),
                        "Vol_Niu": round(ref[2], 1), "ESPmin_over_V_Niu": round(ref[3], 4)})
        else:
            row.update({"RSE_Niu": "", "ESPmin_Niu": "", "Vol_Niu": "", "ESPmin_over_V_Niu": ""})
        row.setdefault("error", "")
        rows.append(row)
        print("  ours:", {k: row[k] for k in ("RSE_ours", "ESPmin_ours", "Vol_ours", "ESPmin_over_V_ours")})
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\nwritten: {args.out}  (compare *_ours with *_Niu)")


if __name__ == "__main__":
    main()
