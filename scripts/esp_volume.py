#!/usr/bin/env python3
"""
esp_volume.py -- ESP_min / Volume descriptor (as used by Niu et al., Angew. Chem. 2026).

Definition used here (the surface/volume definition may differ in detail from Niu et al.):
  * Geometry: DFT-optimised lowest-energy conformer (PBE0-D3(BJ)/def2-SVP).
  * Electron-density isosurface rho = 0.001 a.u. defines the molecular surface.
  * ESP_min  = most negative electrostatic potential on that surface   [kcal/mol]
  * Volume   = volume enclosed by that surface                        [A^3]
  * ESP_min / Volume  in kcal mol^-1 A^-3 (the units of Niu's Figure 1b axis).
A more negative value = stronger, more accessible Na+ binding site; methylation
should make it less negative (Niu: DOL most negative, 24diMeDOL less negative).

Reference ESP_min values from Gao et al. (eV, different code):  EA -2.029, EDFA -1.559,
DFEA -1.462  ->  order of 'most negative first':  EA < EDFA < DFEA.

Usage (same folder as validate_binding.py, same environment):
    python esp_volume.py --workdir run_esp
Output: results_esp.csv
"""
import argparse
import csv
import os

import numpy as np

HARTREE_TO_KCAL = 627.5094740631
BOHR3_TO_A3 = 0.529177210903 ** 3
MOLECULES = ["EA", "EDFA", "DFEA", "DOL", "2MeDOL", "4MeDOL", "24diMeDOL",
             "oxetane", "oxolane", "oxane"]


# ---------------------------------------------------------------- cube analysis
def read_cube(path):
    """Read a Gaussian cube file with orthogonal axes. Returns (data[nx,ny,nz], voxel_volume_bohr3)."""
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
    data = vals.reshape(dims)
    voxel = abs(np.linalg.det(np.array(vecs)))
    return data, voxel


def esp_over_volume(rho, esp, voxel_bohr3, iso=1e-3, band=0.15):
    """Return (ESP_min [kcal/mol], Volume [A^3], ESP_min/Volume).
    Surface points = grid points whose density lies within +/- band*iso of the isovalue."""
    inside = rho > iso
    volume = float(inside.sum()) * voxel_bohr3 * BOHR3_TO_A3
    surf = np.abs(rho - iso) < band * iso
    if not surf.any():
        raise ValueError("no surface points found; refine the grid or widen band")
    esp_min = float(esp[surf].min()) * HARTREE_TO_KCAL
    return esp_min, volume, esp_min / volume


# ---------------------------------------------------------------- DFT + cubes
def descriptor_for(name, vb, workdir, basis, resolution):
    from pyscf import gto, dft
    from pyscf.tools import cubegen

    smiles = vb.REFERENCE_SET[name]
    (e_x, s_sym, s_pos), mol_rd = vb.best_solvent(name, smiles, workdir)
    e, sym, pos = vb.dft_energy_opt(s_sym, s_pos, 0, basis)

    atom = [(s, tuple(c)) for s, c in zip(sym, pos)]
    mol = gto.M(atom=atom, charge=0, spin=0, basis=basis, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "pbe0"
    try:
        mf.disp = "d3bj"
    except Exception:
        pass
    mf = mf.density_fit()
    mf.kernel()
    dm = mf.make_rdm1()

    rho_f = os.path.join(workdir, f"{name}_rho.cube")
    esp_f = os.path.join(workdir, f"{name}_esp.cube")
    cubegen.density(mol, rho_f, dm, resolution=resolution, margin=5.0)
    cubegen.mep(mol, esp_f, dm, resolution=resolution, margin=5.0)
    rho, vox = read_cube(rho_f)
    esp, _ = read_cube(esp_f)
    return esp_over_volume(rho, esp, vox)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="run_esp")
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--resolution", type=float, default=0.3, help="grid spacing in Bohr (smaller = slower, finer)")
    ap.add_argument("--no-dft-opt", action="store_true", help="skip DFT geometry optimisation (faster, less consistent)")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    import validate_binding as vb
    vb.DFT_OPT = not args.no_dft_opt
    vb.check_dispersion()
    os.makedirs(args.workdir, exist_ok=True)

    rows = []
    for name in (args.only or MOLECULES):
        print(f"[{name}] ESP_min / Volume ...", flush=True)
        try:
            emin, vol, ratio = descriptor_for(name, vb, args.workdir, args.basis, args.resolution)
        except Exception as err:
            print("  FAILED:", err)
            emin = vol = ratio = float("nan")
        rows.append({"name": name, "ESPmin_kcal_mol": round(emin, 2),
                     "Volume_A3": round(vol, 2),
                     "ESPmin_over_V_kcal_mol_A3": round(ratio, 4)})
        print(f"  ESPmin = {emin:.2f} kcal/mol   V = {vol:.2f} A^3   ratio = {ratio:.4f}")
        with open("results_esp.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print("\nwritten: results_esp.csv")
    print("Check: Niu order (most negative first): DOL < 4MeDOL ~ 2MeDOL < 24diMeDOL;  Gao: EA < EDFA < DFEA")


if __name__ == "__main__":
    main()
