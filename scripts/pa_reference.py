#!/usr/bin/env python3
"""
pa_reference.py -- check the proton-affinity protocol against experiment.

Experimental gas-phase proton affinities (NIST WebBook, evaluated by Hunter and Lias, 1998):
    tetrahydrofuran   822.1 kJ/mol = 196.5 kcal/mol
    tetrahydropyran   822.8 kJ/mol = 196.7 kcal/mol
Our PA is an ELECTRONIC energy difference (no zero-point or thermal terms), so a roughly constant offset of a few
kcal/mol (the zero-point energy gained by the new O-H bond) is expected; what matters for the screen is that
differences between molecules are right.  Oxetane is run for information only (no experimental value was found).

Usage (same folder as validate_binding.py and compute_pa.py):
    python pa_reference.py
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")      # small molecules: use the 4 cores of this single process

import numpy as np

KJ_TO_KCAL = 1.0 / 4.184
HARTREE_TO_KCAL = 627.5094740631
REFS = [
    ("tetrahydrofuran", "C1CCOC1", 822.1 * KJ_TO_KCAL),
    ("tetrahydropyran", "C1CCCCO1", 822.8 * KJ_TO_KCAL),
    ("oxetane", "C1COC1", None),
]


def main():
    import validate_binding as vb
    import compute_pa as cp
    vb.DFT_OPT = True
    wd = "run_ref"
    os.makedirs(wd, exist_ok=True)
    rows = []
    for name, smi, exp in REFS:
        (_, sym, pos), _ = vb.best_solvent("ref_" + name, smi, wd)
        e_neu, sym, pos = vb.dft_energy_opt(sym, pos, 0, "def2-svp")
        sym2, pos2, o = cp.protonate(list(sym), np.array(pos))
        e_prot, sym3, pos3 = vb.dft_energy_opt(sym2, pos2.tolist(), +1, "def2-svp")
        pa = (e_neu - e_prot) * HARTREE_TO_KCAL
        opened = cp.ring_opened(sym3, np.array(pos3), sym3.index("O"))
        rows.append((name, pa, exp, opened))
        print(f"{name:16s} PA(electronic) = {pa:6.1f} kcal/mol   experiment = "
              + (f"{exp:6.1f}" if exp else "   n/a") + f"   difference = " + (f"{pa-exp:+5.1f}" if exp else "n/a")
              + ("   (ring opened on protonation)" if opened else ""), flush=True)
    thf, thp = rows[0], rows[1]
    print(f"\nTHP minus THF: calculated {thp[1]-thf[1]:+.1f} kcal/mol, experiment {thp[2]-thf[2]:+.1f} kcal/mol")
    print("Please send this output.  If the offset is about 7-9 kcal/mol for both, it is the size of the zero-point correction.")


if __name__ == "__main__":
    main()
