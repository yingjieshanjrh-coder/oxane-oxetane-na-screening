#!/usr/bin/env python3
"""
Enumerate substituted oxetane and oxane (tetrahydropyran) solvent candidates.

Chemistry rules (edit the CONFIG block to change them):
  1. Ring carbons carry up to 2 non-H substituents from SUBSTITUENTS.
  2. No fluorine DIRECTLY on a carbon bonded to ring oxygen (alpha-fluoroethers
     are hydrolytically labile and release HF). CF3/CH2F/CHF2 groups on alpha
     carbons ARE allowed (the F is not on the O-bonded carbon).
  3. At most MAX_SUBS non-H substituents in total, at most MAX_CARBON_SUBS
     carbon-containing groups.
  4. Keep the unfluorinated parents and methyl-only analogues as a reference
     set (flag: is_fluorinated = False).
  5. Duplicates removed by ring-reflection symmetry (stereochemistry ignored
     at enumeration stage; handle stereoisomers later if needed).

Output: library.csv with columns id, scaffold, smiles, n_F, is_fluorinated
Usage:  python enumerate_library.py
Optional: if RDKit is installed, every SMILES is parsed and checked.
"""
import csv
import itertools

# ---------------- CONFIG ----------------
SCAFFOLDS = {"oxetane": 3, "oxane": 5}      # number of ring carbons
SUBSTITUENTS = {                              # name -> SMILES fragment
    "F": "F",
    "CH3": "C",
    "CF3": "C(F)(F)F",
    "CH2F": "CF",
    "CHF2": "C(F)F",
}
MAX_SUBS = 3
MAX_CARBON_SUBS = 1
MAX_PER_CARBON = 2
# ----------------------------------------

def n_fluorine(sub):
    return {"F": 1, "CH3": 0, "CF3": 3, "CH2F": 1, "CHF2": 2}[sub]

def is_carbon_group(sub):
    return sub != "F"

def carbon_options(is_alpha):
    """All allowed substituent multisets for one ring carbon."""
    names = [s for s in SUBSTITUENTS if not (is_alpha and s == "F")]
    opts = [()]
    for k in range(1, MAX_PER_CARBON + 1):
        opts += list(itertools.combinations_with_replacement(sorted(names), k))
    return opts

def carbon_smiles(subs, ring_label=""):
    s = "C" + ring_label
    for sub in subs:
        s += "(" + SUBSTITUENTS[sub] + ")"
    return s

def build_smiles(config):
    n = len(config)
    parts = []
    for i, subs in enumerate(config):
        parts.append(carbon_smiles(subs, "1" if i == 0 else ""))
    return "".join(parts) + "O1"

def canonical_key(config):
    fwd = tuple(config)
    rev = tuple(reversed(config))
    return min(fwd, rev)

def enumerate_scaffold(name, n_c):
    per_carbon = [carbon_options(i in (0, n_c - 1)) for i in range(n_c)]
    seen = {}
    for config in itertools.product(*per_carbon):
        flat = [s for subs in config for s in subs]
        if len(flat) > MAX_SUBS:
            continue
        if sum(is_carbon_group(s) for s in flat) > MAX_CARBON_SUBS:
            continue
        key = canonical_key(config)
        if key in seen:
            continue
        nF = sum(n_fluorine(s) for s in flat)
        seen[key] = (config, nF)
    return seen

def main():
    rows = []
    for name, n_c in SCAFFOLDS.items():
        seen = enumerate_scaffold(name, n_c)
        for key, (config, nF) in sorted(seen.items(), key=lambda kv: (kv[1][1], str(kv[0]))):
            rows.append({
                "scaffold": name,
                "smiles": build_smiles(config),
                "n_F": nF,
                "is_fluorinated": nF > 0,
            })
    try:
        from rdkit import Chem
        bad = 0
        for r in rows:
            m = Chem.MolFromSmiles(r["smiles"])
            if m is None:
                bad += 1
                r["smiles"] = "INVALID:" + r["smiles"]
            else:
                r["smiles"] = Chem.MolToSmiles(m, isomericSmiles=False)
        print(f"RDKit check done, invalid SMILES: {bad}")
    except ImportError:
        print("RDKit not found: SMILES not validated (install it, then rerun).")

    for i, r in enumerate(rows, 1):
        r["id"] = f"M{i:04d}"
    with open("library.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "scaffold", "smiles", "n_F", "is_fluorinated"])
        w.writeheader()
        w.writerows(rows)

    for name in SCAFFOLDS:
        sub = [r for r in rows if r["scaffold"] == name]
        nf = sum(r["is_fluorinated"] for r in sub)
        print(f"{name}: {len(sub)} molecules ({nf} fluorinated, {len(sub)-nf} reference)")
    print(f"total: {len(rows)} -> library.csv")

if __name__ == "__main__":
    main()
