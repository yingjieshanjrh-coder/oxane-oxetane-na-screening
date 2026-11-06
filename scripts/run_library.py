#!/usr/bin/env python3
"""
run_library.py -- full-library calculation (134 fluorinated / methylated oxetane and oxane solvents).

Per molecule it computes (all with the validated settings):
  * Na+ binding energy, global minimum and F-free reference   [PBE0-D3(BJ)/def2-SVP, DFT geometries]
  * ESP_min, volume, ESP_min/Volume                           [M06-2X/def2-TZVP + ddCOSMO(THF), 0.001 a.u. surface]
  * ring-strain release energy, Niu-type ring-opening exchange   RSE = -(E_prod - E_ring - E_EtOMe)
    (positive = strain released on opening; Niu's tables report the same sign)
Ring-opened product: the ring C-O bond at the MORE substituted alpha carbon is cleaved (methoxy goes
there, ethyl goes on the former ring oxygen), as in Niu's Figure S6.  Ties: first carbon.

How to use (all in the same folder as validate_binding.py, validate_niu.py, enumerate_library.py):
    python run_library.py --list                         # show the library
    python run_library.py --ids M0001 M0002              # test 2 molecules in the foreground
    python run_library.py --launch 4                     # start 4 detached workers (use your 4 cores)
    python run_library.py --status                       # progress
    python run_library.py --collect                      # write library_results.csv
Stop everything:   pkill -f run_library.py ; pkill xtb
Resume after any interruption: just run --launch 4 again (finished molecules are skipped).
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
import traceback

HARTREE_TO_KCAL = 627.5094740631
WORKDIR_DEFAULT = "run_lib"
EME = "COCC"


# ----------------------------------------------------------------- library
def build_library():
    """Same molecules and ids as enumerate_library.py, plus ring-opened product SMILES."""
    import enumerate_library as el
    rows = []
    for name, n_c in el.SCAFFOLDS.items():
        seen = el.enumerate_scaffold(name, n_c)
        for key, (config, nF) in sorted(seen.items(), key=lambda kv: (kv[1][1], str(kv[0]))):
            ring = el.build_smiles(config)
            chain = "".join(el.carbon_smiles(subs, "") for subs in config)
            if len(config[-1]) > len(config[0]):
                product = "CCO" + chain + "OC"      # methoxy on the last (more substituted) carbon
            else:
                product = "CO" + chain + "OCC"      # methoxy on the first carbon
            rows.append({"scaffold": name, "ring": ring, "product": product, "nF": nF,
                         "config": [list(c) for c in config]})
    for i, r in enumerate(rows, 1):
        r["id"] = f"M{i:04d}"
    return rows


def heavy_counts(smiles):
    import re
    d = {"C": 0, "O": 0, "F": 0}
    for ch in re.findall(r"[COF]", smiles):
        d[ch] += 1
    return d


def check_balance(r):
    left = {k: heavy_counts(r["ring"])[k] + heavy_counts(EME)[k] for k in "COF"}
    return left == heavy_counts(r["product"])


# ----------------------------------------------------------------- files
def paths(workdir):
    res = os.path.join(workdir, "results")
    clm = os.path.join(workdir, "claims")
    os.makedirs(res, exist_ok=True)
    os.makedirs(clm, exist_ok=True)
    return res, clm


def claim(clm, mid):
    try:
        fd = os.open(os.path.join(clm, mid + ".lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


# ----------------------------------------------------------------- chemistry for one molecule
def ensure_eme(workdir, vb, vn):
    path = os.path.join(workdir, "eme.json")
    if os.path.exists(path):
        return json.load(open(path))["E"]
    seed = os.path.join("run_niu", "energies_m062x.json")      # reuse the validation run if present
    if os.path.exists(seed):
        c = json.load(open(seed))
        if "EME|sol" in c:
            json.dump({"E": c["EME|sol"], "source": "run_niu"}, open(path, "w"))
            return c["EME|sol"]
    sym, pos = vn.optimised_geometry(vb, "EME", EME, workdir, "def2-svp")
    e = vn.m062x_sp(sym, pos)
    json.dump({"E": e, "source": "computed"}, open(path, "w"))
    return e


def run_one(r, workdir, basis_opt="def2-svp", resolution=0.3):
    import validate_binding as vb
    import validate_niu as vn
    vb.DFT_OPT = True
    vn.USE_SOLVENT = True
    res_dir, _ = paths(workdir)
    out = os.path.join(res_dir, r["id"] + ".json")
    if os.path.exists(out):
        return
    t0 = time.time()
    res = {"id": r["id"], "scaffold": r["scaffold"], "ring_smiles": r["ring"], "product_smiles": r["product"],
           "n_F_atoms": r["nF"], "errors": []}
    # 1) Na+ binding energies
    try:
        log = []
        eb_all, eb_nof = vb.binding_energy(r["id"] + "_bind", r["ring"], workdir, basis_opt, log)
        res["Ebind_global_kcal_mol"] = round(eb_all, 2)
        res["Ebind_noF_kcal_mol"] = None if eb_nof != eb_nof else round(eb_nof, 2)
        res["complexes"] = log
    except Exception as err:
        res["errors"].append("binding: " + type(err).__name__ + ": " + str(err)[:200])
        print(traceback.format_exc(), flush=True)
    # 2) ring strain (Niu-type exchange) and ESP descriptors
    try:
        eme = ensure_eme(workdir, vb, vn)
        sym, pos = vn.optimised_geometry(vb, r["id"], r["ring"], workdir, basis_opt)
        e_ring, mol, dm = vn.m062x_sp(sym, pos, want_dm=True)
        sym2, pos2 = vn.optimised_geometry(vb, r["id"] + "_open", r["product"], workdir, basis_opt)
        e_prod = vn.m062x_sp(sym2, pos2)
        res["RSE_release_kcal_mol"] = round(-(e_prod - e_ring - eme) * HARTREE_TO_KCAL, 2)
        try:
            emin, vol, ratio = vn.esp_descriptors(mol, dm, r["id"], workdir, resolution)
            res["ESPmin_kcal_mol"] = round(emin, 2)
            res["Volume_A3"] = round(vol, 1)
            res["ESPmin_over_V"] = round(ratio, 4)
        except Exception as err:
            res["errors"].append("esp: " + type(err).__name__ + ": " + str(err)[:200])
            print(traceback.format_exc(), flush=True)
    except Exception as err:
        res["errors"].append("ring/RSE: " + type(err).__name__ + ": " + str(err)[:200])
        print(traceback.format_exc(), flush=True)
    res["seconds"] = round(time.time() - t0)
    tmp = out + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, out)
    print(f"[{r['id']}] done in {res['seconds']} s  errors={len(res['errors'])}", flush=True)


# ----------------------------------------------------------------- modes
def worker_loop(workdir, wid, ids=None):
    rows = build_library()
    res_dir, clm = paths(workdir)
    for r in rows:
        if ids and r["id"] not in ids:
            continue
        if os.path.exists(os.path.join(res_dir, r["id"] + ".json")):
            continue
        if not claim(clm, r["id"]):
            continue
        print(f"[worker {wid}] {r['id']} {r['scaffold']} {r['ring']}", flush=True)
        try:
            run_one(r, workdir)
        finally:
            try:
                os.remove(os.path.join(clm, r["id"] + ".lock"))
            except OSError:
                pass
    print(f"[worker {wid}] no more molecules to do", flush=True)


def launch(n, workdir):
    rows = build_library()
    bad = [r["id"] for r in rows if not check_balance(r)]
    if bad:
        sys.exit("atom balance failed for " + str(bad[:5]))
    res_dir, clm = paths(workdir)
    for f in glob.glob(os.path.join(clm, "*.lock")):      # nothing is running now, so old locks are stale
        os.remove(f)
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"
    # one-off shared quantity, computed before the workers start
    import validate_binding as vb
    import validate_niu as vn
    vb.DFT_OPT = True
    vn.USE_SOLVENT = True
    print("preparing the ethyl methyl ether reference ...", flush=True)
    ensure_eme(workdir, vb, vn)
    pids = []
    for i in range(n):
        log = open(os.path.join(workdir, f"worker{i}.log"), "a")
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker", str(i),
                              "--workdir", workdir], stdout=log, stderr=subprocess.STDOUT,
                             env=env, start_new_session=True)
        pids.append(p.pid)
    print(f"started {n} detached workers, PIDs {pids}")
    print("they keep running if you close the terminal.  Progress:  python run_library.py --status")


def status(workdir):
    rows = build_library()
    res_dir, clm = paths(workdir)
    done = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(res_dir, "M*.json"))}
    running = [os.path.basename(f)[:-5] for f in glob.glob(os.path.join(clm, "*.lock"))]
    n_err = sum(1 for d in done.values() if d.get("errors"))
    print(f"finished {len(done)}/{len(rows)}   with errors: {n_err}   in progress: {sorted(running)}")
    if done:
        by = {}
        for d in done.values():
            if "seconds" in d:
                by.setdefault(d["scaffold"], []).append(d["seconds"])
        allsec = [x for v in by.values() for x in v]
        overall = sum(allsec) / len(allsec)
        means = {k: sum(v) / len(v) for k, v in by.items()}
        for k, v in sorted(by.items()):
            print(f"  {k:8s}: {len(v):3d} done, mean {means[k]/60:5.1f} min per molecule (per worker)")
        # estimate the rest with the mean of each molecule's OWN scaffold (falls back to the overall mean)
        left = [r for r in rows if r["id"] not in done]
        total = sum(means.get(r["scaffold"], overall) for r in left)
        nw = max(1, len(running))
        print(f"remaining molecules: {len(left)}   workers running: {nw}")
        print(f"rough time left: {total/nw/3600:.1f} h   (later molecules have more substituents, so expect it to run a bit longer)")
    for k, d in sorted(done.items()):
        if d.get("errors"):
            print("  ", k, d["errors"][0][:110])


def collect(workdir, out="library_results.csv"):
    rows = build_library()
    res_dir, _ = paths(workdir)
    cols = ["id", "scaffold", "ring_smiles", "n_F_atoms", "Ebind_global_kcal_mol", "Ebind_noF_kcal_mol",
            "RSE_release_kcal_mol", "ESPmin_kcal_mol", "Volume_A3", "ESPmin_over_V", "seconds", "errors"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        n = 0
        for r in rows:
            p = os.path.join(res_dir, r["id"] + ".json")
            if os.path.exists(p):
                d = json.load(open(p))
                d["errors"] = "; ".join(d.get("errors", []))
                w.writerow(d)
                n += 1
    print(f"wrote {out} with {n} molecules")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=WORKDIR_DEFAULT)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--ids", nargs="*", help="run these molecules in the foreground (test)")
    ap.add_argument("--launch", type=int, metavar="N")
    ap.add_argument("--worker", type=int, help="(internal) worker id")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    if a.list:
        for r in build_library():
            print(r["id"], r["scaffold"], r["nF"], r["ring"], "->", r["product"])
    elif a.ids:
        os.makedirs(a.workdir, exist_ok=True)
        import validate_binding as vb
        import validate_niu as vn
        vb.DFT_OPT = True
        vn.USE_SOLVENT = True
        ensure_eme(a.workdir, vb, vn)
        worker_loop(a.workdir, "test", ids=set(a.ids))
    elif a.launch:
        os.makedirs(a.workdir, exist_ok=True)
        launch(a.launch, a.workdir)
    elif a.worker is not None:
        worker_loop(a.workdir, a.worker)
    elif a.status:
        status(a.workdir)
    elif a.collect:
        collect(a.workdir)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
