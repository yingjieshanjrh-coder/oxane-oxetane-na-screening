# Descriptor-based screening of fluorinated oxetane and oxane (tetrahydropyran) solvents for sodium electrolytes

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24047648.svg)](https://doi.org/10.5281/zenodo.24047648)

Code and data for the manuscript *Descriptor-Based Computational Screening of Fluorinated Oxetane and Tetrahydropyran Solvents for Low-Temperature Sodium Electrolytes* (Yingjie Shan, School of Chemistry, University of St Andrews).

All results are computed (no experiments). The library has 134 methylated and fluorinated ring ethers (23 oxetanes, 111 oxanes).

## Software

PySCF 2.14.0, pyscf-dispersion 1.5.0, geomeTRIC 1.1.1, xtb 6.7.1, RDKit 2026.9.1, NumPy 2.4.6, SciPy 1.17.1, pandas 3.1.0, Matplotlib 3.11.2.
Installation on Windows (WSL2 + conda): see `SETUP_WSL2.md`.

## Workflow (run from the `scripts/` folder, in this order)

| Step | Script | Output |
|---|---|---|
| 1. Build the library | `enumerate_library.py` | library definition |
| 2. Validate the binding protocol | `validate_binding.py`, `validate_niu.py` | validation tables |
| 3. Library calculation (conformers, DFT optimisation, Na+ binding, ESPmin, strain release) | `run_library.py` (`--launch N`, `--status`, `--collect`) | `library_results.csv` |
| 4. Dipole moments | `compute_dipole.py` | `dipoles.csv` |
| 5. HOMO/LUMO | `compute_lumo.py` | `lumo.csv` |
| 6. Proton affinity | `compute_pa.py`, `pa_reference.py` (check against NIST) | `pa.csv` |
| 7. Pareto analysis | `analyze_library.py`, `analyze_final.py`, `extra_analysis.py` | rank tables, summaries |
| 8. Figures | `make_figures.py --data ../data --out ../figures` | `figures/` |

`esp_volume.py` holds the ESPmin/volume routines used by the other scripts.

## Data (`data/`)

* `SI_Table_S2_all_134_molecules.csv`: all descriptors for all 134 molecules.
* `library_results.csv`, `dipoles.csv`, `pa.csv`, `lumo.csv`: raw result tables.
* `ranks_main.csv`, `front_main.csv`, `better_than_parent.csv`, `sensitivity_final.csv`, `summary_final.txt`: output of the analysis.

Units: energies in kcal/mol, volumes in cubic angstrom, dipole moments in debye, orbital energies in eV.
`Ebind_noF_kcal_mol` is available for 40 of the 134 molecules only (those with an F-free reference).

## Levels of theory

Geometries and Na+ binding: PBE0-D3(BJ)/def2-SVP. ESPmin and volume: M06-2X/def2-TZVP with ddCOSMO (epsilon 7.43), 0.001 a.u. density surface. Proton affinity: electronic energy difference, no zero-point correction.

## Known limitations

See the Limitations section of the manuscript. In particular, the ESPmin window used to define the candidate set was set after the results were known, and the dipole moment is an unvalidated polarity proxy.

## License

MIT (see `LICENSE`).

## Citation

If you use this code or data, please cite the archived release:

Shan, Y. Code and data for the computational screening of fluorinated oxetane and oxane solvents for sodium electrolytes. Version v1.0.0. Zenodo, 2026. https://doi.org/10.5281/zenodo.24047648
