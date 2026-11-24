# PeptideAssemblyModel
PeptideAssemblyModel provides Python scripts for reproducing the numerical results and figures of a model that links molecular packing to the shape of chiral peptide assemblies. The scripts compute equilibrium ribbon shapes, fit radius and pitch as functions of width, and estimate terminal nanotube diameters. 

# Reproduction code for *Chirality and Diameter Selection in Peptide Nanotubes from Microscopic Lattice Shear*

This repository contains the Python scripts used for numerical
calculations, parameter fitting, and figure generation in the manuscript.
The digitized datasets used for fitting are embedded in the corresponding
Python scripts. No separate data download is required.

## Setup

Tested with Python [version]. From the repository root, run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the figures

Run the following commands from the repository root. The table lists the
expected output of each script.

| Manuscript result | Command | Expected output |
| --- | --- | --- |
| Fig. 4: curvature sign under shear reversal | `python <fig4_script.py>` | [figure filename] |
| Fig. 5(a): theoretical pitch and radius as functions of width | `python <fig5a_script.py>` | [figure filename] |
| Fig. 5(b): joint fit to the Zhang et al. nanoribbon data | `python <zhang_fit_script.py>` | [figure filename and fitted parameters] |
| Fig. 6: terminal diameter and conditional estimates of lattice shear | `python <fig6_script.py>` | [figure filename] |
| Fig. S1: fit to the Armon et al. elastic-strip data | `python <armon_fit_script.py>` | [figure filename and fitted parameters] |

## Fitting data

- **Zhang et al. [30], Fig. 3(d):** The plotted bin means of pitch and
  radius as functions of amphiphilic nanoribbon width were digitized with
  WebPlotDigitizer. These values are embedded in `[Zhang script path]`.
- **Armon et al. [28], Fig. 2:** The plotted data for the macroscopic elastic
  strips with \(t=2.3\) and \(k_0=0.18\) were digitized with WebPlotDigitizer.
  These values are embedded in `[Armon script path]`.

See Sec. S1 of the Supplemental Material for the digitization procedure
and Sec. VII A of the manuscript for the fitting objective and weights.

## Numerical verification

Run `python <benchmark_script.py>` to compare the numerical solution
with the analytical large-width limit. For \(\gamma=45^\circ\) and
\(\nu=1\), the reduced equilibrium curvature and torsion approach
\(\kappa^*=\Omega^*=1/2\).

## Citation

If you use this code, please cite the accompanying manuscript and specify
the repository release or commit used: [release URL or commit hash].
