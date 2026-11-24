# Reproduction code for *Chirality and Diameter Selection in Peptide Nanotubes from Microscopic Lattice Shear*

This repository contains the Python scripts used for numerical calculations, parameter fitting, and figure generation in the manuscript.
The digitized datasets used for fitting are embedded in the corresponding Python scripts. 
No separate data download is required.

## Setup

Tested with Python [3.10]. 

## Reproduce the figures

Run the following commands from the repository root. 
The table lists the expected output of each script.

| Manuscript result | Command | Expected output |
| Fig. 2 | `python3 plot_full_energy_landscape.py` | Figure_Full_Landscape.png |
| Fig. 3 | `python plot_two_Y.py -g 45 -o Figure_Kappa_Omega_45.png` | Figure_Kappa_Omega_45.png |
| Fig. 4 | `python3 figure4_plot_phase_diagram_full_hierachiy.py` |  PhaseDiagram_Full.png |
| Fig. 5a | `python3 figure5a_plot_pitch_radius_evolution.py`|  Figure_Curves.png |
| Fig. 5b | `python3 figure5b_fit_global_optimization.py` | Figure_Fitted_Perfect_Match.png |
| Fig. 6a | `python3 figure6a_plot_radius_vs_gamma_power4.py` | Radius_vs_Gamma_Universal.png |
| Fig. 6b | `python3 figure5b_extract_gamma_from_radius.py` | Figure_MasterCurve_3Points.png |
| Fig. S1 | `python3 figureS1_fit_pitch_evolution.py` | Figure_Fit_Reduced_Data.png |



## Citation

If you use this code, please cite the accompanying manuscript and specify the repository release or commit used: [release URL or commit hash].
