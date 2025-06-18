# EUCLID_unda

This repository contains a Python-based 2D viscoelastic FEM solver with structured mesh generation, Prony series modeling, and time-dependent simulation.

## Structure
- `synthetic_simulator/` – This folder contains the FEM solver with modules mesh_generator.py and viscoelastic_fem.py. This module so far uses structured meshing technique and the 2d stress viscoelastic problem formulation
- `lin_visco.py` – Sample runner
- `results/` – (ignored) local results only

## Note
Large data files like `.npz` and `.npy` are excluded. If needed, please contact the author for data access.