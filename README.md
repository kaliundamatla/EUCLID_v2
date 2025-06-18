# EUCLID_v2: 2D Viscoelastic FEM Framework

This repository provides a modular Python implementation of a 2D finite element solver with time-dependent linear viscoelastic material behavior. It includes structured mesh generation, Prony series-based viscoelastic modeling, and simulation execution tools.

---

## 📁 Repository Structure

```
📁 EUCLID_v2/
|   📁 concavity/               # Module implemnts convex hull method 
|   |   📄 __init__.py   
|   |   📄 core.py
|   |   📄 utils.py         
│   📁 synthetic_simulator/     # Modularized version with stuctured mesh and stress formulation
│   │   📄 __init__.py                 
│   │   📄 fem_solver.py        # main file to run the simulator
│   │   📄 mesh_generator.py    # Generates the mesh    
│   │   📄 viscoelastic_fem.py  # Contains the 2d stress formulation and FEM 
│   📄 boundary_nodes.py               # old test file for boundary identification
│   📄 boundary_nodes_final_clean.py   # Refined and modular boundary node utility 
│   📄 fem_2d_sttress_sim.py           # Legacy static FEM solver using stress-based formulation
│   📄 lin_visco.py                    # Prototype for linear viscoelastic solver
│   📄 mesh_generator_v2.py            # Experimental mesh generation version
│   📄 model.py                        # Test file
│   📄 model_v2.py                     # Test file
│   📄 Preprocess.py                   # pre-processing for experimental data
│   📄 simulation_FEM_2D_LinViscoElast_TD_v2.py # Main simulation script with unstructured mesh
│   📄 main_FEM_2D_LinViscoElast_TD_28_05.m     # MATLAB version for legacy comparison
│   📄 .gitignore                      # Git ignore rules for results/, cache, and large data
│   📄 README.md                       # Project documentation (this file)
```

---

## 🔍 Description

## 🔍 Description

- This repository contains all core scripts and raw data necessary for the development of the EUCLID framework, which aims to characterize the viscoelastic behavior of materials.
- Preprocessing of experimental data is handled by `Preprocess.py`, while boundary node identification is implemented in `boundary_nodes_final_clean.py` using a convex hull approach (via the `concavity` module).
- For mesh generation, a Delaunay triangulation strategy is applied to complex geometries, enabling accurate discretization.
- The `synthetic_simulator` module uses trial synthetic data to validate and test the functionality of the forward 2D linear viscoelastic stress formulation.


---

## 🚫 Data Notice

Result files and large data files like `.npz`, `.npy`, and entire `results/` folders are ignored from version control. If needed, these can be shared separately via cloud storage.

---

## 📬 Contact

Kali Satya Sri Charan Undamatla  
📧 `undamatl@iwm.fraunhofer.de`  
🔗 GitHub: [kaliundamatla](https://github.com/kaliundamatla)
