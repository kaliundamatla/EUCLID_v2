import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_simulator.mesh_generator import MeshGenerator
from synthetic_simulator.viscoelastic_fem import ViscoelasticFEM  # Assume this is your class defined earlier

# Mesh parameters
width = 20.0
height = 50.0
nx = 20
ny = 50

# Generate mesh
mesh = MeshGenerator(width, height, nx, ny)
nodes, elements = mesh.get_data()
mesh.plot_mesh()

# Define Prony series
prony_series = {
    'g_i': [0.2, 0.2],
    'k_i': [0.1, 0.1],
    'tau_i': [5.0, 20.0]
}

# Initialize FEM solver
fem_solver = ViscoelasticFEM(nodes, elements, prony_series, dt=0.1, T=10.0)

# Visualize mesh and BCs
fem_solver.visualize_boundary_conditions()

# Solve simulation
fem_solver.solve()

# Postprocess
fem_solver.plot_deformation(step=25, scale=10)
fem_solver.plot_deformation(step=100, scale=10)
fem_solver.plot_displacement_history()
