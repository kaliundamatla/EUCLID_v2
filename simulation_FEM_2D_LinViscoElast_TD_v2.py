# 2D FEM simulation of the linear viscoelastic problem with random holes
# The results are visualized at different time steps and will be compared with experimental data

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve
import matplotlib.tri as mtri

#def create_mesh_with_holes(width, height, nx=40, ny=100, hole_centers=None, hole_radii=None):
#    """Create a proper mesh with holes by removing elements that intersect holes"""
#    # Create base grid
#    x = np.linspace(0, width, nx)
#    y = np.linspace(0, height, ny)
#    X, Y = np.meshgrid(x, y)
#    nodes = np.column_stack([X.flatten(), Y.flatten()])
    
    # Create triangulation
#    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1])
#    elements = tri.triangles.copy()
    
    # Remove elements that intersect with holes
#    if hole_centers is not None and hole_radii is not None:
#        valid_elements = []
#        for i, element in enumerate(elements):
            # Get element nodes
#            element_nodes = nodes[element]
            
            # Check element quality first
#            if not check_element_quality(nodes, element):
#                continue
            
            # Check if any node is inside a hole
#            inside_hole = False
#            for (cx, cy), radius in zip(hole_centers, hole_radii):
#                # Calculate element centroid
#                centroid_x = np.mean(element_nodes[:, 0])
#                centroid_y = np.mean(element_nodes[:, 1])
                
                # Calculate distance from centroid to hole center
#                dist = np.sqrt((centroid_x - cx)**2 + (centroid_y - cy)**2)
                
                # If centroid is inside hole, mark element for removal
#                if dist < radius * 1.1:  # Add small buffer
#                    inside_hole = True
#                    break
            
#            if not inside_hole:
#                valid_elements.append(element)
        
#        elements = np.array(valid_elements)
    
#    return nodes, elements

#def check_element_quality(nodes, element):
#    """Check element quality to avoid degenerate elements"""
#    coords = nodes[element]
    # Calculate area
#    area = 0.5 * abs((coords[1,0] - coords[0,0]) * (coords[2,1] - coords[0,1]) - 
#                     (coords[2,0] - coords[0,0]) * (coords[1,1] - coords[0,1]))
    
    # Calculate edge lengths
#    edges = [np.linalg.norm(coords[i] - coords[(i+1)%3]) for i in range(3)]
#    max_edge = max(edges)
#    min_edge = min(edges)
    
    # Reject elements with poor quality
#    return area > 1e-8 and max_edge/min_edge < 20.0

class ViscoelasticFEM:
    def __init__(self, nodes, elements, prony_series, dt=0.01, T=10):
        """Initialize viscoelastic FEM solver"""
        self.nodes = nodes
        self.elements = elements
        self.n_nodes = len(nodes)
        self.n_elements = len(elements)
        self.n_dofs = 2 * self.n_nodes
        
        # Validate and normalize Prony series coefficients
        g_sum = sum(prony_series['g_i'])
        k_sum = sum(prony_series['k_i'])
        
        # Normalize if sum is too large
        if g_sum >= 0.9:
            prony_series['g_i'] = [g * 0.8/g_sum for g in prony_series['g_i']]
            g_sum = sum(prony_series['g_i'])
        if k_sum >= 0.9:
            prony_series['k_i'] = [k * 0.8/k_sum for k in prony_series['k_i']]
            k_sum = sum(prony_series['k_i'])
        
        # Prony series coefficients
        self.g_inf = max(0.1, 1.0 - g_sum)
        self.k_inf = max(0.1, 1.0 - k_sum)
        self.g_i = prony_series['g_i']
        self.k_i = prony_series['k_i']
        self.tau_i = prony_series['tau_i']
        self.n_terms = len(self.g_i)
        
        print(f"Normalized: g_inf = {self.g_inf:.3f}, k_inf = {self.k_inf:.3f}")
        print(f"g_sum = {sum(self.g_i):.3f}, k_sum = {sum(self.k_i):.3f}")
        
        # Material constants
        self.E = 3.3e9 # Pa - reduced modulus
        self.nu = 0.3
        
        # Convert to Lamé parameters
        self.G0 = self.E / (2 * (1 + self.nu))
        self.K0 = self.E / (3 * (1 - 2 * self.nu))
        
        # Time discretization
        self.dt = dt
        self.T = T
        self.n_steps = int(T / dt) + 1
        self.time = np.linspace(0, T, self.n_steps)
        
        # Initialize history variables
        self.e_v = np.zeros((self.n_elements, 3, self.n_terms))
        self.theta_v = np.zeros((self.n_elements, 1, self.n_terms))
        
        # Initialize displacement storage
        self.u = np.zeros((self.n_steps, self.n_dofs))
        
        # Initialize triangulation for plotting
        self.triangulation = mtri.Triangulation(self.nodes[:, 0], self.nodes[:, 1], self.elements)
    
    def assemble_system(self, step):
        """Assemble the global stiffness matrix and force vector"""
        K = np.zeros((self.n_dofs, self.n_dofs))
        F = np.zeros(self.n_dofs)
        
        # Define m vector for plane stress
        m = np.array([1, 1, 0])
        
        # Define D_mu matrix for plane stress
        D_mu = np.diag([2, 2, 1])
        
        # Define projectors
        I_dev = np.eye(3) - 0.5 * np.outer(m, m)
        
        for el in range(self.n_elements):
            # Get element nodes
            nodes_idx = self.elements[el]
            node_coords = self.nodes[nodes_idx]
            
            # Element DOFs
            dofs = np.array([[2*idx, 2*idx+1] for idx in nodes_idx]).flatten()
            
            # Calculate element area and shape function derivatives
            x = node_coords[:, 0]
            y = node_coords[:, 1]
            
            # Area of the triangle
            area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
            
            # Skip degenerate elements
            if area < 1e-10:
                continue
            
            # Shape function derivatives
            b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) / (2 * area)
            c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) / (2 * area)
            
            # B matrix (strain-displacement)
            B = np.zeros((3, 6))
            B[0, 0::2] = b
            B[1, 1::2] = c
            B[2, 0::2] = c
            B[2, 1::2] = b
            
            # Deviatoric B matrix
            B_D = I_dev @ B
            
            # Volumetric operator
            b_vol = m.T @ B
            
            # Time-dependent stiffness components
            G_factor = 0.0
            K_factor = 0.0
            
            for i in range(self.n_terms):
                denominator = 2*self.tau_i[i] + self.dt
                if denominator > 1e-10:
                    factor = (2*self.tau_i[i]) / denominator
                    factor = np.clip(factor, 0.0, 0.99)  # Stability bound
                    
                    G_factor += self.g_i[i] * factor
                    K_factor += self.k_i[i] * factor
            
            G_t = self.G0 * (self.g_inf + G_factor)
            K_t = self.K0 * (self.k_inf + K_factor)
            
            # Ensure positive moduli
            G_t = max(G_t, 0.01 * self.G0)
            K_t = max(K_t, 0.01 * self.K0)
            
            # Element stiffness matrix
            K_el = (G_t * B.T @ D_mu @ B_D + K_t * b_vol.T @ b_vol) * area
            
            # Element history force vector
            F_hist = np.zeros(6)
            if step > 0:
                for i in range(self.n_terms):
                    denominator = 2*self.tau_i[i] + self.dt
                    if denominator < 1e-10:
                        continue
                    
                    alpha = self.dt / denominator
                    if alpha > 0.45:
                        print(f"Step {step}, el {el}, i={i} -> alpha = {alpha:.3f} (clamped)")
                    alpha = min(alpha, 0.5)  # Stability limit
                    
                    # Current strains
                    strain_dev = B_D @ self.u[step-1, dofs]
                    strain_vol = b_vol @ self.u[step-1, dofs]
                    
                    # Update history variables with stability
                    self.e_v[el, :, i] = (1-alpha) * self.e_v[el, :, i] + alpha * strain_dev
                    self.theta_v[el, 0, i] = (1-alpha) * self.theta_v[el, 0, i] + alpha * strain_vol
                    
                    # History force contribution
                    F_hist += (self.G0 * self.g_i[i] * B.T @ D_mu @ self.e_v[el, :, i] + 
                              self.K0 * self.k_i[i] * b_vol.T * self.theta_v[el, 0, i])
            
            # Check for NaN/Inf
            if np.any(np.isnan(K_el)) or np.any(np.isinf(K_el)):
                continue
            if np.any(np.isnan(F_hist)) or np.any(np.isinf(F_hist)):
                F_hist = np.zeros(6)
            
            # Assemble into global system
            for i in range(6):
                F[dofs[i]] += F_hist[i]
                for j in range(6):
                    K[dofs[i], dofs[j]] += K_el[i, j]
        
        return K, F
        
    def visualize_boundary_conditions(self):
        """Debug visualization showing boundary nodes and forces"""
        plt.figure(figsize=(10, 12))
        
        # Plot full mesh
        plt.triplot(self.triangulation, 'k-', lw=0.3, alpha=0.3)
        
        # Find boundary nodes
        bottom_nodes = np.where(np.isclose(self.nodes[:, 1], 0, atol=1e-8))[0]
        top_nodes = np.where(np.isclose(self.nodes[:, 1], np.max(self.nodes[:, 1]), atol=1e-8))[0]
        
        # Highlight boundary nodes
        plt.scatter(self.nodes[bottom_nodes, 0], self.nodes[bottom_nodes, 1], 
                    color='blue', s=30, label='Fixed (bottom)')
        plt.scatter(self.nodes[top_nodes, 0], self.nodes[top_nodes, 1], 
                    color='red', s=30, label='Force applied (top)')
        
        # Show force vectors on top nodes
        force_scale = 1.0  # Adjust for visibility
        for node in top_nodes:
            plt.arrow(self.nodes[node, 0], self.nodes[node, 1], 
                    0, force_scale, color='red', width=0.1, 
                    head_width=0.3, head_length=0.5)
        
        plt.axis('equal')
        plt.title('Boundary Conditions Visualization')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def apply_boundary_conditions(self, K, F, step):
        # --- 1. Identify edge nodes ---
        bottom_nodes = np.where(np.isclose(self.nodes[:, 1], 0.0, atol=1e-8))[0]
        top_y = np.max(self.nodes[:, 1])
        top_nodes = np.where(np.isclose(self.nodes[:, 1], top_y, atol=1e-8))[0]

        # --- 2. Fix bottom edge (both X and Y directions) ---
        for node in bottom_nodes:
            for dof in [2 * node, 2 * node + 1]:
                K[dof, :] = 0
                K[:, dof] = 0
                K[dof, dof] = 1.0
                F[dof] = 0.0

        # --- 3. Sort top nodes by X to define segments ---
        top_coords = self.nodes[top_nodes]
        sorted_indices = np.argsort(top_coords[:, 0])
        sorted_top_nodes = top_nodes[sorted_indices]
        top_coords_sorted = top_coords[sorted_indices]

        # --- 4. Compute traction-based forces on segments ---
        total_force_y = 1e5  # Adjusted load
        force_at_nodes = np.zeros(len(sorted_top_nodes))

        segment_lengths = np.linalg.norm(np.diff(top_coords_sorted, axis=0), axis=1)
        total_length = np.sum(segment_lengths)

        for i in range(len(segment_lengths)):
            L = segment_lengths[i]
            f_segment = total_force_y * (L / total_length)  # Scaled force
            force_at_nodes[i] += 0.5 * f_segment
            force_at_nodes[i + 1] += 0.5 * f_segment

        for idx, node in enumerate(sorted_top_nodes):
            F[2 * node + 1] += force_at_nodes[idx]

        if step == 0:
            print(f"Total vertical force applied: {np.sum(force_at_nodes):.2f} N")

        # --- 5. Fix one top node in X-direction to prevent rigid motion ---
        #rightmost_top_node = sorted_top_nodes[-1]  # Max X node
        #x_dof = 2 * rightmost_top_node
        #K[x_dof, :] = 0
        #K[:, x_dof] = 0
        #K[x_dof, x_dof] = 1.0
        #F[x_dof] = 0.0

        #if step == 0:
        #    print(f"Constraining {len(bottom_nodes)} bottom nodes in X and Y")
        #    print(f"Constraining rightmost top node {rightmost_top_node} in X")

        return K, F


            
    def solve_time_step(self, step):
        """Solve the system for a single time step"""
        try:
            K, F = self.assemble_system(step)
            K, F = self.apply_boundary_conditions(K, F, step)
            
            # Stability checks
            if np.any(np.isnan(K)) or np.any(np.isinf(K)):
                raise ValueError(f"Unstable stiffness matrix at step {step}")
            
            if np.any(np.isnan(F)) or np.any(np.isinf(F)):
                raise ValueError(f"Unstable force vector at step {step}")
            
            # Convert to sparse and add small regularization
            K_sparse = csr_matrix(K)
            regularization = 1e-14 * eye(K_sparse.shape[0])
            K_sparse += regularization
            
            # Solve
            u_solution = spsolve(K_sparse, F)
            
            # Check solution validity
            if np.any(np.isnan(u_solution)) or np.any(np.isinf(u_solution)):
                raise ValueError(f"Invalid solution at step {step}")
            
            # Limit displacement growth for stability
            if step > 0:
                max_disp_change = np.max(np.abs(u_solution - self.u[step-1, :]))
                if max_disp_change > 1.0:  # mm
                    scale_factor = 1.0 / max_disp_change
                    u_solution = self.u[step-1, :] + scale_factor * (u_solution - self.u[step-1, :])
                    print(f"Step {step}: Limited displacement change by factor {scale_factor:.3f}")
                    print(f"Step {step}: Max displacement change = {max_disp_change:.4f}")
            
            self.u[step, :] = u_solution
            
        except Exception as e:
            print(f"Error in step {step}: {e}")
            if step > 0:
                self.u[step, :] = self.u[step-1, :]
            else:
                self.u[step, :] = np.zeros(self.n_dofs)
        
        return self.u[step, :]
    
    def solve(self):
        """Solve the viscoelastic problem for all time steps"""
        for step in range(self.n_steps):
            if step % 20 == 0:
                print(f"Solving step {step+1}/{self.n_steps}")
            self.solve_time_step(step)
        
        return self.u
    
    def plot_mesh(self):
        """Plot the original mesh"""
        plt.figure(figsize=(8, 10))
        plt.triplot(self.triangulation, 'k-', lw=0.5)
        plt.axis('equal')
        plt.title('Original Mesh')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.grid(True)
        plt.show()
    
    def plot_deformation(self, step, scale=10.0):
        """Plot deformation with reasonable scaling"""
        u_step = self.u[step, :]
        
        # Extract displacements
        u_x = u_step[0::2]
        u_y = u_step[1::2]
        
        # Debug info
        max_u_x = np.max(np.abs(u_x))
        max_u_y = np.max(np.abs(u_y))
        print(f"Step {step}: Max |u_x|: {max_u_x:.6f} mm, Max |u_y|: {max_u_y:.6f} mm")
        
        # Adaptive scaling
        max_disp = max(max_u_x, max_u_y)
        #if max_disp > 0:
        #    scale = min(scale, 2.0 / max_disp)  # Limit visual distortion
        
        # Create deformed coordinates
        x_def = self.nodes[:, 0] + scale * u_x
        y_def = self.nodes[:, 1] + scale * u_y
        
        # Create new triangulation for deformed mesh
        try:
            tri_def = mtri.Triangulation(x_def, y_def, self.elements)
        except:
            print(f"Warning: Could not create deformed triangulation at step {step}")
            return
        
        # Create figure
        plt.figure(figsize=(16, 10))
        
        # Plot original mesh
        plt.subplot(1, 2, 1)
        plt.triplot(self.triangulation, 'k-', lw=0.5)
        plt.axis('equal')
        plt.title('Original Mesh')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.grid(True)
        
        # Plot deformed mesh
        plt.subplot(1, 2, 2)
        plt.triplot(tri_def, 'r-', lw=0.5)
        plt.axis('equal')
        plt.title(f'Deformed Mesh at t = {self.time[step]:.2f} s (scale={scale:.1f})')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Plot displacement magnitude
        plt.figure(figsize=(8, 10))
        disp_mag = np.sqrt(u_x**2 + u_y**2)
        if np.max(disp_mag) > 0:
            plt.tripcolor(self.triangulation, disp_mag, shading='gouraud', cmap='viridis')
            plt.colorbar(label='Displacement Magnitude (mm)')
        plt.triplot(self.triangulation, 'k-', lw=0.2, alpha=0.3)
        plt.axis('equal')
        plt.title(f'Displacement Magnitude at t = {self.time[step]:.2f} s')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.grid(True)
        plt.show()
    
    def plot_displacement_history(self):
        """Plot displacement history at top center"""
        # Find top center node
        center_x = 10
        top_y = 50
        distances = np.sqrt((self.nodes[:, 0] - center_x)**2 + (self.nodes[:, 1] - top_y)**2)
        top_center_node = np.argmin(distances)
        
        # Get displacement history
        top_center_y_dof = 2 * top_center_node + 1
        y_displacement = self.u[:, top_center_y_dof]
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.time, y_displacement, 'b-', linewidth=2)
        plt.title('Displacement History at Top Center')
        plt.xlabel('Time (s)')
        plt.ylabel('Y Displacement (mm)')
        plt.grid(True)
        plt.show()
        
        print(f"Final displacement: {y_displacement[-1]:.6f} mm")

def main():
    # Define stable Prony series coefficients
    prony_series = {
        'g_i': [0.1, 0.08806, 0.07907],   # Sum = 0.45 < 1.0
        'k_i': [0.1, 0.08806, 0.07907],   # Sum = 0.45 < 1.0
        'tau_i': [1.847, 19.37, 139.9]   # Reasonable time constants
    }
    
    # Create mesh with holes
    width = 20
    height = 50
    #hole_centers = [(10, 15), (5, 25), (15, 35)]
    #hole_radii = [2.5, 1.5, 2.0]  # Slightly smaller holes
    
#    nodes, elements = create_mesh_with_holes(
    width, height, nx=25, ny=20,
    hole_centers=None,
    hole_radii=None
    )
    
    print(f"Mesh created with {len(nodes)} nodes and {len(elements)} elements")
    
    # Create and solve FEM problem with appropriate parameters
    fem = ViscoelasticFEM(nodes, elements, prony_series, dt=0.5, T=50)  # Smaller time step
    
    # Plot original mesh
    fem.plot_mesh()
    
    # Add visualization of boundary conditions before solving
    fem.visualize_boundary_conditions()

    # Solve
    fem.solve()
    
    # Plot results at different times
    fem.plot_deformation(step=50, scale= 10.0)   # t = 50.0s
    fem.plot_deformation(step=100, scale= 10.0)  # t = 100.0s
    #fem.plot_deformation(step=300, scale= 20.0)  # t = 300.0s
    # fem.plot_deformation(step=600, scale= 1.0)  # t = 600.0s
    # fem.plot_deformation(step=1200, scale= 1.0)  # t = 1200.0s
    fem.plot_displacement_history()

if __name__ == "__main__":
    main()