import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from collections import defaultdict

class ViscoelasticFEM:
    def __init__(self, nodes, elements, prony_series, dt=0.01, T=1.0):
        self.nodes = nodes
        self.elements = elements
        self.n_nodes = len(nodes)
        self.n_elements = len(elements)
        self.n_dofs = 2 * self.n_nodes

        g_sum = sum(prony_series['g_i'])
        k_sum = sum(prony_series['k_i'])
        if g_sum >= 0.9:
            prony_series['g_i'] = [g * 0.8 / g_sum for g in prony_series['g_i']]
        if k_sum >= 0.9:
            prony_series['k_i'] = [k * 0.8 / k_sum for k in prony_series['k_i']]

        self.g_inf = max(0.1, 1.0 - sum(prony_series['g_i']))
        self.k_inf = max(0.1, 1.0 - sum(prony_series['k_i']))
        self.g_i = prony_series['g_i']
        self.k_i = prony_series['k_i']
        self.tau_i = prony_series['tau_i']
        self.n_terms = len(self.g_i)

        self.E = 3.3e9 / 1000  # MPa
        self.nu = 0.3
        self.G0 = self.E / (2 * (1 + self.nu))
        self.K0 = self.E / (3 * (1 - 2 * self.nu))

        self.dt = dt
        self.T = T
        self.n_steps = int(T / dt) + 1
        self.time = np.linspace(0, T, self.n_steps)

        self.e_v = np.zeros((self.n_elements, 3, self.n_terms))
        self.theta_v = np.zeros((self.n_elements, 1, self.n_terms))
        self.u = np.zeros((self.n_steps, self.n_dofs))
        self.triangulation = mtri.Triangulation(self.nodes[:, 0], self.nodes[:, 1], self.elements)

    def assemble_system(self, step):
        K = np.zeros((self.n_dofs, self.n_dofs))
        F = np.zeros(self.n_dofs)

        m = np.array([1, 1, 0])
        D_mu = np.diag([2, 2, 1])
        I_dev = np.eye(3) - 0.5 * np.outer(m, m)

        for el in range(self.n_elements):
            nodes_idx = self.elements[el]
            node_coords = self.nodes[nodes_idx]
            dofs = np.array([[2 * idx, 2 * idx + 1] for idx in nodes_idx]).flatten()

            x = node_coords[:, 0]
            y = node_coords[:, 1]
            area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
            if area < 1e-10:
                continue

            b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) / (2 * area)
            c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) / (2 * area)

            B = np.zeros((3, 6))
            B[0, 0::2] = b
            B[1, 1::2] = c
            B[2, 0::2] = c
            B[2, 1::2] = b

            B_D = I_dev @ B
            b_vol = m.T @ B

            G_factor = 0.0
            K_factor = 0.0

            for i in range(self.n_terms):
                denominator = 2 * self.tau_i[i] + self.dt
                if denominator > 1e-10:
                    factor = (2 * self.tau_i[i]) / denominator
                    G_factor += self.g_i[i] * factor
                    K_factor += self.k_i[i] * factor

            G_t = max(self.G0 * (self.g_inf + G_factor), 0.01 * self.G0)
            K_t = max(self.K0 * (self.k_inf + K_factor), 0.01 * self.K0)

            K_el = (G_t * B.T @ D_mu @ B_D + K_t * b_vol.T @ b_vol) * area

            F_hist = np.zeros(6)
            if step > 0:
                for i in range(self.n_terms):
                    denominator = 2 * self.tau_i[i] + self.dt
                    if denominator < 1e-10:
                        continue
                    alpha = self.dt / denominator
                    strain_dev = B_D @ self.u[step - 1, dofs]
                    strain_vol = b_vol @ self.u[step - 1, dofs]
                    self.e_v[el, :, i] = (1 - alpha) * self.e_v[el, :, i] + alpha * strain_dev
                    self.theta_v[el, 0, i] = (1 - alpha) * self.theta_v[el, 0, i] + alpha * strain_vol
                    F_hist += (self.G0 * self.g_i[i] * B.T @ D_mu @ self.e_v[el, :, i] +
                               self.K0 * self.k_i[i] * b_vol.T * self.theta_v[el, 0, i])

            for i in range(6):
                F[dofs[i]] += F_hist[i]
                for j in range(6):
                    K[dofs[i], dofs[j]] += K_el[i, j]

        return K, F


def ensure_ccw_orientation(nodes, elements):
    def triangle_area(a, b, c):
        return 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
    fixed = []
    for elem in elements:
        a, b, c = nodes[elem]
        if triangle_area(a, b, c) < 0:
            fixed.append([elem[0], elem[2], elem[1]])
        else:
            fixed.append(elem)
    return np.array(fixed)


def generate_mesh(width, length, nx, ny):
    x = np.linspace(0, width, nx + 1)
    y = np.linspace(0, length, ny + 1)
    X, Y = np.meshgrid(x, y)
    nodes = np.vstack([X.ravel(), Y.ravel()]).T
    elements = []
    for j in range(ny):
        for i in range(nx):
            n1 = j * (nx + 1) + i
            n2 = n1 + 1
            n3 = n1 + (nx + 1)
            n4 = n3 + 1
            if (i + j) % 2 == 0:
                elements.append([n1, n2, n4])
                elements.append([n1, n4, n3])
            else:
                elements.append([n1, n2, n3])
                elements.append([n2, n4, n3])

    return nodes, np.array(elements)


def apply_boundary_conditions(fem_solver, K, F, total_force=20000):
    nodes = fem_solver.nodes
    dofs_to_fix = []

    # === 1. Pin-and-Roller BC ===
    bottom_nodes = np.where(np.isclose(nodes[:, 1], 0.0))[0]
    left_bot = bottom_nodes[np.argmin(nodes[bottom_nodes, 0])]
    right_bot = bottom_nodes[np.argmax(nodes[bottom_nodes, 0])]
    dofs_to_fix.extend([2 * left_bot, 2 * left_bot + 1])  # Pin
    dofs_to_fix.append(2 * right_bot + 1)                 # Roller

    # Constrain one top center node in X to eliminate horizontal rotation drift
    top_nodes = np.where(np.isclose(nodes[:, 1], np.max(nodes[:, 1])))[0]
    center_top = top_nodes[np.argmin(np.abs(nodes[top_nodes, 0] - np.mean(nodes[:, 0])))]
    dofs_to_fix.append(2 * center_top)  # X DOF only

    # === 2. Apply Traction on Top Edges ===
    top_y = np.max(nodes[:, 1])
    edge_force = defaultdict(float)
    tol = 1e-8
    total_width = np.ptp(nodes[:, 0])

    for elem in fem_solver.elements:
        for i in range(3):
            j = (i + 1) % 3
            n1, n2 = elem[i], elem[j]
            y1, y2 = nodes[n1, 1], nodes[n2, 1]
            if abs(y1 - top_y) < tol and abs(y2 - top_y) < tol:
                p1, p2 = nodes[n1], nodes[n2]
                edge_length = np.linalg.norm(p2 - p1)
                f_edge = total_force * edge_length / total_width
                edge_force[n1] += 0.5 * f_edge
                edge_force[n2] += 0.5 * f_edge

    for n, f in edge_force.items():
        F[2 * n + 1] += f

    # === 3. Apply Dirichlet DOFs ===
    for dof in dofs_to_fix:
        K[dof, :] = 0
        K[:, dof] = 0
        K[dof, dof] = 1
        F[dof] = 0

    return K, F


def time_step_solver_sparse(fem_solver, total_force=20000):
    for step in range(1, fem_solver.n_steps):
        K, F = fem_solver.assemble_system(step)
        K_bc, F_bc = apply_boundary_conditions(fem_solver, K, F, total_force=total_force)
        u_step = spla.spsolve(sp.csr_matrix(K_bc), F_bc)
        fem_solver.u[step, :] = u_step
    return fem_solver.u


def plot_displacement(fem_solver, step=-1):
    u = fem_solver.u[step]
    scale = 10
    x_def = fem_solver.nodes[:, 0] + scale * u[0::2]
    y_def = fem_solver.nodes[:, 1] + scale * u[1::2]
    plt.figure(figsize=(6, 12))
    plt.triplot(x_def, y_def, fem_solver.elements, color='blue', alpha=0.7, label='Deformed')
    plt.triplot(fem_solver.nodes[:, 0], fem_solver.nodes[:, 1], fem_solver.elements, color='gray', alpha=0.3, label='Original')
    plt.title(f"Deformation at step {step} (scaled x{scale})")
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    width = 20.0
    length = 50.0
    nx, ny = 10, 25
    nodes, elements = generate_mesh(width, length, nx, ny)
    elements = ensure_ccw_orientation(nodes, elements)

    prony_series = {
        'g_i': [0.2, 0.1],
        'k_i': [0.2, 0.1],
        'tau_i': [0.5, 1.0]
    }

    fem_solver = ViscoelasticFEM(nodes, elements, prony_series, dt=0.01, T=1.0)
    u_history = time_step_solver_sparse(fem_solver, total_force=1e6)
    plot_displacement(fem_solver, step=-1)
    print("Max vertical displacement (mm):", np.max(u_history[-1, 1::2]))


if __name__ == "__main__":
    main()
