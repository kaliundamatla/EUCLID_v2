import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def generate_structured_mesh(width, height, nx, ny):
    x = np.linspace(0, width, nx + 1)
    y = np.linspace(0, height, ny + 1)
    xv, yv = np.meshgrid(x, y)
    nodes = np.column_stack([xv.flatten(), yv.flatten()])

    elements = []
    for j in range(ny):
        for i in range(nx):
            n1 = j * (nx + 1) + i
            n2 = n1 + 1
            n3 = n1 + (nx + 1)
            n4 = n3 + 1
            elements.append([n1, n2, n4])
            elements.append([n1, n4, n3])

    return np.array(nodes), np.array(elements)


def get_material_matrix(E, nu):
    coeff = E / (1 - nu ** 2)
    C = coeff * np.array([
        [1,     nu,         0],
        [nu,    1,          0],
        [0,     0,  (1 - nu) / 2]
    ])
    return C


def compute_triangle_stiffness(xy, C):
    x1, y1 = xy[0]
    x2, y2 = xy[1]
    x3, y3 = xy[2]

    A = 0.5 * np.linalg.det(np.array([
        [1, x1, y1],
        [1, x2, y2],
        [1, x3, y3]
    ]))

    beta = np.array([y2 - y3, y3 - y1, y1 - y2])
    gamma = np.array([x3 - x2, x1 - x3, x2 - x1])

    B = np.zeros((3, 6))
    for i in range(3):
        B[0, 2 * i]     = beta[i]
        B[1, 2 * i + 1] = gamma[i]
        B[2, 2 * i]     = gamma[i]
        B[2, 2 * i + 1] = beta[i]
    B /= (2 * A)

    Ke = A * (B.T @ C @ B)
    return Ke, A, B


def assemble_global_stiffness(nodes, elements, C):
    ndof = nodes.shape[0] * 2
    K = np.zeros((ndof, ndof))
    A_list, B_list = [], []

    for elem in elements:
        xy = nodes[elem]
        Ke, A, B = compute_triangle_stiffness(xy, C)

        dof_map = []
        for n in elem:
            dof_map.extend([2 * n, 2 * n + 1])

        for i in range(6):
            for j in range(6):
                K[dof_map[i], dof_map[j]] += Ke[i, j]

        A_list.append(A)
        B_list.append(B)

    return K, A_list, B_list


def apply_boundary_conditions(K, F, fixed_dofs):
    for dof in fixed_dofs:
        K[dof, :] = 0
        K[:, dof] = 0
        K[dof, dof] = 1
        F[dof] = 0
    return K, F


def get_top_and_bottom_nodes(nodes, height, tol=1e-5):
    top_nodes = [i for i, n in enumerate(nodes) if abs(n[1] - height) < tol]
    bottom_nodes = [i for i, n in enumerate(nodes) if abs(n[1]) < tol]
    return top_nodes, bottom_nodes


def apply_uniform_force(nodes, top_nodes, total_force):
    top_coords = nodes[top_nodes]
    sorted_top = [n for _, n in sorted(zip(top_coords[:, 0], top_nodes))]
    top_coords = nodes[sorted_top]

    segment_lengths = np.linalg.norm(np.diff(top_coords, axis=0), axis=1)
    segment_forces = total_force * segment_lengths / np.sum(segment_lengths)

    F = np.zeros(nodes.shape[0] * 2)
    for i in range(len(segment_lengths)):
        ni = sorted_top[i]
        nj = sorted_top[i + 1]
        fy = segment_forces[i] / 2.0
        F[2 * ni + 1] += fy
        F[2 * nj + 1] += fy
    return F


def solve_viscoelastic_prony(K, F, n_steps, dt, g_inf, g_list, tau_list):
    ndof = K.shape[0]
    U = np.zeros((n_steps + 1, ndof))
    q = [np.zeros(ndof) for _ in g_list]

    K_eff = g_inf * K
    for t in range(1, n_steps + 1):
        history = np.zeros(ndof)
        for i, (g, tau) in enumerate(zip(g_list, tau_list)):
            decay = np.exp(-dt / tau)
            q[i] = decay * q[i] + g * (1 - decay) * K @ (U[t - 1] - U[t - 2] if t > 1 else U[t - 1])
            history += q[i]

        rhs = F - history
        U[t] = np.linalg.solve(K_eff, rhs)
    return U


def compute_strain_stress(U, nodes, elements, C, A_list, B_list, step):
    u = U[step]
    strains = []
    stresses = []
    for elem, A, B in zip(elements, A_list, B_list):
        dof_map = []
        for n in elem:
            dof_map.extend([2 * n, 2 * n + 1])
        u_elem = u[dof_map]
        strain = B @ u_elem
        stress = C @ strain
        strains.append(strain)
        stresses.append(stress)
    return np.array(strains), np.array(stresses)


def postprocess_displacement(U, nodes, elements, step, scale=0.1):
    disp = U[step].reshape(-1, 2)
    deformed_nodes = nodes + scale * disp

    fig, axs = plt.subplots(1, 2, figsize=(14, 8))

    axs[0].triplot(nodes[:, 0], nodes[:, 1], elements, color='black', linewidth=0.5)
    axs[0].set_title("Original Mesh")
    axs[0].set_aspect('equal')
    axs[0].set_xlabel("X (mm)")
    axs[0].set_ylabel("Y (mm)")

    axs[1].triplot(deformed_nodes[:, 0], deformed_nodes[:, 1], elements, color='red', linewidth=0.5)
    axs[1].set_title(f"Deformed Mesh at t = {step * 1.0:.2f} s (scale={scale})")
    axs[1].set_aspect('equal')
    axs[1].set_xlabel("X (mm)")
    axs[1].set_ylabel("Y (mm)")

    plt.tight_layout()
    plt.show()


def postprocess_stress(stresses):
    von_mises = np.sqrt(stresses[:, 0]**2 - stresses[:, 0]*stresses[:, 1] + stresses[:, 1]**2 + 3*stresses[:, 2]**2)
    plt.figure()
    plt.plot(von_mises)
    plt.title("Von Mises Stress Over Elements")
    plt.xlabel("Element Index")
    plt.ylabel("Stress [Pa]")
    plt.grid(True)
    plt.show()


def plot_displacement_magnitude(U, nodes, elements, step, title=None):
    u = U[step]
    u_x = u[0::2]
    u_y = u[1::2]
    disp_mag = np.sqrt(u_x**2 + u_y**2)

    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    plt.figure(figsize=(10, 6))
    plt.tripcolor(triang, disp_mag, shading='gouraud', cmap='viridis')
    plt.colorbar(label='Displacement Magnitude (mm)')
    plt.triplot(triang, 'k-', lw=0.2, alpha=0.3)
    plt.axis('equal')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title(title or f"Displacement Magnitude at t = {step:.2f} s")
    plt.grid(True)
    plt.show()


def plot_original_and_deformed_mesh(nodes, elements, U, step, scale=0.1):
    u = U[step].reshape(-1, 2)
    deformed_nodes = nodes + scale * u

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].triplot(nodes[:, 0], nodes[:, 1], elements, color='black', linewidth=0.5)
    axs[0].set_title("Original Mesh")
    axs[0].set_aspect('equal')
    axs[0].set_xlabel("X (mm)")
    axs[0].set_ylabel("Y (mm)")

    axs[1].triplot(deformed_nodes[:, 0], deformed_nodes[:, 1], elements, color='red', linewidth=0.5)
    axs[1].set_title(f"Deformed Mesh at t = {step:.2f} s (scale={scale})")
    axs[1].set_aspect('equal')
    axs[1].set_xlabel("X (mm)")
    axs[1].set_ylabel("Y (mm)")

    plt.tight_layout()
    plt.show()


def plot_displacement_history(U, nodes, height):
    center_x = np.mean(nodes[:, 0])
    top_y = np.max(nodes[:, 1])
    distances = np.sqrt((nodes[:, 0] - center_x)**2 + (nodes[:, 1] - top_y)**2)
    top_center_node = np.argmin(distances)

    dof_y = 2 * top_center_node + 1
    y_displacements = U[:, dof_y]

    time = np.arange(U.shape[0])  # assumes 1 step = 1 sec unless scaled

    plt.figure(figsize=(10, 5))
    plt.plot(time, y_displacements, 'b-', linewidth=2)
    plt.title("Displacement History at Top Center")
    plt.xlabel("Time (s)")
    plt.ylabel("Y Displacement (mm)")
    plt.grid(True)
    plt.show()

    print(f"Final displacement: {y_displacements[-1]:.5f} mm")


# -------------------------------
#            MAIN
# -------------------------------
if __name__ == '__main__':
    width, height = 20.0, 50.0
    nx, ny = 20, 50
    nodes, elements = generate_structured_mesh(width, height, nx, ny)
    print(f"Number of nodes: {nodes.shape[0]}")
    print(f"Number of elements: {elements.shape[0]}")

    E = 3.3e9
    nu = 0.3
    g_inf = 0.6
    g_list = [0.2, 0.2]
    tau_list = [5.0, 20.0]
    dt = 1.0
    n_steps = 100

    C = get_material_matrix(E, nu)
    K, A_list, B_list = assemble_global_stiffness(nodes, elements, C)

    top_nodes, bottom_nodes = get_top_and_bottom_nodes(nodes, height)
    F = apply_uniform_force(nodes, top_nodes, total_force=1e5)

    fixed_dofs = [2 * n + d for n in bottom_nodes for d in [0, 1]]
    fixed_dofs += [2 * n for n in top_nodes]

    K_bc, F_bc = apply_boundary_conditions(K.copy(), F.copy(), fixed_dofs)
    U = solve_viscoelastic_prony(K_bc, F_bc, n_steps, dt, g_inf, g_list, tau_list)

    postprocess_displacement(U, nodes, elements, step=50, scale=10)
    strains, stresses = compute_strain_stress(U, nodes, elements, C, A_list, B_list, step=50)
    postprocess_stress(stresses)

    plot_original_and_deformed_mesh(nodes, elements, U, step=25, scale=0.1)
    plot_original_and_deformed_mesh(nodes, elements, U, step=50, scale=0.1)

    plot_displacement_magnitude(U, nodes, elements, step=25)
    plot_displacement_magnitude(U, nodes, elements, step=50)

    plot_displacement_history(U, nodes, height=50.0)
