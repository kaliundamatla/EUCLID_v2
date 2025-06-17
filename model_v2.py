import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

class TriangularMesh2D:
    def __init__(self, Lx, Ly, nx, ny):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.nodes = None
        self.elements = None
        self.boundary_nodes = {}
        self.element_areas = None
        self.element_centroids = None

        self.generate_nodes()
        self.generate_elements()
        self.identify_boundary_nodes()
        self.compute_element_geometry()

    def generate_nodes(self):
        x = np.linspace(0, self.Lx / 2, self.nx)  # Half-domain in x
        y = np.linspace(0, self.Ly, self.ny)
        xv, yv = np.meshgrid(x, y)
        self.nodes = np.column_stack([xv.ravel(), yv.ravel()])

    def generate_elements(self):
        elements = []
        for i in range(self.ny - 1):
            for j in range(self.nx - 1):
                n1 = i * self.nx + j
                n2 = n1 + 1
                n3 = n1 + self.nx
                n4 = n3 + 1
                if (i + j) % 2 == 0:
                    elements.append([n1, n2, n3])
                    elements.append([n2, n4, n3])
                else:
                    elements.append([n1, n2, n4])
                    elements.append([n1, n4, n3])
        self.elements = np.array(elements)

    def identify_boundary_nodes(self):
        tol = 1e-8
        x, y = self.nodes[:, 0], self.nodes[:, 1]
        self.boundary_nodes['left'] = np.where(np.abs(x - 0) < tol)[0]
        self.boundary_nodes['right'] = np.where(np.abs(x - self.Lx / 2) < tol)[0]
        self.boundary_nodes['bottom'] = np.where(np.abs(y - 0) < tol)[0]
        self.boundary_nodes['top'] = np.where(np.abs(y - self.Ly) < tol)[0]

    def compute_element_geometry(self):
        areas = []
        centroids = []
        for tri in self.elements:
            pts = self.nodes[tri]
            A = 0.5 * np.abs(np.linalg.det(np.array([
                [pts[1, 0] - pts[0, 0], pts[2, 0] - pts[0, 0]],
                [pts[1, 1] - pts[0, 1], pts[2, 1] - pts[0, 1]]
            ])))
            centroid = np.mean(pts, axis=0)
            areas.append(A)
            centroids.append(centroid)
        self.element_areas = np.array(areas)
        self.element_centroids = np.array(centroids)

    def get_element_dofs(self, element_index):
        """
        Returns the global DOF indices for a given triangle (u_x, u_y per node).
        """
        node_indices = self.elements[element_index]
        dofs = []
        for n in node_indices:
            dofs.extend([2 * n, 2 * n + 1])
        return np.array(dofs)

    def get_node_coordinates(self, node_indices):
        return self.nodes[node_indices]

    def plot(self):
        plt.figure(figsize=(6, 9))
        triang = Triangulation(self.nodes[:, 0], self.nodes[:, 1], self.elements)
        plt.triplot(triang, color="gray")
        plt.gca().set_aspect('equal')
        plt.title("Structured Triangular Mesh")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()


# Example usage
if __name__ == "__main__":
    mesh = TriangularMesh2D(Lx=40, Ly=60, nx=11, ny=31)
    mesh.plot()

    print("Total nodes:", len(mesh.nodes))
    print("Total elements:", len(mesh.elements))
    print("Boundary (top) nodes:", mesh.boundary_nodes['top'])

    # Example: Get global DOFs of element 0
    dofs = mesh.get_element_dofs(0)
    print("DOFs of element 0:", dofs)


