import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

class MeshGenerator:
    def __init__(self, width, height, nx, ny):
        self.width = width
        self.height = height
        self.nx = nx
        self.ny = ny
        self.nodes, self.elements = self.generate_structured_mesh()
        self.triangulation = mtri.Triangulation(self.nodes[:, 0], self.nodes[:, 1], self.elements)

    def generate_structured_mesh(self):
        x = np.linspace(0, self.width, self.nx + 1)
        y = np.linspace(0, self.height, self.ny + 1)
        xv, yv = np.meshgrid(x, y)
        nodes = np.column_stack([xv.flatten(), yv.flatten()])

        elements = []
        for j in range(self.ny):
            for i in range(self.nx):
                n1 = j * (self.nx + 1) + i
                n2 = n1 + 1
                n3 = n1 + (self.nx + 1)
                n4 = n3 + 1
                elements.append([n1, n2, n4])
                elements.append([n1, n4, n3])

        return np.array(nodes), np.array(elements)

    def get_boundary_nodes(self, tol=1e-5):
        top_y = np.max(self.nodes[:, 1])
        bottom_nodes = np.where(np.abs(self.nodes[:, 1]) < tol)[0]
        top_nodes = np.where(np.abs(self.nodes[:, 1] - top_y) < tol)[0]
        return top_nodes, bottom_nodes

    def plot_mesh(self):
        plt.figure(figsize=(8, 10))
        plt.triplot(self.triangulation, 'k-', lw=0.5)
        plt.axis('equal')
        plt.title('Structured Triangular Mesh')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.grid(True)
        plt.show()

    def get_data(self):
        return self.nodes, self.elements
