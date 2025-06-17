# Load Data
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import os
import pandas as pd
import concavity
from typing import Optional

# ===== MESH2D CLASS INTEGRATED =====
class Mesh2D:
    def __init__(self):
        self.nodes = None
        self.node_ids = None
        self.elements = None
        self.edge_node_ids = None
        self.hole_node_ids = None

    def set_nodes(self, coordinates: np.ndarray, node_ids: Optional[np.ndarray] = None):
        self.nodes = np.array(coordinates)
        self.node_ids = node_ids if node_ids is not None else np.arange(1, len(coordinates)+1)

    def set_metadata(self, edge_node_ids=None, hole_node_ids=None):
        self.edge_node_ids = edge_node_ids
        self.hole_node_ids = hole_node_ids

    def generate_triangular_mesh(self, respect_holes: bool = True) -> tuple:
        if self.nodes is None:
            raise ValueError("Nodes not set.")

        tri = Delaunay(self.nodes)
        valid_elements = []
        element_id = 1

        for simplex in tri.simplices:
            triangle_nodes = self.nodes[simplex]
            is_valid = True

            edges = [
                np.linalg.norm(triangle_nodes[1] - triangle_nodes[0]),
                np.linalg.norm(triangle_nodes[2] - triangle_nodes[1]),
                np.linalg.norm(triangle_nodes[0] - triangle_nodes[2])
            ]
            max_edge = max(edges)

            if not hasattr(self, '_avg_edge_length'):
                all_edges = []
                for s in tri.simplices[:100]:
                    tn = self.nodes[s]
                    all_edges.extend([
                        np.linalg.norm(tn[1] - tn[0]),
                        np.linalg.norm(tn[2] - tn[1]),
                        np.linalg.norm(tn[0] - tn[2])
                    ])
                self._avg_edge_length = np.median(all_edges)

            if max_edge > 3 * self._avg_edge_length:
                is_valid = False

            if respect_holes and self.hole_node_ids is not None:
                match_count = sum((n + 1) in self.hole_node_ids for n in simplex)
                if match_count == 3:
                    is_valid = False

            if is_valid:
                valid_elements.append([element_id] + (simplex + 1).tolist())
                element_id += 1

        self.elements = np.array(valid_elements)
        return self.nodes, self.elements

    def export_mesh(self, export_dir: str):
        if self.nodes is None or self.elements is None:
            raise ValueError("Mesh not generated.")

        os.makedirs(export_dir, exist_ok=True)
        nodes_file = os.path.join(export_dir, "mesh_nodes.txt")
        elements_file = os.path.join(export_dir, "mesh_elements.txt")

        node_data = np.column_stack([np.arange(1, len(self.nodes)+1), self.nodes])
        np.savetxt(nodes_file, node_data, fmt=['%d', '%.6f', '%.6f'], header='NodeID X Y', comments='')
        np.savetxt(elements_file, self.elements, fmt='%d', header='ElementID Node1 Node2 Node3', comments='')

        print(f"Mesh exported to:\n  {nodes_file}\n  {elements_file}")

    def plot_mesh(self, figsize=(12, 6), save_path=None):
        if self.nodes is None or self.elements is None:
            raise ValueError("Mesh not generated.")

        fig, ax = plt.subplots(figsize=figsize)
        for elem in self.elements:
            pts = self.nodes[elem[1:4] - 1]
            pts = np.vstack([pts, pts[0]])
            ax.plot(pts[:, 0], pts[:, 1], 'b-', linewidth=0.5)

        if self.edge_node_ids is not None:
            all_ids = self.node_ids
            edge_mask = np.isin(all_ids, self.edge_node_ids)
            interior_mask = ~edge_mask
            ax.plot(self.nodes[interior_mask, 0], self.nodes[interior_mask, 1], 'ro', markersize=2, label="Interior Nodes")
            ax.plot(self.nodes[edge_mask, 0], self.nodes[edge_mask, 1], 'go', markersize=3, label="Boundary Nodes")
            ax.legend()
        else:
            ax.plot(self.nodes[:, 0], self.nodes[:, 1], 'ro', markersize=2)

        ax.set_title(f'Triangular Mesh: {len(self.nodes)} nodes, {len(self.elements)} elements')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"Mesh plot saved to {save_path}")
        plt.show()


# ===== EDGE + HOLE DETECTION FUNCTIONS (original) =====
def edge_hole_detection_from_arrays(node_ids, x, y, min_hole_size=0.5, max_hole_size=20.0, combine_output=False, output_dir='results'):
    """
    Detect edge and hole nodes from arrays of node IDs and coordinates.

    Parameters:
        node_ids (array): Array of node IDs.
        x (array): X-coordinates of nodes.
        y (array): Y-coordinates of nodes.
        min_hole_size (float): Minimum hole size for detection.
        max_hole_size (float): Maximum hole size for detection.
        combine_output (bool): Whether to combine edge and hole nodes in one CSV file.
        output_dir (str): Directory to save results.

    Returns:
        tuple: Arrays of edge node IDs and hole node IDs.
    """
    points = np.column_stack((x, y))

    # Get boundary nodes
    edge_node_ids = get_boundary_nodes_concave(points, node_ids, k=5)  # Adjust k as needed

    # Get hole nodes
    hole_node_ids = find_holes(points, node_ids, min_hole_size, max_hole_size)

    # Remove boundary nodes from hole nodes
    hole_node_ids = np.setdiff1d(hole_node_ids, edge_node_ids)

    # Save results
    save_results(output_dir, edge_node_ids, hole_node_ids, combine_output)

    # Visualize results
    visualize_results(x, y, node_ids, edge_node_ids, hole_node_ids, output_dir)

    print(f"Found {len(edge_node_ids)} edge nodes and {len(hole_node_ids)} hole nodes")

    return edge_node_ids, hole_node_ids


def get_boundary_nodes_concave(points, node_ids, k=5):
    """
    Get boundary nodes using a concave hull.

    Parameters:
        points (array): Array of node coordinates.
        node_ids (array): Array of node IDs.
        k (int): Parameter for concave hull generation.

    Returns:
        array: Array of boundary node IDs.
    """
    ch = concavity.concave_hull(points, k)
    boundary_coords = np.array(ch.exterior.coords)
    boundary_node_ids = []

    for bx, by in boundary_coords:
        distances = np.linalg.norm(points - np.array([bx, by]), axis=1)
        closest_idx = np.argmin(distances)
        boundary_node_ids.append(node_ids[closest_idx])

    return np.unique(boundary_node_ids)


def find_holes(points, node_ids, min_hole_size=0.5, max_hole_size=20.0):
    """
    Detect holes based on circumcircle radius of Delaunay triangles.

    Parameters:
        points (array): Array of node coordinates.
        node_ids (array): Array of node IDs.
        min_hole_size (float): Minimum hole size for detection.
        max_hole_size (float): Maximum hole size for detection.

    Returns:
        array: Array of hole node IDs.
    """
    tri = Delaunay(points)
    hole_candidates = []

    for simplex in tri.simplices:
        vertices = points[simplex]
        a, b, c = np.linalg.norm(vertices[1] - vertices[0]), np.linalg.norm(vertices[2] - vertices[1]), np.linalg.norm(vertices[0] - vertices[2])
        s = (a + b + c) / 2
        area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))

        if area > 0:
            circum_r = (a * b * c) / (4 * area)
            if min_hole_size <= circum_r <= max_hole_size:
                hole_candidates.extend(simplex)

    hole_indices = np.unique(hole_candidates)
    return node_ids[hole_indices.astype(int)]


def save_results(output_dir, edge_node_ids, hole_node_ids, combine_output):
    """
    Save edge and hole node results to CSV files.

    Parameters:
        output_dir (str): Directory to save results.
        edge_node_ids (array): Array of edge node IDs.
        hole_node_ids (array): Array of hole node IDs.
        combine_output (bool): Whether to combine edge and hole nodes in one CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)

    if combine_output:
        combined_ids = np.concatenate([edge_node_ids, hole_node_ids])
        pd.DataFrame({'node_id': combined_ids}).to_csv(os.path.join(output_dir, 'combined_nodes.csv'), index=False)
    else:
        pd.DataFrame({'node_id': edge_node_ids}).to_csv(os.path.join(output_dir, 'edge_nodes.csv'), index=False)
        pd.DataFrame({'node_id': hole_node_ids}).to_csv(os.path.join(output_dir, 'hole_nodes.csv'), index=False)


def visualize_results(x, y, node_ids, edge_node_ids, hole_node_ids, output_dir):
    """
    Visualize edge and hole nodes.

    Parameters:
        x (array): X-coordinates of nodes.
        y (array): Y-coordinates of nodes.
        node_ids (array): Array of node IDs.
        edge_node_ids (array): Array of edge node IDs.
        hole_node_ids (array): Array of hole node IDs.
        output_dir (str): Directory to save visualization.
    """
    plt.figure(figsize=(12, 10))
    plt.scatter(x, y, s=1, color='lightblue', label='All Nodes')

    edge_mask = np.isin(node_ids, edge_node_ids)
    plt.scatter(x[edge_mask], y[edge_mask], s=8, color='red', label='Edge Nodes')

    hole_mask = np.isin(node_ids, hole_node_ids)
    plt.scatter(x[hole_mask], y[hole_mask], s=8, color='green', label='Hole Nodes')

    plt.title('Edge and Hole Detection')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'node_detection.png'), dpi=300)
    plt.show()


# ===== MAIN ENTRY POINT =====
def main():
    try:
        experiment_number = input("Please enter the experiment number: ")
        experiment = int(experiment_number)
        base_path = f"results_preprocessing/experiment_{experiment}/"

        try:
            force_data = np.load(base_path + "force_data.npy")
            print("First three rows of force_data:")
            print(force_data[:3])
        except Exception as e:
            print(f"⚠️ Could not load force_data.npy — {e}")

        try:
            nodes_data = np.load(base_path + "nodes_data.npy", allow_pickle=True)
            print("First three rows of nodes_data:")
            print(nodes_data[:3])
        except Exception as e:
            print(f"❌ Could not load nodes_data.npy — {e}")

        try:
            time_data = np.load(base_path + "time_data.npy")
            print("First three rows of time_data:")
            print(time_data[:3])
        except Exception as e:
            print(f"❌ Could not load time_data.npy — {e}")


        nodes_data_frame0 = nodes_data[0]
        node_ids = nodes_data_frame0[:, 0].astype(int)
        x = nodes_data_frame0[:, 1].astype(float)
        y = nodes_data_frame0[:, 2].astype(float)

        min_hole_size = float(input("Enter minimum hole size (default 0.5): ") or 0.5)
        max_hole_size = float(input("Enter maximum hole size (default 20.0): ") or 20.0)
        combine_output = input("Combine edge and hole nodes in one CSV file? (y/n, default n): ").lower() == 'y'

        edge_nodes, hole_nodes = edge_hole_detection_from_arrays(
            node_ids, x, y, min_hole_size, max_hole_size, combine_output, output_dir=base_path
        )

        print("\n--- Proceeding to Mesh Generation ---")
        coordinates = np.column_stack((x, y))

        mesh = Mesh2D()
        mesh.set_nodes(coordinates, node_ids=node_ids)

        print("Generating triangular mesh...")
        nodes, elements = mesh.generate_triangular_mesh(
            respect_holes=True,
            hole_node_ids=hole_nodes
        )

        print("Identifying boundary nodes...")
        boundaries = mesh.identify_boundary_nodes()

        export_path = os.path.join("results", "mesh_exports", f"experiment_{experiment}")
        os.makedirs(export_path, exist_ok=True)

        print("Exporting mesh files...")
        mesh.export_mesh(export_dir=export_path)

        print("Plotting mesh...")
        mesh_plot_path = os.path.join("results", "mesh_exports", f"experiment_{experiment_number}_mesh_plot.png")
        all_boundary_ids = np.unique(np.concatenate(list(boundaries.values())))
        mesh.plot_mesh(
            save_path=mesh_plot_path,
            boundary_node_ids=all_boundary_ids
        )

        print("\n=== MESH SUMMARY ===")
        print(f"Nodes: {len(nodes)}")
        print(f"Elements: {len(elements)}")
        for side, ids in boundaries.items():
            print(f"{side.capitalize()} boundary: {len(ids)} nodes")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()