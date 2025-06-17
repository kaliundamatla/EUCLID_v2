# Generate mesh from DIC coordinates, triangular meshing with hole awareness complete
# Subset meshing is still in development

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from typing import Tuple, Optional, List
import csv

class Mesh2D:
    """2D Mesh Generator for Finite Element Method"""
    
    def __init__(self):
        self.nodes = None
        self.elements = None
        self.boundary_nodes = {}
        
    def load_nodes_from_csv(self, filename: str, x_col: int = 1, y_col: int = 2, 
                           delimiter: str = ';') -> np.ndarray:
        """Load node coordinates from CSV file with DIC data format"""
        coords = []
        
        with open(filename, 'r') as file:
            lines = file.readlines()
            
            # Find header line containing 'id;x;y;z;'
            data_start = 0
            for i, line in enumerate(lines):
                if 'id;x;y;z;' in line:
                    data_start = i + 1
                    break
            
            # Parse data lines
            for line in lines[data_start:]:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split(delimiter)
                try:
                    if len(parts) >= 3:
                        x = float(parts[x_col])
                        y = float(parts[y_col])
                        coords.append([x, y])
                except (ValueError, IndexError):
                    continue
                    
        self.nodes = np.array(coords)
        print(f"Loaded {len(self.nodes)} nodes from {filename}")
        print(f"X range: {self.nodes[:,0].min():.3f} to {self.nodes[:,0].max():.3f}")
        print(f"Y range: {self.nodes[:,1].min():.3f} to {self.nodes[:,1].max():.3f}")
        return self.nodes
    
    def set_nodes(self, coordinates: np.ndarray):
        """Set node coordinates directly"""
        self.nodes = np.array(coordinates)
    
    def subset_nodes(self, max_nodes: int = 300, method: str = 'uniform', 
                    preserve_holes: bool = True) -> np.ndarray:
        """Select subset of nodes for meshing"""
        if len(self.nodes) <= max_nodes:
            return self.nodes
            
        if method == 'uniform':
            # Uniform sampling
            indices = np.linspace(0, len(self.nodes)-1, max_nodes, dtype=int)
            self.nodes = self.nodes[indices]
            
        elif method == 'boundary_preserve':
            # Preserve boundary nodes + uniform interior sampling
            hull_indices = self._get_convex_hull_indices()
            interior_indices = np.setdiff1d(np.arange(len(self.nodes)), hull_indices)
            
            n_interior = max_nodes - len(hull_indices)
            if n_interior > 0 and len(interior_indices) > 0:
                interior_sample = np.random.choice(interior_indices, 
                                                 min(n_interior, len(interior_indices)), 
                                                 replace=False)
                selected = np.concatenate([hull_indices, interior_sample])
            else:
                selected = hull_indices
                
            self.nodes = self.nodes[selected]
            
        return self.nodes
    
    def _get_convex_hull_indices(self) -> np.ndarray:
        """Get indices of nodes on convex hull"""
        from scipy.spatial import ConvexHull
        hull = ConvexHull(self.nodes)
        return hull.vertices
    
    def classify_nodes(self, edge_threshold: float = 2.0) -> dict:
        """Classify nodes as edge, hole, or interior based on local density"""
        from sklearn.neighbors import NearestNeighbors
        
        # Find k nearest neighbors for each node
        k = min(8, len(self.nodes) - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(self.nodes)
        distances, indices = nbrs.kneighbors(self.nodes)
        
        # Calculate local density (average distance to k nearest neighbors)
        local_density = np.mean(distances[:, 1:], axis=1)  # Exclude self (index 0)
        density_threshold = np.median(local_density) * edge_threshold
        
        # Classify nodes
        sparse_nodes = np.where(local_density > density_threshold)[0]
        
        # Further classify sparse nodes
        outer_boundary = self._get_convex_hull_indices()
        
        node_classes = {
            'edge': outer_boundary,
            'hole': np.setdiff1d(sparse_nodes, outer_boundary),
            'interior': np.setdiff1d(np.arange(len(self.nodes)), 
                                   np.union1d(outer_boundary, sparse_nodes))
        }
        
        print(f"Node classification:")
        print(f"  Edge nodes: {len(node_classes['edge'])}")
        print(f"  Hole nodes: {len(node_classes['hole'])}")
        print(f"  Interior nodes: {len(node_classes['interior'])}")
        
        return node_classes
    
    def point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """Check if point is inside polygon using ray casting"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def generate_triangular_mesh(self, respect_holes: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Generate triangular mesh using constrained Delaunay triangulation"""
        if self.nodes is None:
            raise ValueError("Nodes not set. Use load_nodes_from_csv() or set_nodes() first.")
        
        # Standard Delaunay triangulation
        tri = Delaunay(self.nodes)
        
        if not respect_holes:
            # Original behavior
            n_elements = len(tri.simplices)
            self.elements = np.zeros((n_elements, 4), dtype=int)
            self.elements[:, 0] = np.arange(1, n_elements + 1)
            self.elements[:, 1:4] = tri.simplices + 1
            return self.nodes, self.elements
        
        # Filter out elements that cross holes
        valid_elements = []
        element_id = 1
        
        for simplex in tri.simplices:
            # Get triangle vertices
            triangle_nodes = self.nodes[simplex]
            centroid = np.mean(triangle_nodes, axis=0)
            
            # Check if triangle is valid (not crossing holes)
            is_valid = True
            
            # Simple heuristic: check triangle area and edge lengths
            # Large triangles likely span holes
            edges = [
                np.linalg.norm(triangle_nodes[1] - triangle_nodes[0]),
                np.linalg.norm(triangle_nodes[2] - triangle_nodes[1]),
                np.linalg.norm(triangle_nodes[0] - triangle_nodes[2])
            ]
            max_edge = max(edges)
            
            # Calculate average edge length for reference
            if not hasattr(self, '_avg_edge_length'):
                all_edges = []
                for s in tri.simplices[:100]:  # Sample for efficiency
                    tn = self.nodes[s]
                    all_edges.extend([
                        np.linalg.norm(tn[1] - tn[0]),
                        np.linalg.norm(tn[2] - tn[1]),
                        np.linalg.norm(tn[0] - tn[2])
                    ])
                self._avg_edge_length = np.median(all_edges)
            
            # Reject triangles with excessively long edges (likely spanning holes)
            if max_edge > 3 * self._avg_edge_length:
                is_valid = False
            
            if is_valid:
                valid_elements.append([element_id] + (simplex + 1).tolist())
                element_id += 1
        
        self.elements = np.array(valid_elements)
        return self.nodes, self.elements
    
    def generate_quadrilateral_mesh(self, nx: int, ny: int, 
                                   x_range: Tuple[float, float] = (0, 1),
                                   y_range: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, np.ndarray]:
        """Generate structured quadrilateral mesh"""
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        
        # Generate nodes
        nodes = []
        for j in range(ny):
            for i in range(nx):
                nodes.append([x[i], y[j]])
        self.nodes = np.array(nodes)
        
        # Generate quad elements
        elements = []
        elem_id = 1
        for j in range(ny-1):
            for i in range(nx-1):
                n1 = j * nx + i + 1
                n2 = j * nx + i + 2
                n3 = (j + 1) * nx + i + 2
                n4 = (j + 1) * nx + i + 1
                elements.append([elem_id, n1, n2, n3, n4])
                elem_id += 1
                
        self.elements = np.array(elements)
        return self.nodes, self.elements
    
    def identify_boundary_nodes(self, tolerance: float = 1e-6) -> dict:
        """Identify boundary nodes (top, bottom, left, right)"""
        if self.nodes is None:
            raise ValueError("Nodes not set.")
            
        x_coords = self.nodes[:, 0]
        y_coords = self.nodes[:, 1]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        self.boundary_nodes = {
            'left': np.where(np.abs(x_coords - x_min) < tolerance)[0] + 1,
            'right': np.where(np.abs(x_coords - x_max) < tolerance)[0] + 1,
            'bottom': np.where(np.abs(y_coords - y_min) < tolerance)[0] + 1,
            'top': np.where(np.abs(y_coords - y_max) < tolerance)[0] + 1
        }
        
        return self.boundary_nodes
    
    def export_mesh(self, nodes_file: str = 'nodes.txt', 
                   elements_file: str = 'elements.txt'):
        """Export mesh to text files"""
        if self.nodes is None or self.elements is None:
            raise ValueError("Mesh not generated.")
            
        # Export nodes: [node_id, x, y]
        node_data = np.column_stack([np.arange(1, len(self.nodes)+1), self.nodes])
        np.savetxt(nodes_file, node_data, fmt=['%d', '%.6f', '%.6f'], 
                  header='NodeID X Y', comments='')
        
        # Export elements
        np.savetxt(elements_file, self.elements, fmt='%d', 
                  header='ElementID Node1 Node2 Node3 [Node4]', comments='')
        
        print(f"Mesh exported: {nodes_file}, {elements_file}")
    
    def plot_mesh(self, show_node_ids: bool = False, show_element_ids: bool = False,
                 show_classification: bool = True, figsize: Tuple[int, int] = (12, 8)):
        """Visualize the mesh with node classification"""
        if self.nodes is None:
            raise ValueError("Mesh not generated.")
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Left plot: Node classification
        if show_classification:
            node_classes = self.classify_nodes()
            
            # Plot different node types
            if len(node_classes['interior']) > 0:
                interior_coords = self.nodes[node_classes['interior']]
                ax1.plot(interior_coords[:, 0], interior_coords[:, 1], 'o', 
                        color='lightblue', markersize=2, label='Interior nodes')
            
            if len(node_classes['edge']) > 0:
                edge_coords = self.nodes[node_classes['edge']]
                ax1.plot(edge_coords[:, 0], edge_coords[:, 1], 'o', 
                        color='red', markersize=4, label='Edge nodes')
            
            if len(node_classes['hole']) > 0:
                hole_coords = self.nodes[node_classes['hole']]
                ax1.plot(hole_coords[:, 0], hole_coords[:, 1], 'o', 
                        color='green', markersize=4, label='Hole nodes')
        else:
            ax1.plot(self.nodes[:, 0], self.nodes[:, 1], 'o', markersize=2)
        
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_title('Node Classification')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        if show_classification:
            ax1.legend()
        
        # Right plot: Mesh
        if self.elements is not None:
            for elem in self.elements:
                if len(elem) == 4:  # Triangular
                    nodes_in_elem = elem[1:4] - 1
                    triangle = self.nodes[nodes_in_elem]
                    triangle = np.vstack([triangle, triangle[0]])
                    ax2.plot(triangle[:, 0], triangle[:, 1], 'b-', linewidth=0.5)
            
            ax2.plot(self.nodes[:, 0], self.nodes[:, 1], 'ro', markersize=2)
            ax2.set_title(f'Mesh: {len(self.nodes)} nodes, {len(self.elements)} elements')
        else:
            ax2.plot(self.nodes[:, 0], self.nodes[:, 1], 'ro', markersize=2)
            ax2.set_title('Nodes Only')
        
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        plt.tight_layout()
        plt.show()

# Usage Example
if __name__ == "__main__":
    # Initialize mesh generator
    mesh = Mesh2D()
    
    # Example 1: Load from your DIC CSV file
    print("Loading nodes from Creep_Perforated_WJ_3holes2_0001.csv...")
    mesh.load_nodes_from_csv('Creep_Perforated_WJ_3holes2_0001.csv')
    
    # Subset nodes for manageable FEM mesh (300 nodes as requested)
    print("\nSubsetting to the defined number of nodes...")
    mesh.subset_nodes(max_nodes=3500, method='boundary_preserve')
    
    # Generate triangular mesh with hole awareness
    print("\nGenerating hole-aware triangular mesh...")
    nodes, elements = mesh.generate_triangular_mesh(respect_holes=True)
    
    # Identify boundary nodes for FEM boundary conditions
    print("\nIdentifying boundary nodes...")
    boundaries = mesh.identify_boundary_nodes()
    
    # Export mesh files for FEM
    print("\nExporting mesh files...")
    mesh.export_mesh('mesh_nodes.txt', 'mesh_elements.txt')
    
    # Visualize the mesh
    print("\nVisualizing mesh...")
    mesh.plot_mesh(show_node_ids=False, show_element_ids=False)
    
    # Print mesh statistics
    print(f"\n=== MESH STATISTICS ===")
    print(f"Nodes: {len(nodes)}")
    print(f"Elements: {len(elements)}")
    print(f"Boundary nodes:")
    for boundary, node_ids in boundaries.items():
        print(f"  {boundary}: {len(node_ids)} nodes")
    
    # Example 2: Generate sample nodes (alternative approach)
    """
    np.random.seed(42)
    sample_nodes = np.random.rand(100, 2) * 10
    mesh.set_nodes(sample_nodes)
    mesh.subset_nodes(max_nodes=50, method='uniform')
    nodes, elements = mesh.generate_triangular_mesh()
    boundaries = mesh.identify_boundary_nodes()
    mesh.export_mesh('sample_nodes.txt', 'sample_elements.txt')
    mesh.plot_mesh()
    """