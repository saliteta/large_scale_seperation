import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from plyfile import PlyData, PlyElement



def create_plane_grid(plane_point, u, v, projected_frustums, grid_resolution):
    # Collect all intersection points to determine the grid extent
    all_points = np.vstack(projected_frustums)
    coords_u = np.dot(all_points - plane_point, u)
    coords_v = np.dot(all_points - plane_point, v)
    u_min, u_max = coords_u.min(), coords_u.max()
    v_min, v_max = coords_v.min(), coords_v.max()

    # Create the grid
    u_grid = np.arange(u_min, u_max + grid_resolution, grid_resolution)
    v_grid = np.arange(v_min, v_max + grid_resolution, grid_resolution)
    uu, vv = np.meshgrid(u_grid, v_grid)
    grid_shape = uu.shape
    grid_points = plane_point + uu[..., np.newaxis] * u + vv[..., np.newaxis] * v

    return grid_points.reshape(-1, 3), grid_shape, u_grid, v_grid

def create_grid_points(plane_point, u, v, u_grid, v_grid):
    uu, vv = np.meshgrid(u_grid, v_grid)
    grid_points = plane_point + uu[..., np.newaxis] * u + vv[..., np.newaxis] * v
    return grid_points.reshape(-1, 3)

def compute_coverage(grid_points, grid_shape, plane_point, u, v, projected_frustums):
    # Convert grid points to 2D coordinates in the plane
    coords_u = np.dot(grid_points - plane_point, u)
    coords_v = np.dot(grid_points - plane_point, v)
    grid_coords_2d = np.column_stack((coords_u, coords_v))

    # Initialize coverage counts
    coverage_counts = np.zeros(len(grid_points), dtype=int)

    # For each camera frustum polygon
    for frustum in projected_frustums:
        # Project frustum points onto the plane axes to get 2D polygon
        frustum_coords_u = np.dot(frustum - plane_point, u)
        frustum_coords_v = np.dot(frustum - plane_point, v)
        polygon_coords = np.column_stack((frustum_coords_u, frustum_coords_v))
        path = Path(polygon_coords)

        # Vectorized point-in-polygon test
        inside = path.contains_points(grid_coords_2d)
        coverage_counts += inside.astype(int)

    return coverage_counts

def create_heatmap_animation(heatmaps, u_grid, v_grid, animation_location):
    fig, ax = plt.subplots(figsize=(10, 8))
    def update(frame):
        ax.clear()
        heatmap = heatmaps[frame]
        im = ax.pcolormesh(u_grid, v_grid, heatmap, shading='auto', cmap='hot')
        ax.set_title(f'Camera Coverage Heatmap at Step {frame+1}')
        ax.set_xlabel('U Axis')
        ax.set_ylabel('V Axis')
        return im,

    ani = animation.FuncAnimation(fig, update, frames=len(heatmaps), blit=False)
    plt.show()
    ani.save(animation_location, writer='pillow')
    plt.close()
    
def plot_coverage_heatmap(coverage_counts, grid_shape, u_grid, v_grid, step=None):
    heatmap = coverage_counts.reshape(grid_shape)
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(u_grid, v_grid, heatmap, shading='auto', cmap='hot')
    plt.colorbar(label='Number of Cameras')
    if step is not None:
        plt.title(f'Camera Coverage Heatmap at Step {step}')
    else:
        plt.title('Camera Coverage Heatmap')
    plt.xlabel('U Axis')
    plt.ylabel('V Axis')
    plt.axis('equal')
    plt.show()
    # Optionally, save the figure
    plt.close()

def compute_mean_positions(points, camera_poses):
    mean_point_cloud = np.mean(points, axis=0)
    camera_positions = []
    for pose in camera_poses.values():
        R = pose['R']
        t = pose['t']
        camera_center = -R.T @ t  # Camera center in world coordinates
        camera_positions.append(camera_center)
    camera_positions = np.array(camera_positions)
    mean_camera_position = np.mean(camera_positions, axis=0)
    return mean_point_cloud, mean_camera_position, camera_positions

def compute_total_distance(mean_point_cloud, mean_camera_position, normal_vector):
    vector = mean_camera_position - mean_point_cloud
    total_distance = np.dot(vector, normal_vector)
    return total_distance

def define_plane_axes(plane_normal):
    # Choose an arbitrary vector not parallel to plane_normal
    if np.allclose(plane_normal[:2], 0):
        reference_vector = np.array([0, 1, 0])
    else:
        reference_vector = np.array([0, 0, 1])
    u = np.cross(plane_normal, reference_vector)
    u /= np.linalg.norm(u)
    v = np.cross(plane_normal, u)
    v /= np.linalg.norm(v)
    return u, v

def gaussian_confidence_visualization(witness_counts: np.ndarray, xyz: np.ndarray, output_location: str):
    """_summary_

    Args:
        witness_counts (np.ndarray): For each Gaussian, how many time it has been witnessed
        xyz (np.ndarray): The mean of that Gaussian
        output_location (str): the output confidence ply location
    """
    threshold = -np.inf  # Keep all points

    # Create a mask for points above the threshold
    mask = witness_counts > threshold  # Remove points with counts <= threshold
    
    # Apply the mask to xyz and witness_counts
    xyz_filtered = xyz[mask]
    witness_counts_filtered = witness_counts[mask]
    
    # Recalculate scaled_counts
    max_count_filtered = np.max(witness_counts_filtered)
    if max_count_filtered > 0:
        scaled_counts_filtered = witness_counts_filtered / max_count_filtered  # Values between 0 and 1
    else:
        scaled_counts_filtered = witness_counts_filtered  # All zeros
    
    # Map to colors
    colormap = cm.get_cmap('jet')
    colors = colormap(scaled_counts_filtered)  # Returns RGBA values
    
    # Prepare vertex data
    vertex_data = np.empty(len(xyz_filtered), dtype=[
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4'),
        ('red', 'u1'),
        ('green', 'u1'),
        ('blue', 'u1'),
    ])
    
    vertex_data['x'] = xyz_filtered[:, 0]
    vertex_data['y'] = xyz_filtered[:, 1]
    vertex_data['z'] = xyz_filtered[:, 2]
    vertex_data['red'] = (colors[:, 0] * 255).astype(np.uint8)
    vertex_data['green'] = (colors[:, 1] * 255).astype(np.uint8)
    vertex_data['blue'] = (colors[:, 2] * 255).astype(np.uint8)
    
    # Write to output ply file
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    plydata = PlyData([vertex_element])
    output_filename = output_location
    plydata.write(output_filename)
    

def plot_heatmap_and_seperation(weights: np.ndarray, cluster_map: np.ndarray, K):
    # Visualize and save the weight map
    plt.figure(figsize=(8, 6))
    plt.imshow(weights, cmap='viridis')
    plt.title('Weight Map')
    plt.colorbar(label='Weight')
    plt.xlabel('Grid X-axis')
    plt.ylabel('Grid Y-axis')
    plt.savefig('weight_map.png', dpi=300)
    plt.close()

    # Visualize and save the clustered grid map
    plt.figure(figsize=(8, 6))
    plt.imshow(cluster_map, cmap='tab20')
    plt.title('Clustered Grid Map')
    plt.colorbar(label='Cluster Label')
    plt.xlabel('Grid X-axis')
    plt.ylabel('Grid Y-axis')
    plt.savefig('clustered_grid_map.png', dpi=300)
    plt.close()

    # Calculate and display total weights per cluster
    total_weights = []
    for k in range(K):
        total_weights.append(weights[cluster_map == k].sum())

    print("Total weights per cluster:")
    for k in range(K):
        print(f"Cluster {k}: {total_weights[k]:.4f}")