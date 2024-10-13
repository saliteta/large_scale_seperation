import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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

def create_heatmap_animation(heatmaps, u_grid, v_grid, grid_shape):
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
    ani.save('../../visualization/heatmap_animation.gif', writer='pillow')
    plt.show()

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
    plt.savefig(f'../../visualization/heatmap_step_{step}.png')
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
