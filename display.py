import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import matplotlib.animation as animation
from matplotlib.path import Path

def load_points3D(file_path):
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            tokens = line.strip().split()
            point_id = int(tokens[0])
            x = float(tokens[1])
            y = float(tokens[2])
            z = float(tokens[3])
            # We can ignore color and error for now
            points.append([x, y, z])
    return np.array(points)

def load_camera_poses(file_path):
    camera_poses = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if line.startswith('#'):
                idx += 1
                continue
            tokens = line.strip().split()
            if len(tokens) < 8:
                idx += 1
                continue
            image_id = int(tokens[0])
            qw = float(tokens[1])
            qx = float(tokens[2])
            qy = float(tokens[3])
            qz = float(tokens[4])
            tx = float(tokens[5])
            ty = float(tokens[6])
            tz = float(tokens[7])
            camera_id = int(tokens[8])
            name = str(tokens[9])
            # Skip the next line (2D point observations)
            idx += 2
            # Convert quaternion to rotation matrix
            R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
            t = np.array([tx, ty, tz])
            camera_poses[image_id] = {'R': R, 't': t, 'camera_id': camera_id, 'name':name}
        return camera_poses

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz])
    q_norm = q / np.linalg.norm(q)
    qw, qx, qy, qz = q_norm
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R

def compute_plane_pca(points):
    mean_point = np.mean(points, axis=0)
    centered_points = points - mean_point
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # The normal vector corresponds to the smallest eigenvalue
    normal_vector = eigenvectors[:, 0]
    return mean_point, normal_vector

def load_camera_intrinsics(file_path):
    intrinsics = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            tokens = line.strip().split()
            if len(tokens) == 0:
                continue
            camera_id = int(tokens[0])
            model = tokens[1]
            width = int(tokens[2])
            height = int(tokens[3])
            params = list(map(float, tokens[4:]))
            if model == 'PINHOLE':
                fx, fy, cx, cy = params
            elif model == 'SIMPLE_PINHOLE':
                f, cx, cy = params
                fx = fy = f
            else:
                # Handle other models as needed
                continue
            intrinsics[camera_id] = {
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
                'width': width,
                'height': height
            }
    return intrinsics

def get_camera_frustum_corners(camera_pose, camera_intrinsic, max_depth=1000):
    fx = camera_intrinsic['fx']
    fy = camera_intrinsic['fy']
    cx = camera_intrinsic['cx']
    cy = camera_intrinsic['cy']
    width = camera_intrinsic['width']
    height = camera_intrinsic['height']

    # Image corners in pixel coordinates
    image_corners = np.array([
        [0, 0],               # Top-left
        [width, 0],           # Top-right
        [width, height],      # Bottom-right
        [0, height]           # Bottom-left
    ])

    # Convert pixel coordinates to normalized device coordinates (NDC)
    ndc_corners = np.zeros((4, 3))
    for i, (u, v) in enumerate(image_corners):
        x = (u - cx) / fx
        y = (v - cy) / fy
        ndc_corners[i] = np.array([x, y, 1])  # z = 1 in camera space

    # Transform NDC corners to world coordinates
    R = camera_pose['R']
    t = camera_pose['t']
    camera_center = -R.T @ t

    frustum_corners = []
    for corner in ndc_corners:
        # Scale the direction vector by max_depth
        direction = R.T @ corner
        point = camera_center + direction * max_depth
        frustum_corners.append(point)

    return np.array(frustum_corners)

def intersect_ray_plane(ray_origin, ray_direction, plane_point, plane_normal):
    denominator = np.dot(ray_direction, plane_normal)
    if np.abs(denominator) < 1e-6:
        # Ray is parallel to the plane
        return None
    d = np.dot(plane_point - ray_origin, plane_normal) / denominator
    if d < 0:
        # Intersection is behind the ray origin
        return None
    intersection_point = ray_origin + d * ray_direction
    return intersection_point

def compute_frustum_plane_intersection(camera_pose, camera_intrinsic, plane_point, plane_normal):
    R = camera_pose['R']
    t = camera_pose['t']
    camera_center = -R.T @ t

    fx = camera_intrinsic['fx']
    fy = camera_intrinsic['fy']
    cx = camera_intrinsic['cx']
    cy = camera_intrinsic['cy']
    width = camera_intrinsic['width']
    height = camera_intrinsic['height']

    # Image corners in pixel coordinates
    image_corners = np.array([
        [0, 0],               # Top-left
        [width, 0],           # Top-right
        [width, height],      # Bottom-right
        [0, height]           # Bottom-left
    ])

    intersection_points = []
    for u, v in image_corners:
        x = (u - cx) / fx
        y = (v - cy) / fy
        ray_direction = R.T @ np.array([x, y, 1])
        ray_direction /= np.linalg.norm(ray_direction)
        intersection_point = intersect_ray_plane(camera_center, ray_direction, plane_point, plane_normal)
        if intersection_point is not None:
            intersection_points.append(intersection_point)

    if len(intersection_points) >= 3:
        # Return the polygon formed by the intersection points
        return np.array(intersection_points)
    else:
        # The camera frustum does not intersect the plane
        return None


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

def main():
    # Paths to COLMAP output files
    points3D_path = 'points3D.txt'
    images_path = 'images.txt'
    cameras_path = 'cameras.txt'

    # Load data
    points = load_points3D(points3D_path)
    camera_poses = load_camera_poses(images_path)
    camera_intrinsics = load_camera_intrinsics(cameras_path)

    # Compute the plane (mean point and normal vector)
    mean_point_cloud, normal_vector = compute_plane_pca(points)

    # Compute mean camera position
    _, mean_camera_position, _ = compute_mean_positions(points, camera_poses)

    # Compute total distance to move along the normal vector
    total_distance = compute_total_distance(mean_point_cloud, mean_camera_position, normal_vector)

    # Number of steps to move the plane
    N = 10  # Adjust as needed

    # Create distances along the normal vector from 0 to total_distance
    distances = np.linspace(0, total_distance, N)

    # Define plane axes (u, v) orthogonal to the normal vector
    u, v = define_plane_axes(normal_vector)

    # Initialize lists to store data
    heatmaps = []
    plane_points_list = []

    # Loop over the steps
    for i, d in enumerate(distances):
        print(f"Processing step {i+1}/{N}")
        # Move the plane
        plane_point = mean_point_cloud + d * normal_vector
        plane_points_list.append(plane_point)

        # For each camera, compute the frustum-plane intersection
        projected_frustums = []
        for image_id, pose in camera_poses.items():
            camera_id = pose['camera_id']
            intrinsic = camera_intrinsics.get(camera_id)
            if intrinsic is None:
                continue  # Skip if intrinsics are missing
            frustum_polygon = compute_frustum_plane_intersection(
                pose, intrinsic, plane_point, normal_vector
            )
            if frustum_polygon is not None:
                projected_frustums.append(frustum_polygon)

        if not projected_frustums:
            print(f"No frustums intersect with the plane at step {i+1}.")
            continue

        # Create a grid over the plane (ensure consistent grid across steps)
        grid_resolution = 0.2  # Adjust as needed
        if i == 0:
            # For the first step, compute the overall grid bounds
            grid_points, grid_shape, u_grid, v_grid = create_plane_grid(
                plane_point, u, v, projected_frustums, grid_resolution
            )
        else:
            # For subsequent steps, use the same grid
            grid_points = create_grid_points(plane_point, u, v, u_grid, v_grid)

        # Compute coverage
        coverage_counts = compute_coverage(
            grid_points, grid_shape, plane_point, u, v, projected_frustums
        )

        # Store heatmap
        heatmaps.append(coverage_counts.reshape(grid_shape))

        # Optionally, visualize or save the heatmap at each step
        plot_coverage_heatmap(coverage_counts, grid_shape, u_grid, v_grid, step=i+1)

    # After the loop, you can create an animation of the heatmaps
    create_heatmap_animation(heatmaps, u_grid, v_grid, grid_shape)

if __name__ == '__main__':
    main()
