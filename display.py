import numpy as np
from utils.io import load_camera_intrinsics, load_camera_poses, load_points3D
from utils.cameras import compute_frustum_plane_intersection, compute_plane_pca
from utils.visualization import compute_mean_positions, compute_coverage, compute_total_distance, define_plane_axes, create_grid_points, create_heatmap_animation, create_plane_grid, plot_coverage_heatmap


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
