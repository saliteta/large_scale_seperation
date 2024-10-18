import numpy as np
from utils.io import load_camera_intrinsics, load_camera_poses, load_points3D, regional_map_to_colmap, create_dir_and_hint
from utils.cameras import compute_frustum_plane_intersection, compute_plane_pca
from utils.visualization import compute_mean_positions, compute_coverage, compute_total_distance, define_plane_axes, create_grid_points, create_plane_grid
from utils.heatmap_processing import preprocess_heatmap, segment_heatmap, extract_boundaries, plot_heatmap_with_boundaries, associate_images_with_regions
import pycolmap
import os
from tqdm import tqdm
import argparse
from colorama import Fore, Back
from utils.visualization import create_heatmap_animation

def parser():
    parser = argparse.ArgumentParser()
    
    # Required positional arguments
    parser.add_argument("--colmap_path", help="The original large scale scene colmap path, something like sparse/0")
    parser.add_argument("--output", help="This is the output project name, related results will be stored here")
    
    # Optional arguments
    parser.add_argument("--image_upper_bound", type=int, help="The upper bound of each region we need to use. CUDA memory is larger, this number is large, 24GB is around 280", default=285)
    parser.add_argument("--image_lower_bound", type=int, help="Usually, we require the image lower bound as high as possible, default is 230", default=200)
    
    args = parser.parse_args()
    return args

def main():
    # Paths to COLMAP output files

    args = parser()
    model_path = args.colmap_path
    output_path = args.output # project_name/segmentation
    
    create_dir_and_hint(directory_path=output_path)
    
    lower_bound = args.image_lower_bound
    upper_bound = args.image_upper_bound
    
    reconstruction = pycolmap.Reconstruction(model_path) # we directly load the database
    print(Fore.GREEN, f"The colmap sparse reconstruction model: {model_path} correctly loaded")
    print(Fore.WHITE, reconstruction.summary())
    # Load data
    points = load_points3D(reconstruction)
    camera_poses = load_camera_poses(reconstruction)
    camera_intrinsics = load_camera_intrinsics(reconstruction)
    
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
    filter_seperation = []
    # Loop over the steps
    print("Start Processing and Segmentation")
    for i, d in enumerate(tqdm(distances)):
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
        
        # Preprocess the heatmap
        heatmap = coverage_counts.reshape(grid_shape)
        heatmap_smooth = preprocess_heatmap(heatmap)
        
        # Segment the heatmap
        labels = segment_heatmap(heatmap_smooth)
        
        # Extract boundaries
        boundaries = extract_boundaries(labels)
        
        # Plot the heatmap with boundaries
        
        # Associate images with regions
        region_image_map = associate_images_with_regions(
            camera_poses, camera_intrinsics, labels, plane_point, normal_vector, u, v, u_grid, v_grid
        )
        
        distance_seperation_folder = os.path.join(output_path, f"distance_{i}")
        flag = regional_map_to_colmap(
            region_image_map=region_image_map,
            reconstruction=reconstruction,
            output_folder=distance_seperation_folder,
            bound=(upper_bound, lower_bound)
        )
        
        if flag == False:
            filter_seperation.append(i)
            continue
        
        plot_heatmap_with_boundaries(heatmap, u_grid, v_grid, boundaries, distance_seperation_folder, step=i+1)
    
    create_heatmap_animation(heatmaps, 
                             u_grid, 
                             v_grid,
                             animation_location=os.path.join(output_path, "summary.gif"))
    
    if len(filter_seperation) == 10:
        print(Fore.RED, "Current reginal map does not satisfy the bounded requirment, try to decrease lower bound")
        exit(1)
    else:
        print(Fore.GREEN, f"Segment Accomplish, we filtered: {filter_seperation} distance, result stored in the output folder: {output_path}")
    
if __name__ == "__main__":
    main()
    
    