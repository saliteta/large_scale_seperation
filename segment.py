from scipy.ndimage import gaussian_filter
import numpy as np
from skimage import morphology
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage import segmentation
import matplotlib.pyplot as plt
from display import *

def preprocess_heatmap(heatmap):
    # Normalize the heatmap to range [0, 1]
    heatmap_normalized = heatmap / np.max(heatmap)
    
    # Apply Gaussian filter to smooth the heatmap
    heatmap_smooth = gaussian_filter(heatmap_normalized, sigma=2)
    
    return heatmap_smooth


def segment_heatmap(heatmap_smooth):
    # Compute the local minima (negative peaks) as markers
    local_maxi = peak_local_max(
        heatmap_smooth, indices=False, footprint=np.ones((3, 3)), labels=None
    )
    markers = ndi.label(local_maxi)[0]
    
    # Apply watershed segmentation
    labels = watershed(-heatmap_smooth, markers, mask=heatmap_smooth)
    
    return labels

def extract_boundaries(labels):
    # Find boundaries between regions
    boundaries = segmentation.find_boundaries(labels, mode='outer')
    return boundaries

def plot_heatmap_with_boundaries(heatmap, u_grid, v_grid, boundaries, step=None):
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(u_grid, v_grid, heatmap, shading='auto', cmap='hot')
    plt.colorbar(label='Number of Cameras')
    if step is not None:
        plt.title(f'Camera Coverage Heatmap with Boundaries at Step {step}')
    else:
        plt.title('Camera Coverage Heatmap with Boundaries')
    plt.xlabel('U Axis')
    plt.ylabel('V Axis')
    plt.axis('equal')
    
    
    # Overlay boundaries
    plt.contour(u_grid, v_grid, boundaries, colors='blue', linewidths=1)
    plt.savefig(f'visualization/boundaries/boundaries_{step}.png')
    plt.close()

def associate_images_with_regions(camera_poses, camera_intrinsics, labels, plane_point, normal_vector, u, v, u_grid, v_grid):
    from shapely.geometry import Polygon, Point
    from shapely.ops import unary_union
    
    # Create a mapping of regions (labels) to image IDs
    region_image_map = {}
    
    # Create grid coordinate mapping
    uu, vv = np.meshgrid(u_grid, v_grid)
    grid_shape = uu.shape
    grid_points_2d = np.column_stack((uu.ravel(), vv.ravel()))
    
    # For each region label, create a polygon of its area
    regions = {}
    for region_label in np.unique(labels):
        if region_label == 0:
            continue  # Skip background
        mask = labels == region_label
        region_coords = grid_points_2d[mask.ravel()]
        if len(region_coords) < 3:
            continue  # Need at least 3 points to form a polygon
        polygon = Polygon(region_coords)
        regions[region_label] = polygon
    
    # For each camera, determine which regions it covers
    for image_id, pose in camera_poses.items():
        camera_id = pose['camera_id']
        intrinsic = camera_intrinsics.get(camera_id)
        if intrinsic is None:
            continue  # Skip if intrinsics are missing
        # Compute the frustum-plane intersection
        frustum_polygon = compute_frustum_plane_intersection(
            pose, intrinsic, plane_point, normal_vector
        )
        if frustum_polygon is None:
            continue
        # Project frustum points onto plane axes to get 2D polygon
        frustum_coords_u = np.dot(frustum_polygon - plane_point, u)
        frustum_coords_v = np.dot(frustum_polygon - plane_point, v)
        frustum_coords_2d = np.column_stack((frustum_coords_u, frustum_coords_v))
        frustum_poly = Polygon(frustum_coords_2d)
        
        # Check for intersection with each region
        for region_label, region_polygon in regions.items():
            if frustum_poly.intersects(region_polygon):
                region_image_map.setdefault(region_label, []).append(image_id)
    
    return region_image_map


def main():
    # Paths to COLMAP output files
    points3D_path = 'sparse/0/points3D.txt'
    images_path = 'sparse/0/images.txt'
    cameras_path = 'sparse/0/cameras.txt'

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
        #plot_coverage_heatmap(coverage_counts, grid_shape, u_grid, v_grid, step=i+1)
        
        # Preprocess the heatmap
        heatmap = coverage_counts.reshape(grid_shape)
        heatmap_smooth = preprocess_heatmap(heatmap)
        
        # Segment the heatmap
        labels = segment_heatmap(heatmap_smooth)
        
        # Extract boundaries
        boundaries = extract_boundaries(labels)
        
        # Plot the heatmap with boundaries
        plot_heatmap_with_boundaries(heatmap, u_grid, v_grid, boundaries, step=i+1)
        
        # Associate images with regions
        region_image_map = associate_images_with_regions(
            camera_poses, camera_intrinsics, labels, plane_point, normal_vector, u, v, u_grid, v_grid
        )
if __name__ == "__main__":
    main()