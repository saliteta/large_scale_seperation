from scipy.ndimage import gaussian_filter
import numpy as np
from sklearn.cluster import KMeans
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage import segmentation
import matplotlib.pyplot as plt
from utils.cameras import compute_frustum_plane_intersection
from shapely.geometry import Polygon, Point


def preprocess_heatmap(heatmap):
    # Normalize the heatmap to range [0, 1]
    heatmap_normalized = heatmap / np.max(heatmap)
    
    # Apply Gaussian filter to smooth the heatmap
    heatmap_smooth = gaussian_filter(heatmap_normalized, sigma=2)
    
    return heatmap_smooth

def generate_points_with_heat(heatmap):
    """Generates points according to the heat score at each location."""
    points = []
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            heat = heatmap[i, j]
            if heat > 0:
                points.extend([(i, j)] * int(heat))  # Append the point according to heat value
    return np.array(points)

def assign_points_to_clusters(heatmap, n_clusters):
    """Assigns points to clusters, ensuring heat balance."""
    points = generate_points_with_heat(heatmap)

    # Perform K-means clustering on the points
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    point_labels = kmeans.fit_predict(points)

    # Reassign all points of the same location to the same cluster using majority voting
    label_map = {}
    for i, (x, y) in enumerate(points):
        if (x, y) not in label_map:
            label_map[(x, y)] = []
        label_map[(x, y)].append(point_labels[i])

    # Assign all points at (x, y) to the cluster that has the most votes
    final_labels = np.zeros(heatmap.shape, dtype=int)
    for (x, y), labels in label_map.items():
        majority_label = np.bincount(labels).argmax()  # Get the most common label
        final_labels[x, y] = majority_label

    return final_labels


def extract_boundaries(labels):
    # Find boundaries between regions
    boundaries = segmentation.find_boundaries(labels, mode='outer')
    return boundaries

def plot_heatmap_with_boundaries(heatmap, u_grid, v_grid, boundaries, save_location, step=None):
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
    plt.contour(u_grid, v_grid, boundaries, colors='blue', linewidths=3)
    plt.savefig(f'{save_location}/seperation.png')
    plt.close()

def _associate_images_with_regions(camera_poses, camera_intrinsics, labels, plane_point, normal_vector, u, v, u_grid, v_grid):
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

def associate_images_with_regions(camera_poses, camera_intrinsics, labels, plane_point, plane_normal, u, v, u_grid, v_grid):
    """
    Associate images with regions based on the intersection of camera frustums with regions.

    Parameters:
    - camera_poses: Dictionary mapping image IDs to camera pose dictionaries.
    - camera_intrinsics: Dictionary mapping camera IDs to intrinsic parameter dictionaries.
    - labels: 2D numpy array of region labels.
    - plane_point: A point on the plane (numpy array of shape (3,)).
    - plane_normal: Normal vector of the plane (numpy array of shape (3,)).
    - u, v: Basis vectors defining the plane axes (numpy arrays of shape (3,)).
    - u_grid, v_grid: 1D numpy arrays defining the grid along u and v axes.

    Returns:
    - region_image_map: Dictionary mapping region labels to lists of image IDs.
    """
    # Create a mapping of regions (labels) to image IDs
    region_image_map = {}
    
    # Create grid coordinate mapping
    uu, vv = np.meshgrid(u_grid, v_grid, indexing='ij')
    grid_shape = uu.shape
    grid_points_2d = np.column_stack((uu.ravel(), vv.ravel()))
    
    # For each region label, create a polygon of its area
    regions = {}
    for region_label in np.unique(labels):
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
        # Compute the frustum-plane intersection using the provided function
        frustum_points = compute_frustum_plane_intersection(
            pose, intrinsic, plane_point, plane_normal
        )
        if frustum_points is None:
            continue
        # Project frustum points onto plane axes to get 2D polygon
        frustum_coords_u = np.dot(frustum_points - plane_point, u)
        frustum_coords_v = np.dot(frustum_points - plane_point, v)
        frustum_coords_2d = np.column_stack((frustum_coords_u, frustum_coords_v))
        frustum_poly = Polygon(frustum_coords_2d)
        
        # Check for intersection with each region
        for region_label, region_polygon in regions.items():
            if frustum_poly.intersects(region_polygon):
                region_image_map.setdefault(region_label, []).append(image_id)
    
    return region_image_map


def recursive_partition(weights, mask, clusters, cluster_id, axis, depth, max_depth):
    """
    Recursively partition the grid to balance the total weights.

    Parameters:
    - weights: 2D numpy array of weights.
    - mask: Boolean mask indicating the current region to partition.
    - clusters: 2D array to store cluster labels.
    - cluster_id: Current cluster ID or list of cluster IDs.
    - axis: Axis along which to split (0 for rows, 1 for columns).
    - depth: Current recursion depth.
    - max_depth: Maximum recursion depth (log2 of number of clusters).
    """
    # Base case: If we've reached the maximum depth, assign the cluster ID
    if depth == max_depth or np.sum(mask) == 0:
        clusters[mask] = cluster_id.pop(0)
        return

    # Sum weights along the splitting axis
    sum_weights_along_axis = weights * mask
    sum_weights = sum_weights_along_axis.sum(axis=1 - axis)

    # Compute cumulative weights along the axis
    cumulative_weights = np.cumsum(sum_weights)

    # Find the splitting index
    total_weight = cumulative_weights[-1]
    split_weight = total_weight / 2
    split_index = np.searchsorted(cumulative_weights, split_weight)

    # Create masks for the two subregions
    if axis == 0:
        mask1 = mask.copy()
        mask1[split_index:, :] = False
        mask2 = mask.copy()
        mask2[:split_index, :] = False
    else:
        mask1 = mask.copy()
        mask1[:, split_index:] = False
        mask2 = mask.copy()
        mask2[:, :split_index] = False

    # Alternate axis for the next level
    next_axis = 1 - axis

    # Recursively partition the two subregions
    recursive_partition(weights, mask1, clusters, cluster_id, next_axis, depth + 1, max_depth)
    recursive_partition(weights, mask2, clusters, cluster_id, next_axis, depth + 1, max_depth)
