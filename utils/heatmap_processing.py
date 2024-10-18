from scipy.ndimage import gaussian_filter
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage import segmentation
import matplotlib.pyplot as plt
from utils.cameras import compute_frustum_plane_intersection

def preprocess_heatmap(heatmap):
    # Normalize the heatmap to range [0, 1]
    heatmap_normalized = heatmap / np.max(heatmap)
    
    # Apply Gaussian filter to smooth the heatmap
    heatmap_smooth = gaussian_filter(heatmap_normalized, sigma=2)
    
    return heatmap_smooth

def segment_heatmap(heatmap_smooth):
    # Compute the local maxima coordinates
    local_maxi_coords = peak_local_max(
        heatmap_smooth, footprint=np.ones((3, 3)), labels=None
    )
    
    # Create an empty array of the same shape as heatmap_smooth to hold the markers
    markers = np.zeros_like(heatmap_smooth, dtype=int)
    
    # Label each local maxima with a unique value
    for i, coords in enumerate(local_maxi_coords, 1):
        markers[tuple(coords)] = i
    
    # Apply watershed segmentation using the labeled markers
    labels = watershed(-heatmap_smooth, markers, mask=heatmap_smooth)
    
    return labels

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



