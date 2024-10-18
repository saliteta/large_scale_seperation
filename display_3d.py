import numpy as np
import pycolmap
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils.io import load_camera_intrinsics, load_camera_poses
from tqdm import tqdm

def read_ply_xyz(filename):
    plydata = PlyData.read(filename)
    vertex_data = plydata['vertex']
    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']
    xyz = np.vstack((x, y, z)).T  # Shape: (N, 3)
    return xyz

def main():
    # Load reconstruction
    reconstruction = pycolmap.Reconstruction('/data/butian/GauUscene/GauU_Scene/CUHK_LOWER_CAMPUS_COLMAP/seperation_code/result/distance_0/1')
    
    # Load camera poses and intrinsics
    camera_poses = load_camera_poses(reconstruction)
    camera_intrinsics = load_camera_intrinsics(reconstruction)
    
    # Read ply file
    ply_filename = '/data/butian/GauUscene/GauU_Scene/CUHK_LOWER_CAMPUS_COLMAP/gaussian-splatting/output/4a3d50b4-f/point_cloud/iteration_30000/point_cloud.ply'
    xyz = read_ply_xyz(ply_filename)  # Shape: (N, 3)
    
    witness_counts = np.zeros(xyz.shape[0], dtype=int)
    
    for image_id, pose in tqdm(camera_poses.items()):
        R = pose['R']  # (3,3)
        t = pose['t']  # (3,)
        camera_id = pose['camera_id']
        name = pose['name']
        
        # Get camera intrinsics
        intrinsics = camera_intrinsics[camera_id]
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx = intrinsics['cx']
        cy = intrinsics['cy']
        width = intrinsics['width']
        height = intrinsics['height']
        
        # Transform points to camera coordinates
        X_world = xyz.T  # Shape: (3, N)
        X_cam = R @ X_world + t[:, np.newaxis]  # Shape: (3, N)
        
        # Keep points with positive depth
        valid_indices = X_cam[2, :] > 0
        X_cam = X_cam[:, valid_indices]
        indices = np.where(valid_indices)[0]
        
        # Project points to image plane
        x = (fx * X_cam[0, :] / X_cam[2, :]) + cx
        y = (fy * X_cam[1, :] / X_cam[2, :]) + cy
        
        # Check if points are within image boundaries
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        
        # Increment witness counts
        witness_counts[indices[valid]] += 1
        
# Generate 10 PLY files, removing least confident points
    percentages = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    for percent in percentages:
        if percent == 0:
            threshold = -np.inf  # Keep all points
        else:
            # Compute the threshold value for the given percentage
            threshold = np.percentile(witness_counts, percent)
        
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
        output_filename = f'/data/butian/GauUscene/GauU_Scene/CUHK_LOWER_CAMPUS_COLMAP/confidence_2/output_confidence_{percent}percent.ply'
        plydata.write(output_filename)
        print(f'Generated {output_filename}')

if __name__ == '__main__':
    main()