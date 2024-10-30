import os
import numpy as np
import pycolmap
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from utils.io import load_camera_intrinsics, load_camera_poses
import argparse
from typing import List, Tuple
from utils.visualization import gaussian_confidence_visualization
from colorama import Fore, Back

def read_ply_xyz(filename):
    plydata = PlyData.read(filename)
    vertex_data = plydata['vertex']
    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']
    xyz = np.vstack((x, y, z)).T  # Shape: (N, 3)
    return xyz

def read_ply_vertex_data(filename):
    plydata = PlyData.read(filename)
    vertex_data = plydata['vertex']
    return vertex_data


def parser():
    parser = argparse.ArgumentParser()
    
    # Required positional arguments
    parser.add_argument("--colmap_folder", help="The seperation code, it should contains many sparse reconstruction model")
    parser.add_argument("--model_folder", help="The model_folder location")
    parser.add_argument("--output_folder", help="The project location")
    
    # Optional arguments
    parser.add_argument("--filter_percentage", type=int, help="The upper bound of each region we need to use. CUDA memory is larger, this number is large, 24GB is around 280", default=35)
    
    args = parser.parse_args()
    return args

def to_list(colmap_folder: str, model_folder: str) -> Tuple[List[str]]:
    """
    We should make pair 
    Args:
        colmap_folder (str): colmap seperation, it should contain, 1, 2, 3, 4 and so on
        model_folder (str): model folder, it should also contain, 1, 2, 3, 4 and so on
    Returns:
        Tuple[List[str]]: [colmap reconstruction list, ply list]
    """
    colmap_list = [ name for name in os.listdir(colmap_folder) if os.path.isdir(os.path.join(colmap_folder, name)) ]
    model_folder_list = []
    colmap_reconstruction_list = []
    error = False
    for i in range(len(colmap_list)):
        model_location = os.path.join(model_folder, colmap_list[i], 'point_cloud', 'iteration_30000', 'point_cloud.ply')
        if os.path.exists(model_location): # Seperation Generate Accomplished
            model_folder_list.append(os.path.join(model_location))
            colmap_reconstruction_list.append(os.path.join(colmap_folder, colmap_list[i]))
        else:
            log_location = os.path.join(model_folder, colmap_list[i], 'training.log')
            print(Fore.RED, Back.YELLOW, f"[Error!]: Gaussian Training Does Not Accomplsihed! We do not have the following file {model_location}, Training Log at: {log_location}")
            error = True
    if error and (len(model_folder_list)>1):
        print(Fore.YELLOW, Back.GREEN, "[Warning!]: Some seperation did not reconstruct successfully, we merge other part together, if this is not desired, stop the process and exam the trainning log")
    return colmap_reconstruction_list, model_folder_list

def main():

    args = parser()
    seperation_folder = args.colmap_folder
    model_folder = args.model_folder
    output_folder = args.output_folder
    filter_percentage = args.filter_percentage
    
    colmap_folders, ply_files = to_list(colmap_folder=seperation_folder, model_folder=model_folder)
    
    
    assert len(ply_files) == len(colmap_folders), "Number of PLY files and COLMAP folders must be the same."
    
    filtered_vertex_data_list = []
    print(Fore.RESET, Back.RESET)
    for ply_filename, colmap_folder in zip(ply_files, colmap_folders):
        print(f'Processing PLY file: {ply_filename}')
        print(f'Using COLMAP reconstruction: {colmap_folder}')
        
        # Load reconstruction
        reconstruction = pycolmap.Reconstruction(colmap_folder)
        
        # Load camera poses and intrinsics
        camera_poses = load_camera_poses(reconstruction)
        camera_intrinsics = load_camera_intrinsics(reconstruction)
        
        # Read PLY file (including all vertex attributes)
        vertex_data = read_ply_vertex_data(ply_filename)
        num_points = len(vertex_data)
        
        # Extract xyz coordinates
        xyz = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T  # Shape: (N, 3)
        
        witness_counts = np.zeros(num_points, dtype=int)
        
        X_world = xyz.T  # Shape: (3, N)
        
        for image_id, pose in tqdm(camera_poses.items(), desc='Projecting points'):
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
        confidence_score_location = os.path.join('/'.join(ply_filename.split('/')[:-3]), 'confidence_heatmap.ply')
        gaussian_confidence_visualization(witness_counts=witness_counts, xyz=xyz, output_location=confidence_score_location) # output to model location
        
        # Remove bottom pecent Gaussians based on witness counts
        percent = filter_percentage
        threshold = np.percentile(witness_counts, percent)
        
        mask = witness_counts > threshold  # Keep points with counts greater than threshold
        
        # Apply mask to vertex data
        filtered_vertex_data = vertex_data[mask]
        
        # Append to list
        filtered_vertex_data_list.append(filtered_vertex_data)
    
    # Concatenate all filtered vertex data
    concatenated_vertex_data = np.concatenate(filtered_vertex_data_list)
    
    # Create PlyElement
    vertex_element = PlyElement.describe(concatenated_vertex_data, 'vertex')
    plydata = PlyData([vertex_element])
    
    # Write to output PLY file
    outcome_ply = os.path.join(output_folder, 'merged_filtered.ply')
    plydata.write(outcome_ply)
    print(f'Final concatenated PLY file written to {outcome_ply}')

if __name__ == '__main__':
    main()