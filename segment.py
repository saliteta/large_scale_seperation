import numpy as np
from utils.io import load_camera_poses, load_points3D, create_dir_and_hint, reginal_map_to_colmap
from utils.clustering import project_camera_to_plan, cluster_positions, clustering_expansion
import pycolmap
import os
from tqdm import tqdm
from typing import Dict, List
import argparse
from colorama import Fore, Back
import matplotlib.pyplot as plt

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


def extract_camera_positions(camera_poses):
    positions = []
    image_ids = []
    image_names = []
    for image_id, pose in camera_poses.items():
        t = pose['t']
        R = pose['R']
        camera_center = -R.T @ t  # Camera center in world coordinates
        positions.append(camera_center)
        image_ids.append(image_id)
        image_names.append(pose['name'])
    return np.array(positions), image_ids, image_names


def index_to_id(labels: Dict[int, int], image_id: List[int]) ->Dict[int, int]:
    for key in labels:
        for i in range(len(labels[key])):
            labels[key][i] = image_id[labels[key][i]]
    
    return labels


def visualize_clusters(positions_2d, labels, image_names):
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(positions_2d[:, 0], positions_2d[:, 1], c=labels, cmap='tab20', s=5)
    plt.title('Image Clusters Projected onto 2D Plane')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter, label='Cluster Label')
    plt.savefig(image_names)

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
    
   # Extract camera positions
    positions, image_ids, _ = extract_camera_positions(camera_poses)

    # Project positions onto 2D plane using PCA
    positions_2d = project_camera_to_plan(camera_pose=positions, points3D=points)

    # Perform clustering on 2D points
    labels, kmeans_model = cluster_positions(positions_2d, lower_bound, upper_bound)
    image_name = os.path.join(output_path, 'summary.png')
    visualize_clusters(positions_2d, labels, image_name)
    
    # Create overlapping clusters

    cluster_assignments = clustering_expansion(positions_2d=positions_2d, labels=labels, expanding_target=upper_bound+30)
    
    regional_map = index_to_id(cluster_assignments, image_id=image_ids)

    # Save cluster assignments
    reginal_map_to_colmap(region_image_map=regional_map, reconstruction=reconstruction, output_folder=output_path)
    
    print(Fore.GREEN, Back.RESET, f"Segmentation Accomplished result save in {output_path}")
if __name__ == "__main__":
    main()
    
    