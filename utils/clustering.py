import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# If not already imported
from colorama import Fore, Style
from typing import Tuple

def compute_pca_on_points3D(points3D):
    mean_points3D = np.mean(points3D, axis=0)
    points3D_centered = points3D - mean_points3D
    pca = PCA(n_components=3)
    pca.fit(points3D_centered)
    components = pca.components_
    explained_variance = pca.explained_variance_
    return pca, components, explained_variance



def project_camera_positions_onto_plane(camera_positions, normal_vector):
    """
    Project camera positions onto the plane orthogonal to the given normal vector.
    """
    camera_positions = camera_positions - np.mean(camera_positions, axis=0)
    projected_positions = []
    for pos in camera_positions:
        # Compute the component along the normal vector
        component = np.dot(pos, normal_vector) * normal_vector
        # Subtract the component to get the projection onto the plane
        projected_pos = pos - component
        projected_positions.append(projected_pos)
    return np.array(projected_positions)

def reduce_to_2d(projected_positions):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    positions_2d = pca.fit_transform(projected_positions)
    return positions_2d

def project_camera_to_plan(camera_pose, points3D):
    pca, components, explained_variance = compute_pca_on_points3D(points3D=points3D)
    projected_camera_pose = project_camera_positions_onto_plane(camera_positions=camera_pose, normal_vector=components[2])
    reduced_2d_pose = reduce_to_2d(projected_positions=projected_camera_pose)
    
    return reduced_2d_pose




def cluster_positions(positions_2d, lower_bound, upper_bound):
    N = positions_2d.shape[0]
    average_cluster_size = (lower_bound + upper_bound) / 2
    k = int(np.ceil(N / average_cluster_size))
    print(Fore.BLUE + f"Clustering into {k} clusters to maintain cluster size between {lower_bound} and {upper_bound}")
    print(Style.RESET_ALL)
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(positions_2d)
    return labels, kmeans


import numpy as np

def clustering_expansion(positions_2d: np.ndarray, labels:np.ndarray, expanding_target:int):
    """
    Expands each cluster to reach the target number of images by adding nearest neighbors to the cluster center.

    Args:
        positions_2d (np.ndarray): The 2D positions of data points, shape (N, 2).
        labels (np.ndarray): The initial cluster labels, shape (N,).
        expanding_target (int): The desired number of images per cluster after expansion.

    Returns:
        expanded_clusters (dict): Dictionary where key is cluster label and value is a list of indices of images assigned to that cluster.
    """
    """
    Expands each cluster to reach the target number of images by adding nearest neighbors to the cluster center,
    allowing overlaps between clusters.

    Args:
        positions_2d (np.ndarray): The 2D positions of data points, shape (N, 2).
        labels (np.ndarray): The initial cluster labels, shape (N,).
        expanding_target (int): The desired number of images per cluster after expansion.

    Returns:
        expanded_clusters (dict): Dictionary where key is cluster label and value is a list of indices of images assigned to that cluster.
    """
    # Initialize the output dictionary
    expanded_clusters = {}

    # For each cluster
    unique_labels = np.unique(labels)
    for label in unique_labels:
        # Get indices of points in the current cluster
        cluster_indices = np.where(labels == label)[0]
        cluster_points = positions_2d[cluster_indices]

        # Compute the cluster center
        cluster_center = np.mean(cluster_points, axis=0)

        # Initialize the set of points in the expanded cluster
        expanded_indices = set(cluster_indices.tolist())

        # If the cluster already has the target size or more, continue
        print(f"Initial size of cluster {label}: {len(expanded_indices)}")
        if len(expanded_indices) >= expanding_target:
            expanded_clusters[label] = list(expanded_indices)
            continue

        # Find distances from all points to the cluster center
        distances = np.linalg.norm(positions_2d - cluster_center, axis=1)

        # Get indices of all points sorted by distance to cluster center
        sorted_indices = np.argsort(distances)

        # Add nearest points to the cluster until reaching the target size
        for idx in sorted_indices:
            if len(expanded_indices) >= expanding_target:
                break
            expanded_indices.add(idx)

        # Store the expanded cluster
        expanded_clusters[label] = list(expanded_indices)
        print(f"Expanded size of cluster {label}: {len(expanded_indices)}")

    return expanded_clusters