import numpy as np

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
