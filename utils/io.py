import numpy as np
from utils.cameras import quaternion_to_rotation_matrix

def load_points3D(file_path):
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            tokens = line.strip().split()
            point_id = int(tokens[0])
            x = float(tokens[1])
            y = float(tokens[2])
            z = float(tokens[3])
            # We can ignore color and error for now
            points.append([x, y, z])
    return np.array(points)

def load_camera_poses(file_path):
    camera_poses = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if line.startswith('#'):
                idx += 1
                continue
            tokens = line.strip().split()
            if len(tokens) < 8:
                idx += 1
                continue
            image_id = int(tokens[0])
            qw = float(tokens[1])
            qx = float(tokens[2])
            qy = float(tokens[3])
            qz = float(tokens[4])
            tx = float(tokens[5])
            ty = float(tokens[6])
            tz = float(tokens[7])
            camera_id = int(tokens[8])
            name = str(tokens[9])
            # Skip the next line (2D point observations)
            idx += 2
            # Convert quaternion to rotation matrix
            R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
            t = np.array([tx, ty, tz])
            camera_poses[image_id] = {'R': R, 't': t, 'camera_id': camera_id, 'name':name}
        return camera_poses

def load_camera_intrinsics(file_path):
    intrinsics = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            tokens = line.strip().split()
            if len(tokens) == 0:
                continue
            camera_id = int(tokens[0])
            model = tokens[1]
            width = int(tokens[2])
            height = int(tokens[3])
            params = list(map(float, tokens[4:]))
            if model == 'PINHOLE':
                fx, fy, cx, cy = params
            elif model == 'SIMPLE_PINHOLE':
                f, cx, cy = params
                fx = fy = f
            else:
                # Handle other models as needed
                continue
            intrinsics[camera_id] = {
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
                'width': width,
                'height': height
            }
    return intrinsics
