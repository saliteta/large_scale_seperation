import numpy as np
from utils.cameras import quaternion_to_rotation_matrix
import pycolmap
from typing import Dict, List
import os
from pathlib import Path
def load_points3D(reconstruction: pycolmap.Reconstruction):
    points = []
    for point_id, point in reconstruction.points3D.items():
        # We can ignore color and error for now
        points.append(point.xyz)  # point.xyz is already a numpy array
    return np.array(points)

def load_camera_poses(reconstruction: pycolmap.Reconstruction):
    camera_poses = {}
    for image_id, image in reconstruction.images.items():
        img_dict = image.cam_from_world.todict()
        quat = img_dict['rotation']['quat']
        translation = img_dict['translation']
        camera_id = image.camera_id
        name = image.name
        
        # Convert quaternion to rotation matrix
        R = quaternion_to_rotation_matrix(quat[3], quat[0], quat[1], quat[2])
        
        camera_poses[image_id] = {'R': R, 't': translation, 'camera_id': camera_id, 'name': name}
    
    return camera_poses

def load_camera_intrinsics(reconstruction: pycolmap.Reconstruction):
    intrinsics = {}
    for camera_id, camera in reconstruction.cameras.items():
        model = camera.model
        width = camera.width
        height = camera.height
        params = camera.params
        
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

def regional_map_to_colmap(region_image_map:Dict[int, Dict[int, List]], reconstruction: pycolmap.Reconstruction, output_folder: Path):
    # Process each region
    print(region_image_map.keys())
    print(region_image_map[1].keys())
    exit()
    for region_label, image_ids in region_image_map.items():
        output_folder = os.path.join('separation', f'region_{region_label}')
        os.makedirs(output_folder, exist_ok=True)

        # Create a new Reconstruction object for this region
        region_reconstruction = pycolmap.Reconstruction()

        # Copy cameras used by images in this region
        camera_ids = set()
        for image_id in image_ids:
            if image_id in reconstruction.images:
                camera_ids.add(reconstruction.images[image_id].camera_id)
            else:
                print(f"Image ID {image_id} not found in the reconstruction.")

        for camera_id in camera_ids:
            if camera_id in reconstruction.cameras:
                region_reconstruction.add_camera(reconstruction.cameras[camera_id])
            else:
                print(f"Camera ID {camera_id} not found in the reconstruction.")

        # Copy images and their poses for this region
        for image_id in image_ids:
            if image_id in reconstruction.images:
                image = reconstruction.images[image_id]
                region_reconstruction.add_image(image)
            else:
                print(f"Image ID {image_id} not found in the reconstruction.")

        # Copy Points3D that are visible in these images
        point3D_ids = set()
        for image in region_reconstruction.images.values():
            for point2D in image.points2D:
                if point2D.has_point3D():
                    point3D_ids.add(point2D.point3D_id)

        for point3D_id in point3D_ids:
            if point3D_id in reconstruction.points3D:
                point3D = reconstruction.points3D[point3D_id]
                region_reconstruction.add_point3D(point3D)
            else:
                print(f"Point3D ID {point3D_id} not found in the reconstruction.")

        # Save the region reconstruction in binary format
        region_reconstruction.write(output_folder, binary=True)
    