import numpy as np
from utils.cameras import quaternion_to_rotation_matrix
import pycolmap
from typing import Dict, List, Tuple
import os
from pathlib import Path
import copy
from colorama import Fore, Back

from tqdm import tqdm

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
    assert len(camera_poses) > 0, " At least have more than one images"
    return camera_poses

def load_camera_intrinsics(reconstruction: pycolmap.Reconstruction):
    intrinsics = {}
    for camera_id, camera in reconstruction.cameras.items():
        camera = camera.todict()
        model = camera['model']
        width = camera['width']
        height = camera['height']
        params = camera['params']
        
        if model.name == 'PINHOLE':
            fx, fy, cx, cy = params
        elif model.name == 'SIMPLE_PINHOLE':
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
        
    assert len(intrinsics) >= 1, "at least have one camera, we have zero"
    return intrinsics

def verify_bounded(region_image_map:Dict[int, Dict[int, List]], bound: Tuple[int]) -> bool:
    """_summary_

    Args:
        region_image_map (Dict[int, Dict[int, List]]): Regional Description, each list contain Image ID
        bound (Tuple[int]): One upper bound and one lower bound 
    """
    for key in region_image_map:
        sequence_ids = region_image_map[key]
        length = len(sequence_ids)
        if length <= bound[0] and length>= bound[1]:
            continue
        else:
            return False

    return True        
    
def reginal_map_to_colmap(region_image_map:Dict[int, Dict[int, List]], 
                           reconstruction: pycolmap.Reconstruction, 
                           output_folder: Path,
                           bound: Tuple[int] = (220, 280)) -> bool:
    """_summary_

    Args:
        region_image_map (Dict[int, List]): Is a dictionary, key is the region ID, and the content is a list of image ID 
        reconstruction (pycolmap.Reconstruction): Original COLMAP reconstruction for large scale reconstruction
        output_folder (Path): The base output folder
    Output: 
        The output will be a sequence of folder, for example output_folder/0, output_folder/1, and so on
    """
    #if verify_bounded(region_image_map=region_image_map, bound=bound) == False:
    #    return False
    os.makedirs(output_folder, exist_ok=True)
    # Process each region
    for region_label, image_ids in tqdm(region_image_map.items(), desc="saving sub-colmap and point cloud"):

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

                # Deep copy the image to avoid modifying the original
                new_image = copy.deepcopy(image)

                # Add the image to the reconstruction
                region_reconstruction.add_image(new_image)
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

                # Extract necessary information for the Point3D object
                xyz = np.expand_dims(point3D.xyz, axis=1).astype(np.float64)
                color = np.expand_dims(point3D.color, axis=1).astype(np.uint8)
                track = pycolmap.Track()

                # Add the Point3D to the region_reconstruction
                point3D_id_new = region_reconstruction.add_point3D(xyz, track, color)
            else:
                print(f"Point3D ID {point3D_id} not found in the reconstruction.")
        
        region_folder = os.path.join(output_folder, f"{region_label}")
        os.makedirs(region_folder, exist_ok=True)
        # Save the region reconstruction in binary format
        region_reconstruction.write(region_folder)
        region_reconstruction.export_PLY(os.path.join(region_folder, 'points3D.ply'))
        
    return True

def create_dir_and_hint(directory_path: str):

    if os.path.exists(directory_path):
        print(Fore.YELLOW, f"Directory exist, continue processing")
    else:
        os.makedirs(directory_path)
