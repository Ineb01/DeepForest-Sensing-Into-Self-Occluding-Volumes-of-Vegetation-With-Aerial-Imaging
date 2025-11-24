import cv2
import os

output_resolution = 2256  # Set the desired output size
scale_factor = 0.5  # Example scale factor (e.g., 0.5 means 50% of the cropped size)


# Paths
input_folder = r"C:\\Users\\benja\\Desktop\\git\\university\\master-thesis\\practical\\data\\dataset_May\\RED_irradiancee_RGB\\cropped"    # original images that needs to be cropped
output_folder = r"C:\\Users\\benja\\Desktop\\git\\university\\master-thesis\\practical\\data\\dataset_May\\RED_LFR"            # cropped_images path (This is then the folder that the renderer will load images from)
path = r'C:\\Users\\benja\\Desktop\\git\\university\\master-thesis\\practical\\data\\dataset_May\\RED_colmap_alignment\\output'                     # folder where colmap data are stored (all the txt files )
outpath = r'C:\\Users\\benja\\Desktop\\git\\university\\master-thesis\\practical\\data\\dataset_May\\RED_colmap_alignment\\poses'                    # path where the json will be written


rgbsrcpath = output_folder
script_dir = os.path.dirname(os.path.realpath(__file__))

os.chdir(script_dir)

print(f"Current working directory: {os.getcwd()}")


def crop_to_square(input_folder, output_folder, target_size, scale_factor):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'JPG', 'PNG')):
            image_path = os.path.join(input_folder, filename)
            img = cv2.imread(image_path)

            if img is None:
                print(f"Could not read {filename}, skipping.")
                continue

            height, width = img.shape[:2]

            crop_x_start = (width - target_size) // 2
            crop_y_start = (height - target_size) // 2

            cropped_img = img[crop_y_start:crop_y_start + target_size, crop_x_start:crop_x_start + target_size]

            new_size = (int(target_size * scale_factor), int(target_size * scale_factor))
            scaled_img = cv2.resize(cropped_img, new_size, interpolation=cv2.INTER_AREA)

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, scaled_img)
            print(f"Cropped, scaled, and saved {filename} to {new_size[0]}x{new_size[1]}")




try:
    crop_to_square(input_folder, output_folder, output_resolution, scale_factor)
    print("All images processed successfully.")
except ValueError:
    print("Please enter a valid integer for the target size.")



RUN_FROM_SCRIPT = False
calibrationMatrix = None
newImageSize = None
camids = []
camFilenames = []
THERMAL = False
USE_TRANSLATION = False
TRANSLATION_SCALING = 1
Ms = None  


def quat2rotmat(qvec):
   
    qw, qx, qy, qz = qvec
    rotmat = np.array([
        [1 - 2 * qy**2 - 2 * qz**2,     2 * qx * qy - 2 * qw * qz,     2 * qx * qz + 2 * qw * qy],
        [2 * qx * qy + 2 * qw * qz,     1 - 2 * qx**2 - 2 * qz**2,     2 * qy * qz - 2 * qw * qx],
        [2 * qx * qz - 2 * qw * qy,     2 * qy * qz + 2 * qw * qx,     1 - 2 * qx**2 - 2 * qy**2]
    ])
    return rotmat

def read_model(path):
    
    if path and not path.endswith('/'):
        path = f"{path}/"

    cameras = read_cameras(os.path.join(path, 'cameras.txt'))
    images = read_images(os.path.join(path, 'images.txt'))
    points3D = read_points3D(os.path.join(path, 'points3D.txt'))

    return cameras, images, points3D

def read_cameras(path):
    cameras = {}
    with open(path, 'r') as file:
        for line in file:
            elems = line.split()
            if len(elems) < 4 or elems[0] == '#':
                continue

            if len(cameras) % 10 == 0:
                print(f"Reading camera {len(cameras)}")

            camera_id = int(elems[0])
            camera = {
                'camera_id': camera_id,
                'model': elems[1],
                'width': int(elems[2]),
                'height': int(elems[3]),
                'params': np.array([float(x) for x in elems[4:]])
            }
            cameras[camera_id] = camera

    return cameras

import json

def writeJSON(data, filename):
    """
    Write a dictionary to a JSON file after converting numpy arrays to lists.
    """
    data = convert_ndarray_to_list(data)
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

 
def read_images(path):
    images = {}
    with open(path, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break
            elems = line.split()
            if len(elems) < 4 or elems[0] == '#':
                continue

            if len(images) % 10 == 0:
                print(f"Reading image {len(images)}")

            image_id = int(elems[0])
            qw, qx, qy, qz = map(float, elems[1:5])
            image = {
                'image_id': image_id,
                'R': quat2rotmat([qw, qx, qy, qz]), 
                't': np.array([float(elems[5]), float(elems[6]), float(elems[7])]),
                'camera_id': int(elems[8]),
                'name': elems[9]
            }

            
            line = file.readline()
            point_data = np.fromstring(line, sep=' ')
            point_data = point_data.reshape(-1, 3)
            image['xys'] = point_data[:, :2]
            image['point3D_ids'] = point_data[:, 2].astype(int)

            images[image_id] = image

    return images

def read_points3D(path):
    points3D = {}
    with open(path, 'r') as file:
        for line in file:
            if not line or line.startswith('#'):
                continue
            elems = np.fromstring(line, sep=' ')
            if elems.size == 0:
                continue

            if len(points3D) % 1000 == 0:
                print(f"Reading point {len(points3D)}")

            point3D_id = int(elems[0])
            point = {
                'point3D_id': point3D_id,
                'xyz': elems[1:4],
                'rgb': elems[4:7].astype(np.uint8),
                'error': elems[7],
                'track': elems[8:].astype(int).reshape(-1, 2)
            }
            
            point['track'][:, 1] += 1

            points3D[point3D_id] = point

    return points3D

import numpy as np
import os

def createCamPosConfig_RGB(images, camids=None, filenames=None, colmapwithrectified=False):
    
    if camids is None:
        camids = list(images.keys())

    RGB = []

    for cam_id in camids:
        
        image = images[cam_id]
        image_name = image['name']
        
       
        entry = {}
        entry['imagefile'] = image_name  
        
        Rmat = np.eye(4)
        Rmat[:3, :3] = np.array(image['R']).T 
        
        Tmat = np.eye(4)
        Tmat[:3, 3] = -np.array(image['t'])  
        
        M = np.linalg.inv(Rmat @ Tmat)
        entry['M3x4'] = M[:3, :] 

       
        RGB.append(entry)

    return RGB

import numpy as np
from typing import List, Dict, Any, Tuple

def createCamPosConfig2(images: Dict[int, Dict[str, Any]], camids: List[int] = None, pairs: List[Dict[str, Any]] = None,
                        calibrationMatrix: np.ndarray = None, colmapwithrectified: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    
    if camids is None:
        camids = list(images.keys())
    
    
    if calibrationMatrix is None:
        C = np.eye(4)
    elif calibrationMatrix.shape == (3, 3):
        C = np.eye(4)
        C[:3, :3] = calibrationMatrix
    else:
        assert calibrationMatrix.shape == (4, 4)
        C = calibrationMatrix

   
    RGB = []
    thermal = []
    allInfo = []

    for cam_id in camids:
        
        image = images[cam_id]
        image_name = image['name']
        
        if colmapwithrectified:
            iname = os.path.splitext(image_name)[0]
            pairid = int(iname) + 1
        else:
            pairid = next((idx for idx, p in enumerate(pairs) if p['rgb_name'] == image_name), None)
        
        if pairid is not None:
           
            rgb_entry = {'imagefile': pairs[pairid]['rgb_name']}
            thermal_entry = {'imagefile': pairs[pairid]['thermal_name']}
            
            Rmat = np.eye(4)
            Rmat[:3, :3] = np.array(image['R']).T 
            
            Tmat = np.eye(4)
            Tmat[:3, 3] = -np.array(image['t']) 
            
            M_rgb = np.linalg.inv(Rmat @ Tmat)
            rgb_entry['M3x4'] = M_rgb[:3, :]  
            RGB.append(rgb_entry)
            
            M_thermal = np.linalg.inv(Rmat @ Tmat @ np.linalg.inv(C))
            thermal_entry['M3x4'] = M_thermal[:3, :]  
            thermal.append(thermal_entry)

            allInfo.append({
                'rgbfile': pairs[pairid]['rgb_name'],
                'thermalfile': pairs[pairid]['thermal_name'],
                'rgbfolder': pairs[pairid]['rgb_folder'],
                'thermalfolder': pairs[pairid]['thermal_folder'],
                'rgbdate': pairs[pairid]['rgb_date'],
                'thermaldate': pairs[pairid]['thermal_date'],
                'M3x4': M_rgb[:3, :],
                'R': image['R'],
                't': image['t']
            })

    return RGB, thermal, allInfo

def convert_ndarray_to_list(data):
    if isinstance(data, dict):
        return {k: convert_ndarray_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_ndarray_to_list(v) for v in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

camids_todelete = []
oldcolmapstyle = False  

if not RUN_FROM_SCRIPT:
    calibrationMatrix = None
    camids_todelete = []
    oldcolmapstyle = False

cameras, images, points = read_model(path)

image_names = []
camids = []
for i, key in enumerate(images.keys()):
    image_names.append(images[key]['name'])
    camids.append(i + 1)

if not camids and not camFilenames:
    camFilenames = sorted(image_names)

if not camids and camFilenames:
    camids = []
    for filename in camFilenames:
        if filename in image_names:
            camids.append(image_names.index(filename) + 1)

file_pattern = os.path.join(rgbsrcpath, '*.png')
jpeg_files = [f for f in os.listdir(rgbsrcpath) if f.endswith('.png')]

if camids_todelete:
    camids = [cid for cid in camids if cid not in camids_todelete]

RGB_images = createCamPosConfig_RGB(images)
RGB = {'images': RGB_images}
RGB_converted = convert_ndarray_to_list(RGB)

if not os.path.exists(outpath):
    os.makedirs(outpath)

with open(os.path.join(outpath, 'poses.json'), 'w') as f:
    json.dump(RGB_converted, f, indent=4)




