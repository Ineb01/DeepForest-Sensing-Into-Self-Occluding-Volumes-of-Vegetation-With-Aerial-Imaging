import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import os
import cv2
import re
import time
import glob
import tifffile
import torch 
from concurrent.futures import ThreadPoolExecutor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#########################################################################
## This File extract the slices from all the layers (our PSF function) ##
#########################################################################

start_time = time.time()
# Below is a function to sort the image while loading from the folder
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def load_image_stack(directory, image_type, layer_number, axil = 1):
    # Pattern to match the desired image files
    pattern = os.path.join(directory, '*.png')
    
    # Sorted list of matching files
    file_list = sorted(glob.glob(pattern), key=numericalSort)

    file_list = file_list[layer_number - 1::axil]

    def process_image(filename):
        img = Image.open(filename).convert('L')
        image_array = np.array(img, dtype=np.float32)
        image_array = image_array / 255.0  # Normalize to [0, 1]

        return torch.tensor(image_array, dtype=torch.float32)

    # Use ThreadPoolExecutor to load and process images in parallel
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(process_image, file_list))

    # Stack all tensors into a single tensor
    return torch.stack(images)


# Save intersected planes as images
def save_intersected_plane(plane, filename):
    if plane.shape[0] != 0:
        tifffile.imwrite(filename,  (plane).astype(np.uint8))


# Create a mask from the intersection points
def create_polygon_mask(shape, polygon):
    mask = np.zeros(shape, dtype=np.uint8)
    polygon = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 1)
    return mask

# Clamp the points to image boundaries
def clamp_points(points, width, height):
    clamped_points = []
    for x, y in points:
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)
        clamped_points.append((x, y))
    return clamped_points

# Apply mask to the image slice
def apply_mask(image, mask):
    # Convert the mask and image to torch tensors if not already
    if isinstance(mask, np.ndarray):
        mask = torch.tensor(mask, dtype=torch.float32)
    if isinstance(image, np.ndarray):
        image = torch.tensor(image, dtype=torch.float32)
    return image * mask

def crop_to_nonzero(image):
    if image.is_cuda:
        image = image.cpu().numpy()
    else:
        image = image.cpu().numpy()

    coords = cv2.findNonZero(image)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]
    # return image

def slice_contribution(x_point, y_point, z_point, planes_number, image_type, remove_zeros, training, axil = 1):

    save_path = r'data/train'

    # Directory containing the image stack
    if training:
        image_stack_dir = image_type
    else:
        # This part for testing - use NIR data
        image_stack_dir = '../data/dataset_March/NIR_layers_cropped'

            

    # Load the images
    image_stack = load_image_stack(image_stack_dir, remove_zeros, z_point,  axil=axil)
    height, width = 440,440

    # Define the number of slices
    len_img = len(image_stack)
    num_planes, num_slices = len_img, len_img

    slice_height = 0.03 * axil
    
    total_height_meters = 35
    
    meters_per_slice = total_height_meters / num_slices
    
    # Define an arbitrary point at the base
    px, py = x_point, y_point  # Example coordinates
    base_point = [px, py, z_point * 0.03]

    # Define the top rectangle vertices
    top_rect_vertices = [
        [0, 0, total_height_meters],
        [width-1, 0, total_height_meters],
        [width-1, height-1, total_height_meters],
        [0, height-1, total_height_meters]
    ]


    # Plot the top rectangle
    top_rect = Poly3DCollection([top_rect_vertices], alpha=.25, linewidths=1, edgecolors='r')
    top_rect.set_facecolor((0, 1, 1, 0.1))  # Light blue color with transparency


    # Function to get the intersection points of the projection lines with each slice
    def get_intersection_points(base_point, vertex, z):
        t = (z - base_point[2]) / (vertex[2] - base_point[2])
        x = (1 - t) * base_point[0] + t * vertex[0]
        y = (1 - t) * base_point[1] + t * vertex[1]
        return [x, y, z]

    # Plot the intersection planes and save them
    top = []
    # plane_counter = 0
    for i in range(0, num_planes):
        z = (i * slice_height) + (z_point * 0.03)
        z = round(z, 2)

        if z < total_height_meters:
            intersection_points = [get_intersection_points(base_point, vertex, z) for vertex in top_rect_vertices]

            clamped_points = clamp_points([(point[0], point[1]) for point in intersection_points], width, height)

            # Create a polygon mask for the intersection plane
            mask = create_polygon_mask((height, width), clamped_points)

            plane = image_stack[i, :, :]

            masked_plane = apply_mask(plane, mask)

            # Crop the masked plane to the non-zero region
            cropped_plane = crop_to_nonzero(masked_plane)
 
            # Save the intersected plane as an image
            z_point = 1 if z_point == 0 else z_point
            top.append(cropped_plane)

    return top


