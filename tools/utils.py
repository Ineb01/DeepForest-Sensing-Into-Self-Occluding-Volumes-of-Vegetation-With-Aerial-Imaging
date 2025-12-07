import re
import cv2
import math
from PIL import Image, ImageOps
import numpy as np
from scipy.ndimage import distance_transform_edt


numbers = re.compile(r'(\d+)')
def numericalSort(value):
  parts = numbers.split(value)
  parts[1::2] = map(int, parts[1::2])
  return parts

def calculate_mean(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    # mean_rgb = np.mean(image_array, axis=(0, 1))
    # Convert the image to a one-dimensional array

    # Comment the following 2 lines if you use CNN instead of MLP

    flattened_image = image_array.flatten()
    # Calculate the mean of the flattened array
    mean_value = np.mean(flattened_image)
    
    return mean_value

def pad_image_to_fit(image, target_width, target_height):
    # Calculate the padding required to reach the target dimensions
    pad_width = (target_width - image.width) % target_width
    pad_height = (target_height - image.height) % target_height
    
    # Calculate padding on each side
    padding = (0, 0, pad_width, pad_height)
    
    # Pad the image with the calculated padding
    padded_image = ImageOps.expand(image, padding)
    
    return padded_image

# def split_image_into_equal_tiles(image_path, num_tiles, target_size=(440, 440)):
#     """
#     Splits an image into equal tiles based on the specified number of tiles per dimension,
#     padding the image if necessary.

#     :param image: Input image (PIL Image).
#     :param num_tiles: Number of tiles per dimension (e.g., 2 for 2x2, 4 for 4x4).
#     :return: List of image tiles.
#     """
#     img_height, img_width = image_path.shape 

#     # Calculate the size of each tile
#     tile_width = math.ceil(img_width / num_tiles)
#     tile_height = math.ceil(img_height / num_tiles)

#     # Calculate padding required to ensure the image can be split into equal tiles
#     pad_x = (tile_width * num_tiles) - img_width
#     pad_y = (tile_height * num_tiles) - img_height
#     # Pad the image
#     image = Image.fromarray(image_path)
#     padded_image = Image.new('L', (img_width + pad_x, img_height + pad_y))
#     padded_image.paste(image, (0, 0))

#     # Split the image into tiles
#     tiles = []
#     for i in range(num_tiles):
#         for j in range(num_tiles):
#             left = j * tile_width
#             upper = i * tile_height
#             right = left + tile_width
#             lower = upper + tile_height
#             tile = padded_image.crop((left, upper, right, lower))
#             # tile.save(f'check/{index}.png')
#             # print(np.array(tile))
#             tiles.append(np.mean(np.array(tile)))  
#     tiles = np.array(tiles).reshape((num_tiles , num_tiles))
#     return tiles



def split_image_into_equal_tiles(image_path, num_tiles, target_size=(440, 440)):
    """
    Splits an image into equal tiles based on the specified number of tiles per dimension,
    padding the image symmetrically if necessary.

    :param image_path: Input image as a numpy array.
    :param num_tiles: Number of tiles per dimension (e.g., 2 for 2x2, 4 for 4x4).
    :param target_size: Target size of the original image before splitting.
    :return: A 2D numpy array where each element is the mean value of the corresponding tile.
    """
    img_height, img_width = image_path.shape  # Get dimensions of the input image

    # Calculate the size of each tile
    tile_width = math.ceil(img_width / num_tiles)
    tile_height = math.ceil(img_height / num_tiles)

    # Calculate padding required to ensure the image can be split into equal tiles
    pad_x = (tile_width * num_tiles) - img_width
    pad_y = (tile_height * num_tiles) - img_height

    # Symmetric padding (divide the padding equally on both sides)
    left_pad = pad_x // 2
    top_pad = pad_y // 2

    # Pad the image symmetrically
    # larger_matrix = np.repeat(np.repeat(np.array(image_path), num_tiles, axis=0), num_tiles, axis=1)

    # Convert to proper format for PIL
    if image_path.dtype == np.uint8:
        image = Image.fromarray(image_path, mode='L')  # 8-bit grayscale
    else:
        image = Image.fromarray(image_path.astype(np.float32), mode='F')  # 32-bit float

    # Use same mode as the input image
    mode = 'L' if image_path.dtype == np.uint8 else 'F'
    padded_image = Image.new(mode, (img_width + pad_x, img_height + pad_y))
    padded_image.paste(image, (left_pad, top_pad))

    
    # Split the image into tiles
    tiles = []
    for i in range(num_tiles):
        for j in range(num_tiles):
            left = j * tile_width
            upper = i * tile_height
            right = left + tile_width
            lower = upper + tile_height
            tile = padded_image.crop((left, upper, right, lower))
            tiles.append(np.mean(np.array(tile)))  # Calculate mean value for each tile

    # Reshape the list of tiles into a 2D array (num_tiles x num_tiles)
    tiles = np.array(tiles).reshape((num_tiles, num_tiles))

    # Find the mask where zeros are located
    image_np = np.array(tiles)

    mask = (image_np == 0)

    # Use distance transform to find the nearest non-zero neighbor for each zero pixel
    _, indices = distance_transform_edt(mask, return_indices=True)

    # Fill zeros with the nearest non-zero neighbor's value
    image_np[mask] = image_np[tuple(indices[:, mask])]

    # Convert back to PIL image
    # padded_image = Image.fromarray(image_np.astype(np.uint8))
   
    return image_np
