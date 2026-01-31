import numpy as np
import cv2
import os
import re
from PIL import Image

def natural_key(string):
    # Use regex to split the string into numeric and non-numeric parts
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', string)]


def load_image(image_path, DIR):
    """Loads an image and converts it to a numpy array."""
    return np.array(Image.open(os.path.join(DIR, image_path)))


def sensor_matcing(source_image, reference_image, target_image):
    """
    Match the contrast and brightness of a source image to a target image.
    
    Parameters:
    source_image (np.ndarray): Source grayscale image to be adjusted
    target_image (np.ndarray): Target grayscale image to match
    
    Returns:
    np.ndarray: Adjusted source image with matched contrast and brightness
    """
    
    flatten_source_image =[]
    flatten_reference_image =[]

    rows, cols = 302, 302
    for i in range(rows):
        for j in range(cols):
            pixel = source_image[i, j]  # Get the RGB pixel value
            b, g, r = pixel  # Blue, Green, Red channels
            if b != 255:
                flatten_source_image.append(pixel)

    rows, cols = 302, 302
    for i in range(rows):
        for j in range(cols):
            pixel = reference_image[i, j]  # Get the RGB pixel value
            b, g, r = pixel  # Blue, Green, Red channels
            # if pixel != 255:
            if b != 255:
                flatten_reference_image.append(pixel)
    
    # Calculate mean and standard deviation of source and target images
    source_mean = np.mean(flatten_source_image)
    source_std = np.std(flatten_source_image)
    reference_mean = np.mean(flatten_reference_image)
    reference_std = np.std(flatten_reference_image)
    
    resulting_image = (reference_std * (target_image - source_mean)) / source_std + reference_mean
    
    # scale = reference_std / source_std
    # offset = reference_mean - (scale * source_mean)
    # matched_image = cv2.convertScaleAbs(target_image, alpha=scale, beta=offset)
    return resulting_image

def process_images(source_path, reference_path):
    """
    Read grayscale images, match their contrast and brightness, and save the result.
    
    Parameters:
    source_path (str): Path to the source image to be adjusted
    target_path (str): Path to the target image to match
    output_path (str): Path to save the adjusted image
    """
    
    DIR = '../../data/dataset_May/RED_layers_cleaned'
    OUTDIR = '../../data/dataset_May/RED_layers_cleaned_corrected'
    os.makedirs(OUTDIR, exist_ok=True)
    image_paths = os.listdir(DIR)
    image_paths = sorted(image_paths, key=natural_key)
    images = [load_image(image_path, DIR) for image_path in image_paths]

    for index, img in enumerate(images):
        print(index)
        # Read images in grayscale mode
        target_image = img
        source_image = cv2.imread(source_path, -1)
        # source_image = cv2.resize(source_image, (440, 440))
        
        reference_image = cv2.imread(reference_path, -1)
        # reference_image = cv2.resize(reference_image, (440, 440))

        if source_image is None or target_image is None:
            raise FileNotFoundError("Unable to read one or both images")
        
        # Sensor Matching
        matched_image = sensor_matcing(source_image, reference_image, target_image)

        cv2.imwrite(os.path.join(OUTDIR, 'Layer_'+str(index+1)+'.png'), matched_image)

# Example usage
if __name__ == "__main__":
    process_images(
        source_path='../../data/dataset_May/RED_colmap_alignment/target.jpg',
        reference_path='../../data/dataset_May/RED_colmap_alignment/topdown_view.png',
    )
