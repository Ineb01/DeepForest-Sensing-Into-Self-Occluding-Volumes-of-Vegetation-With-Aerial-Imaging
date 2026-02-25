import numpy as np
import cv2
import os
import re
from PIL import Image

DATASET_DIR = '../../data/dataset_March'
BAND = 'NIR'


def natural_key(string):
    # Use regex to split the string into numeric and non-numeric parts
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', string)]


def load_image(image_path, DIR):
    """Loads an image and converts it to a numpy array."""
    return np.array(Image.open(os.path.join(DIR, image_path)))


def sensor_matcing(source_mean, source_std, reference_mean, reference_std, target_image):
    """
    Match the contrast and brightness of a source image to a target image.
    
    Parameters:
    source_mean (float): Mean pixel value of the source image
    source_std (float): Standard deviation of pixel values in the source image
    reference_mean (float): Mean pixel value of the reference image
    reference_std (float): Standard deviation of pixel values in the reference image
    
    
    Returns:
    np.ndarray: Adjusted source image with matched contrast and brightness
    """
    
    
    
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
    reference_path (str): Path to the reference image to match
    output_path (str): Path to save the adjusted image
    """
    
    DIR = os.path.join(DATASET_DIR, BAND+'_layers_cleaned')
    OUTDIR = os.path.join(DATASET_DIR, BAND+'_layers_cleaned_corrected')
    os.makedirs(OUTDIR, exist_ok=True)
    image_paths = os.listdir(DIR)
    image_paths = sorted(image_paths, key=natural_key)
    images = [load_image(image_path, DIR) for image_path in image_paths]
     
    source_image = cv2.imread(source_path, -1)
        # source_image = cv2.resize(source_image, (440, 440))
        
    reference_image = cv2.imread(reference_path, -1)
    # reference_image = cv2.resize(reference_image, (440, 440))

    if source_image is None or reference_image is None:
        raise FileNotFoundError("Unable to read one or both images")
    
    flatten_source_image =[]
    flatten_reference_image =[]

    rows, cols = 439,439
    for i in range(rows):
        for j in range(cols):
            pixel = source_image[i, j]  # Get the RGB pixel value
            #b, g, r = pixel  # Blue, Green, Red channels
            b = pixel
            if b != 255:
                flatten_source_image.append(pixel)

    rows, cols = 859, 859
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

    for index, img in enumerate(images):
        print(index)
        # Read images in grayscale mode
        target_image = img
        
        
        # Sensor Matching
        matched_image = sensor_matcing(source_mean, source_std, reference_mean, reference_std, target_image)

        cv2.imwrite(os.path.join(OUTDIR, 'Layer_'+str(index+1)+'.png'), matched_image)

# Example usage
if __name__ == "__main__":
    process_images(
        source_path=os.path.join(DATASET_DIR, BAND+'_colmap_alignment/topdown_view.png'),
        reference_path=os.path.join(DATASET_DIR, BAND+'_colmap_alignment/reference.jpg'),
    )
