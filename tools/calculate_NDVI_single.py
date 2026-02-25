import os
import numpy as np
from PIL import Image
import re

DATASET_DIR = '../../data/dataset_October'

def natural_key(string):
    # Use regex to split the string into numeric and non-numeric parts
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', string)]
red_img = Image.open('../../data/dataset_October/RED_irradiancee_RGB/cropped/IMG_241025_105418_0000_RED.JPG')
nir_img = Image.open('../../data/dataset_October/NIR_irradiancee_RGB/cropped/IMG_241025_105418_0000_NIR.JPG')
red_array = np.array(red_img).astype(float)
nir_array = np.array(nir_img).astype(float)

# Avoid division by zero
denominator = (nir_array + red_array)
denominator[denominator == 0] = 1e-10

ndvi_array = (nir_array - red_array) / denominator

image = Image.fromarray(ndvi_array[:,:,0].astype(np.float32), mode='F')
os.makedirs('../../data/dataset_October/NDVI_irradiancee_RGB/cropped', exist_ok=True)
image.save('../../data/dataset_October/NDVI_irradiancee_RGB/cropped/center.tiff')
