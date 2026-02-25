import os
import numpy as np
from PIL import Image
import re

DATASET_DIR = '../../data/dataset_March'

def natural_key(string):
    # Use regex to split the string into numeric and non-numeric parts
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', string)]


RED_image_paths = os.listdir(os.path.join(DATASET_DIR, 'RED_layers_cleaned_corrected'))
RED_image_paths = sorted(RED_image_paths, key=natural_key)
RED_images = [Image.open(os.path.join(DATASET_DIR, 'RED_layers_cleaned_corrected', image_path)) for image_path in RED_image_paths]

NIR_image_paths = os.listdir(os.path.join(DATASET_DIR, 'NIR_layers_cleaned_corrected'))
NIR_image_paths = sorted(NIR_image_paths, key=natural_key)
NIR_images = [Image.open(os.path.join(DATASET_DIR, 'NIR_layers_cleaned_corrected', image_path)) for image_path in NIR_image_paths]

for index, (red_img, nir_img) in enumerate(zip(RED_images, NIR_images)):
    red_array = np.array(red_img).astype(float)
    nir_array = np.array(nir_img).astype(float)
    
    # Avoid division by zero
    denominator = (nir_array + red_array)
    denominator[denominator == 0] = 1e-10
    
    ndvi_array = (nir_array - red_array) / denominator
    
    #image = Image.fromarray(ndvi_array[:,:,0].astype(np.float32), mode='F')
    #os.makedirs('../../data/dataset_October/NDVI_irradiancee_RGB/cropped', exist_ok=True)
    #image.save('../../data/dataset_October/NDVI_irradiancee_RGB/cropped/center.tiff')
    os.makedirs(os.path.join(DATASET_DIR, 'NDVI_layers'), exist_ok=True)
    image = Image.fromarray(ndvi_array.astype(np.float32), mode='F')
    image.save(os.path.join(DATASET_DIR, 'NDVI_layers', f'NDVI_layer_{index+1}.tiff'))
