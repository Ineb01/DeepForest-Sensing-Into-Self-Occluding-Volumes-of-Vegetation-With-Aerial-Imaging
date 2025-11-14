import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from .base_module import BaseProcessingModule

class SaveGrayscaleStep(BaseProcessingModule):
    def action(self):
        # Implement code to save the calibrated images using ImageJ or other Python methods.
        # This could involve using the ImageJ or PIL library to save the images as grayscale.

        bands = [f'{band}_irradiancee' for band in self.channel_names]
        for band in bands:
            imgs = os.listdir(os.path.join(self.DIR, band))

            for img in imgs:

                imagePath = os.path.join(self.DIR, band)
                imagePath = os.path.join(imagePath, img)

                # Read raw image DN values
                imageRaw=cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
                
                if imageRaw.dtype != np.uint8:
                    min_value = np.min(imageRaw)
                    max_value = np.max(imageRaw)

                    # Step 2: Handle case where all values are the same
                    if min_value == max_value:
                        image_8bit = np.zeros_like(imageRaw, dtype=np.uint8)
                    else:
                        # Step 3: Linearly scale the 32-bit values to the range [0, 255]
                        image_8bit = 255 * (imageRaw - min_value) / (max_value - min_value)
                        
                        # Step 4: Convert to 8-bit (uint8)
                        image_8bit = image_8bit.astype(np.uint8)
                else:
                    image_8bit = imageRaw        
                
                image_name = img.split('.')[0]
                            
                if os.path.exists(os.path.join(self.DIR, band+'_RGB')):
                    if os.path.exists(os.path.join(self.DIR, band+'_RGB', 'images')):
                        cv2.imwrite(os.path.join(self.DIR, band+'_RGB', 'images', image_name + '.jpg'), image_8bit)
                    else:
                        os.mkdir(os.path.join(self.DIR, band+'_RGB', 'images'))
                        cv2.imwrite(os.path.join(self.DIR, band+'_RGB', 'images', image_name + '.jpg'), image_8bit)
                else:
                    os.mkdir(os.path.join(self.DIR, band+'_RGB'))
                    os.mkdir(os.path.join(self.DIR, band+'_RGB', 'images'))
                    cv2.imwrite(os.path.join(self.DIR, band+'_RGB', 'images', image_name + '.jpg'), image_8bit)
