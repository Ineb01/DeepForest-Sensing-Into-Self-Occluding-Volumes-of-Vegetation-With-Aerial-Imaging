from PIL import Image
import os
from .base_module import BaseProcessingModule

class HorizontalFlipStep(BaseProcessingModule):
    def action(self):
        # This step will flip images horizontally.
        # It reads images from the 'cropped' directory, flips them,
        # and saves them back to the 'cropped' directory, overwriting the original files.
        bands = [f'{band}_irradiancee_RGB' for band in self.channel_names]
        for band in bands:
            img_dir = os.path.join(self.DIR, band, 'cropped')
            
            if not os.path.exists(img_dir):
                print(f"Directory not found, skipping: {img_dir}")
                continue
                
            imgs = os.listdir(img_dir)
            imgs.sort()
            print(f"Flipping images horizontally in {img_dir}")
            for img_name in imgs:
                img_path = os.path.join(img_dir, img_name)
                try:
                    with Image.open(img_path) as img:
                        # FLIP_LEFT_RIGHT performs a horizontal flip
                        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                        flipped_img.save(img_path)
                except Exception as e:
                    print(f"Could not process {img_path}: {e}")
